import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential as Seq
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from torch.utils.data import Dataset
import argparse
import os
import re
from datetime import datetime

### Graph Network Definition
class GraphNetwork(MessagePassing):
    def __init__(self, in_channels, hidden_dim, msg_dim, out_channels, aggr='add'):
        super(GraphNetwork, self).__init__(aggr=aggr)

        self.message_net = Seq(
            Linear(2 * in_channels, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, msg_dim)
        )

        self.update_net = Seq(
            Linear(msg_dim + in_channels, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, out_channels)
        )

        # Add message activations storage
        self.message_activations = None
        self._store_messages = False

    def reset_message_activations(self):
        self.message_activations = {
            'messages': [],
            'edge_index': None,
            'distances': []
            }
        self._store_messages = True

    def forward(self, x, edge_index):
        self.reset_message_activations()
        # Store edge_index for this pass
        self.message_activations['edge_index'] = edge_index.detach().clone()
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        edge_features = torch.cat([x_i, x_j], dim=1)
        messages = self.message_net(edge_features)
        # Compute distances between x_i and x_j (first ndim are positions)
        # If x_i and x_j have more than just positions, use the first ndim columns
        positions_i = x_i[:, :2]  # Default to 2D
        positions_j = x_j[:, :2]
        distances = torch.norm(positions_i - positions_j, dim=1)
        if self._store_messages:
            self.message_activations['messages'].append(messages.detach())
            self.message_activations['distances'].append(distances.detach())
        return messages

    def aggregate(self, inputs, index, dim_size=None):
        out = torch.zeros(dim_size, inputs.size(1), device=inputs.device)
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(inputs), inputs)
        return out

    def update(self, aggr_out, x):
        combined = torch.cat([x, aggr_out], dim=1)
        return self.update_net(combined)


### N-body Graph Network
class NbodyGraph(GraphNetwork):
    def __init__(self, in_channels, hidden_dim, msg_dim, out_channels, dt, nt, ndim, aggr='add'):
        super(NbodyGraph, self).__init__(in_channels, hidden_dim, msg_dim, out_channels, aggr=aggr)
        self.dt = dt
        self.nt = nt
        self.ndim = ndim

    def simple_derivative(self, g):
        return self.propagate(g.edge_index, x=g.x)

    def loss(self, g):
        pred_dv_dt = self.simple_derivative(g)[:, self.ndim:]
        return torch.sum(torch.abs(g.y - pred_dv_dt))


### Graph Utilities
def connect_all(num_nodes):
    indices = torch.combinations(torch.arange(num_nodes), with_replacement=False).T
    return torch.cat([indices, indices.flip(0)], dim=1)  # Bidirectional edges


def prepare_graph_from_simulation(simulation_index=0, time_index=0):
    x_np = positions_velocities[simulation_index, time_index]
    x = torch.tensor(x_np, dtype=torch.float32).clone()

    edge_index = connect_all(num_bodies)

    y_np = accelerations[simulation_index, time_index]
    y = torch.tensor(y_np, dtype=torch.float32).clone()

    return Data(x=x, edge_index=edge_index, y=y)


### Dataset Loader
class NBodyDataset(Dataset):
    def __init__(self, positions_velocities, accelerations):
        self.positions_velocities = positions_velocities
        self.accelerations = accelerations
        self.num_simulations, self.num_timesteps, self.num_bodies, _ = positions_velocities.shape

    def __len__(self):
        return self.num_simulations

    def __getitem__(self, index):
        sim_idx = index // self.num_timesteps
        time_idx = index % self.num_timesteps

        x_np = self.positions_velocities[sim_idx, time_idx]
        x = torch.tensor(x_np, dtype=torch.float32).clone()

        edge_index = connect_all(self.num_bodies)

        y_np = self.accelerations[sim_idx, time_idx]
        y = torch.tensor(y_np, dtype=torch.float32).clone()

        return Data(x=x, edge_index=edge_index, y=y)


def find_latest_checkpoint(checkpoint_dir, hidden_dim, msg_dim, batch_size):
    pattern = re.compile(rf"nbody_h{hidden_dim}_m{msg_dim}_b{batch_size}_e\d+.pt")
    print(f"nbody_h{hidden_dim}_m{msg_dim}_b{batch_size}_e.pt")
    max_epoch = 0
    latest_ckpt = None
    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            # Extract the epoch number after 'e' and before '.pt'
            epoch = int(fname.split('_e')[-1].split('.pt')[0])
            if epoch > max_epoch:
                max_epoch = epoch
                latest_ckpt = os.path.join(checkpoint_dir, fname)
    return latest_ckpt, max_epoch


def get_device(device_arg):
    """
    Get the appropriate device based on user argument and system availability.
    Args:
        device_arg (str): User's device choice ('cpu' or 'cuda')
    Returns:
        torch.device: The appropriate device to use
    """
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device('cpu')
        return torch.device('cuda')
    return torch.device('cpu')


def train(model, dataloader, epochs=10, lr=0.001, device='cpu', checkpoint_dir=None, model_prefix='', start_epoch=0):
    device = get_device(device)  # Convert string to torch.device
    print(f"Using device: {device}")
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in dataloader:
            batch = batch.to(device)  # Move batch to GPU/CPU
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss/len(dataloader)
        print(f"Epoch {start_epoch+epoch+1}, Loss: {avg_loss:.6f}")

        # Save checkpoint every epoch
        if checkpoint_dir and epoch%10 == 0:
            checkpoint = {
                'epoch': start_epoch+epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'hidden_dim': model.message_net[0].in_features // 2,
                'msg_dim': model.message_net[-1].out_features,
                'ndim': model.ndim,
                'dt': model.dt,
                'nt': model.nt
            }
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"{model_prefix}_e{start_epoch+epoch+1}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train N-body Graph Neural Network")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension size")
    parser.add_argument("--msg_dim", type=int, default=16, help="Message dimension size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to use for training")
    parser.add_argument("--data_path", type=str, default="nbody_simulation.npz", help="Path to the simulation data")
    parser.add_argument("--checkpoint_dir", type=str, default="Models", help="Directory to save model checkpoints")
    args = parser.parse_args()

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    # Load dataset
    loaded_data = np.load(args.data_path)
    positions_velocities = loaded_data["data"]
    accelerations = loaded_data["accelerations"]

    # Extract dataset values
    times = loaded_data["times"]
    num_bodies = int(loaded_data["num_bodies"])
    num_timesteps = int(loaded_data["num_timesteps"])
    spatial_dim = int(loaded_data["spatial_dim"])
    
    # Prepare first graph for model initialization
    test_graph = prepare_graph_from_simulation(0, 0)

    # Model prefix for checkpoint naming (no timestamp)
    model_prefix = f"nbody_h{args.hidden_dim}_m{args.msg_dim}_b{args.batch_size}"

    print(model_prefix)

    # Check for existing checkpoint
    latest_ckpt, current_epochs = find_latest_checkpoint(args.checkpoint_dir, args.hidden_dim, args.msg_dim, args.batch_size)
    print(f"Resuming from checkpoint: {latest_ckpt}" if latest_ckpt else "No checkpoint found, starting from scratch.")

    # Initialize model
    model = NbodyGraph(
        in_channels=2 * spatial_dim + 2,  # positions + velocities + 2 additional parameters
        hidden_dim=args.hidden_dim, 
        msg_dim=args.msg_dim, 
        out_channels=spatial_dim,  # forces/accelerations in each dimension
        dt=0.01, 
        nt=num_timesteps, 
        ndim=spatial_dim
    )

    # Load dataset & DataLoader
    dataset = NBodyDataset(positions_velocities, accelerations)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Train model
    train(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        model_prefix=model_prefix,
        start_epoch=current_epochs
    )

    # Test forward pass
    model = model.to(device)  # Ensure model is on the correct device
    test_graph = test_graph.to(device)  # Move test graph to the same device
    output = model(test_graph.x, test_graph.edge_index)

    print("Training completed. Model checkpoints saved in:", args.checkpoint_dir)


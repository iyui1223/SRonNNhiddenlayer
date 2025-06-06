import re
from model_util import *
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
from datetime import datetime

def prepare_graph_from_simulation(simulation_index=0, time_index=0):
    x_np = positions_velocities[simulation_index, time_index]
    x = torch.tensor(x_np, dtype=torch.float32).clone()

    edge_index = connect_all(num_bodies)

    y_np = accelerations[simulation_index, time_index]
    y = torch.tensor(y_np, dtype=torch.float32).clone()

    return Data(x=x, edge_index=edge_index, y=y)

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


def train(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu', checkpoint_dir=None, model_prefix='', start_epoch=0, patience=52, model_type='L1'):
    device = get_device(device)  # Convert string to torch.device
    print(f"Using device: {device}")
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Get loss function with model parameter for L1 regularization
    if model_type == 'L1':
        loss_fn = get_loss_function(reg_lambda=0.001)
    else:
        loss_fn = get_loss_function()
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index)
            # Pass model and batch parameters for regularization
            if model_type in ['L1', 'KL']:
                loss = loss_fn(pred, batch.y, model, batch)
            else:
                loss = loss_fn(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss/len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index)
                # Pass model parameter for L1 regularization
                if model_type in ['L1', 'KL']:
                    loss = loss_fn(pred, batch.y, model, batch)
                else:
                    loss = loss_fn(pred, batch.y)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss/len(val_loader)
        
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {avg_train_loss:.6f}")
        print(f"Validation Loss: {avg_val_loss:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            if checkpoint_dir:
                checkpoint = {
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'hidden_dim': model.message_net[0].in_features // 2,
                    'msg_dim': model.message_net[-1].out_features,
                    'ndim': model.ndim,
                    'dt': model.dt,
                    'nt': model.nt
                }
                checkpoint_path = os.path.join(
                    checkpoint_dir, 
                    f"{model_prefix}_best.pt"
                )
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save regular checkpoint every 10 epochs
        if checkpoint_dir and epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'hidden_dim': model.message_net[0].in_features // 2,
                'msg_dim': model.message_net[-1].out_features,
                'ndim': model.ndim,
                'dt': model.dt,
                'nt': model.nt
            }
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"{model_prefix}_e{epoch+1}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train N-body Graph Neural Network")
    parser.add_argument("--model_type", type=str, default="", help="Model type (regularization, loss function etc.)")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension size")
    parser.add_argument("--msg_dim", type=int, default=16, help="Message dimension size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dt", type=float, default=0.01, help="time delta for model integration")
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
    
    # Create dataset with time step interval
    dataset = NBodyDataset(positions_velocities, accelerations)
    
    # Split dataset into train and validation
    train_dataset, val_dataset = NBodyDataset.split_dataset(
        dataset
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Model prefix for checkpoint naming
    model_prefix = f"{args.model_type}_h{args.hidden_dim}_m{args.msg_dim}_b{args.batch_size}"

    # Check for existing checkpoint
    latest_ckpt, current_epochs = find_latest_checkpoint(args.checkpoint_dir, args.model_type, args.hidden_dim, args.msg_dim, args.batch_size)
    print(f"Resuming from checkpoint: {latest_ckpt}" if latest_ckpt else "No checkpoint found, starting from scratch.")

    # Initialize model
    model = NbodyGraph(
        in_channels=2 * spatial_dim + 2,  # positions + velocities + 2 additional parameters
        hidden_dim=args.hidden_dim, 
        msg_dim=args.msg_dim, 
        out_channels=spatial_dim,  # forces/accelerations in each dimension
        dt=args.dt, 
        nt=num_timesteps, 
        ndim=spatial_dim
    )

    # Train model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.learning_rate,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        model_prefix=model_prefix,
        start_epoch=current_epochs,
        model_type=args.model_type
    )

    print("Training completed. Model checkpoints saved in:", args.checkpoint_dir)


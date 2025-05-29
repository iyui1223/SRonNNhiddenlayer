import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential as Seq
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data
import numpy as np
import re
import os

class GraphNetwork(MessagePassing):
    def __init__(self, in_channels, hidden_dim, msg_dim, out_channels, aggr='add'):
        super(GraphNetwork, self).__init__(aggr=aggr)

        self.message_net = Seq(
            Linear(2 * in_channels, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, msg_dim)
        )

        self.update_net = Seq(
            Linear(msg_dim + in_channels, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, hidden_dim), ReLU(),
            Linear(hidden_dim, out_channels)
        )

        self.message_activations = None
        self._store_messages = False

    def reset_message_activations(self):
        self.message_activations = {'messages': [], 'edge_index': None, 'distances': []}
        self._store_messages = True

    def forward(self, x, edge_index):
        self.reset_message_activations()
        self.message_activations['edge_index'] = edge_index.detach().clone()
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        edge_features = torch.cat([x_i, x_j], dim=1)
        messages = self.message_net(edge_features)
        positions_i = x_i[:, :self.ndim]
        positions_j = x_j[:, :self.ndim]
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
        return self.update_net(torch.cat([x, aggr_out], dim=1))

def get_loss_function():
    return torch.nn.MSELoss()

# below are all shared components regardless to model settings 
def connect_all(num_nodes):
    indices = torch.combinations(torch.arange(num_nodes), with_replacement=False).T
    return torch.cat([indices, indices.flip(0)], dim=1)

class NbodyGraph(GraphNetwork):
    def __init__(self, in_channels, hidden_dim, msg_dim, out_channels, dt, nt, ndim, aggr='add'):
        super(NbodyGraph, self).__init__(in_channels, hidden_dim, msg_dim, out_channels, aggr=aggr)
        self.dt = dt
        self.nt = nt
        self.ndim = ndim

    def simple_derivative(self, g):
        return self.propagate(g.edge_index, x=g.x)

#    def loss(self, g):
#        pred_dv_dt = self.simple_derivative(g)[:, self.ndim:]
#        return torch.sum(torch.abs(g.y - pred_dv_dt))

class NBodyDataset(torch.utils.data.Dataset):
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
    pattern = re.compile(rf"nbody_h{hidden_dim}_m{msg_dim}_b{batch_size}_e\\d+.pt")
    max_epoch = 0
    latest_ckpt = None
    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(fname.split('_e')[-1].split('.pt')[0])
            if epoch > max_epoch:
                max_epoch = epoch
                latest_ckpt = os.path.join(checkpoint_dir, fname)
    return latest_ckpt, max_epoch

def get_device(device_arg):
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device('cpu')
        return torch.device('cuda')
    return torch.device('cpu')

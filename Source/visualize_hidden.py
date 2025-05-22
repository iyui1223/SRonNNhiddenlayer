import argparse
import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from celluloid import Camera
from scipy.optimize import minimize
from copy import copy
from torch_geometric.data import Data
from train import NbodyGraph  # Update path if needed

def single_graph(model, graph_data): 
    model.eval()
    with torch.no_grad():
        _ = model(graph_data.x, graph_data.edge_index)
    activations = model.message_activations
    messages = activations['messages']
    distances = activations['distances']
    edge_index = activations['edge_index']

    # Get the message tensor from the list
    if isinstance(messages, list):
        messages_tensor = messages[0]  # Shape: [num_edges, msg_dim]
    if isinstance(distances, list):
        distances_tensor = distances[0]  # Shape: [num_edges]
    
    # Get node positions and accelerations
    if hasattr(graph_data, 'y') and graph_data.y is not None:
        ndim = model.ndim  # Get number of dimensions from model
        # Get positions and accelerations
        positions = graph_data.x[:, :ndim].cpu().numpy()  # Shape: [num_nodes, ndim]
        accelerations = graph_data.y.cpu().numpy()     # Shape: [num_nodes, ndim]
        
        # Get source and target nodes for each edge
        src_nodes = edge_index[0].cpu().numpy()
        tgt_nodes = edge_index[1].cpu().numpy()
        
        # Calculate edge direction vectors
        edge_vecs = positions[tgt_nodes] - positions[src_nodes]  # Shape: [num_edges, ndim]
        edge_vecs_norm = np.linalg.norm(edge_vecs, axis=1, keepdims=True) + 1e-8
        edge_dirs = edge_vecs / edge_vecs_norm  # Normalized direction vectors
        
        # Get accelerations of source nodes
        src_accels = accelerations[src_nodes]  # Shape: [num_edges, ndim]
        
        # Project source node accelerations onto edge directions
        force_along_edge = np.sum(src_accels * edge_dirs, axis=1)  # Shape: [num_edges]
        
        # Get message strengths (maximum absolute value across message dimensions)
        message_strengths = torch.max(torch.abs(messages_tensor), dim=1)[0].cpu().numpy()  # Shape: [num_edges]
        
        return {
            'force_along_edge': force_along_edge,
            'message_strengths': message_strengths,
            'edge_dirs': edge_dirs,
            'src_accels': src_accels,
            'positions': positions,
            'edge_index': edge_index.cpu().numpy(),
            'messages': messages_tensor.cpu().numpy(),
            'distances': distances_tensor.cpu().numpy()
        }
    else:
        print("No acceleration data available")
        return None

def create_graphs(npz_path):
    loaded_data = np.load(npz_path)
    positions_velocities = loaded_data["data"]
    accelerations = loaded_data["accelerations"]
    num_simulations, num_timesteps, num_bodies, _ = positions_velocities.shape

    def connect_all(num_nodes):
        indices = torch.combinations(torch.arange(num_nodes), with_replacement=False).T
        return torch.cat([indices, indices.flip(0)], dim=1)  # Bidirectional edges

    graphs = []
    for sim in range(300): #num_simulations):
        for t in range(1):  # Only take first timestep
            x_np = positions_velocities[sim, t]
            x = torch.tensor(x_np, dtype=torch.float32)
            edge_index = connect_all(num_bodies)
            y_np = accelerations[sim, t]
            y = torch.tensor(y_np, dtype=torch.float32)
            graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs

def parse_model_params(model_path):
    filename = os.path.basename(model_path)
    match = re.search(r'nbody_h(\d+)_m(\d+)', filename)
    if not match:
        raise ValueError(f"Could not parse model parameters from: {filename}")
    hidden_dim, msg_dim = map(int, match.groups())
    return hidden_dim, msg_dim


def percentile_sum(x):
    x = x.ravel()
    bot = x.min()
    top = np.percentile(x, 90)
    msk = (x >= bot) & (x <= top)
    frac_good = (msk).sum() / len(x)
    return x[msk].sum() / frac_good


def scatter_all_force_message(messages_over_time, msg_dim, dim=2, sim='spring', title="GNN Force Matching"):
    pos_cols = ['dx', 'dy', 'dz'][:dim]  # Handle both 2D and 3D
    fig, axes = plt.subplots(1, dim, figsize=(5 * dim, 5))

    # Handle both single and multiple axes
    if dim == 1:
        axes = [axes]

    all_force_proj = [[] for _ in range(dim)]
    all_msg_comp = [[] for _ in range(dim)]

    for msgs in messages_over_time:
        msgs = copy(msgs)
        msgs['bd'] = msgs.r + 1e-2

        msg_columns = [f"e{k+1}" for k in range(msg_dim)]
        msg_array = np.array(msgs[msg_columns])
        msg_importance = msg_array.std(axis=0)
        most_important = np.argsort(msg_importance)[-dim:]
        msgs_to_compare = msg_array[:, most_important]
        msgs_to_compare = (msgs_to_compare - np.mean(msgs_to_compare, axis=0)) / np.std(msgs_to_compare, axis=0)

        force_fnc = lambda msg: -(msg['bd'].to_numpy() - 1)[:, None] * msg[pos_cols].to_numpy() / msg['bd'].to_numpy()[:, None]
        expected_forces = force_fnc(msgs)

        def linear_transformation(alpha):
            if dim == 2:
                lin1 = alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1] + alpha[2]
                lin2 = alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1] + alpha[5]
                return (
                    percentile_sum((msgs_to_compare[:, 0] - lin1) ** 2) +
                    percentile_sum((msgs_to_compare[:, 1] - lin2) ** 2)
                ) / 2.0
            else:  # dim == 3
                lin1 = alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1] + alpha[2] * expected_forces[:, 2] + alpha[3]
                lin2 = alpha[4] * expected_forces[:, 0] + alpha[5] * expected_forces[:, 1] + alpha[6] * expected_forces[:, 2] + alpha[7]
                lin3 = alpha[8] * expected_forces[:, 0] + alpha[9] * expected_forces[:, 1] + alpha[10] * expected_forces[:, 2] + alpha[11]
                return (
                    percentile_sum((msgs_to_compare[:, 0] - lin1) ** 2) +
                    percentile_sum((msgs_to_compare[:, 1] - lin2) ** 2) +
                    percentile_sum((msgs_to_compare[:, 2] - lin3) ** 2)
                ) / 3.0

        def out_linear_transformation(alpha):
            if dim == 2:
                lin1 = alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1] + alpha[2]
                lin2 = alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1] + alpha[5]
                return lin1, lin2
            else:  # dim == 3
                lin1 = alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1] + alpha[2] * expected_forces[:, 2] + alpha[3]
                lin2 = alpha[4] * expected_forces[:, 0] + alpha[5] * expected_forces[:, 1] + alpha[6] * expected_forces[:, 2] + alpha[7]
                lin3 = alpha[8] * expected_forces[:, 0] + alpha[9] * expected_forces[:, 1] + alpha[10] * expected_forces[:, 2] + alpha[11]
                return lin1, lin2, lin3

        # Initialize parameters based on dimension
        init_params = np.ones(dim * (dim + 1))  # dim^2 + dim parameters
        min_result = minimize(linear_transformation, init_params, method='Powell')
        lincombs = out_linear_transformation(min_result.x)

        for i in range(dim):
            all_force_proj[i].append(lincombs[i])
            all_msg_comp[i].append(msgs_to_compare[:, i])

    for i in range(dim):
        px = np.concatenate(all_force_proj[i])
        py = np.concatenate(all_msg_comp[i])
        axes[i].scatter(px, py, s=1, alpha=0.1, color='black')
        axes[i].set_xlabel('Linear combination of forces')
        axes[i].set_ylabel(f'Message Element {i+1}')
        axes[i].set_title(f'Message {i+1} vs Force Projection')
        axes[i].grid(True)
        axes[i].set_xlim(-1, 1)
        axes[i].set_ylim(-1, 1)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("force_message_relation.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ndim", type=int, required=True, help="Number of spatial dimensions")
    parser.add_argument("--dt", type=float, required=True, help="Time step size")
    args = parser.parse_args()

    hidden_dim, msg_dim = parse_model_params(args.model_path)

    model = NbodyGraph(
        in_channels=2 * args.ndim + 2,  # positions + velocities + 2 additional parameters
        hidden_dim=hidden_dim,
        msg_dim=msg_dim,
        out_channels=args.ndim,  # forces/accelerations in each dimension
        dt=args.dt,
        nt=1,
        ndim=args.ndim
    )

    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    graphs = create_graphs(args.data_path)

    all_data = [single_graph(model, g) for g in graphs if single_graph(model, g) is not None]
    print(f"Processed {len(all_data)} simulations")

    messages_over_time = []
    for data in all_data:
        msgs = data['messages']
        df = pd.DataFrame(msgs, columns=[f"e{i+1}" for i in range(msgs.shape[1])])
        df['r'] = data['distances']
        df['bd'] = data['distances'] + 1e-2
        # Add position columns based on dimension
        for i in range(args.ndim):
            df[f'd{"xyz"[i]}'] = data['edge_dirs'][:, i]
        messages_over_time.append(df)

    scatter_all_force_message(messages_over_time, msg_dim=msg_dim, dim=args.ndim)


if __name__ == "__main__":
    main()

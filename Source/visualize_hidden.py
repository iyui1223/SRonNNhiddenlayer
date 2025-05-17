import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from typing import Dict, Any
import os
from copy import copy
import argparse
from train import NbodyGraph

# obtain the prediction and the latent expression of single time step
# for the given list of initial values
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
        # Get positions (first 2 columns of x) and accelerations
        positions = graph_data.x[:, :2].cpu().numpy()  # Shape: [num_nodes, 2]
        accelerations = graph_data.y.cpu().numpy()     # Shape: [num_nodes, 2]
        
        # Get source and target nodes for each edge
        src_nodes = edge_index[0].cpu().numpy()
        tgt_nodes = edge_index[1].cpu().numpy()
        
        # Calculate edge direction vectors
        edge_vecs = positions[tgt_nodes] - positions[src_nodes]  # Shape: [num_edges, 2]
        edge_vecs_norm = np.linalg.norm(edge_vecs, axis=1, keepdims=True) + 1e-8
        edge_dirs = edge_vecs / edge_vecs_norm  # Normalized direction vectors
        
        # Get accelerations of source nodes
        src_accels = accelerations[src_nodes]  # Shape: [num_edges, 2]
        
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
    for sim in range(num_simulations):
        for t in range(1):  # Only take first timestep
            x_np = positions_velocities[sim, t]
            x = torch.tensor(x_np, dtype=torch.float32)
            edge_index = connect_all(num_bodies)
            y_np = accelerations[sim, t]
            y = torch.tensor(y_np, dtype=torch.float32)
            graphs.append(Data(x=x, edge_index=edge_index, y=y))
    return graphs



def parse_model_params(model_path):
    """
    Extract hidden_dim and msg_dim from model_path using naming pattern
    e.g., 'nbody_h32_m16_b1_e91.pt' → hidden_dim=32, msg_dim=16
    """
    import re
    from pathlib import Path
    filename = Path(model_path).name
    match = re.search(r'nbody_h(\d+)_m(\d+)', filename)
    if not match:
        raise ValueError(f"Could not parse model parameters from: {filename}")
    hidden_dim, msg_dim = map(int, match.groups())
    return hidden_dim, msg_dim

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze message passing in N-body GNN")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Output directory")

    args = parser.parse_args()

    # Extract parameters from filename
    hidden_dim, msg_dim = parse_model_params(args.model_path)

    # Instantiate model using parsed parameters
    model = NbodyGraph(
        in_channels=6,
        hidden_dim=hidden_dim,
        msg_dim=msg_dim,
        out_channels=2, 
        dt=0.01,
        nt=1,
        ndim=2
    )

    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    trajectory_data = create_graphs(args.data_path)

    # Collect data from all simulations
    all_data = []
    
    for graph in trajectory_data[:]:  # Use all simulations
        data = single_graph(model, graph)
        if data is not None:
            all_data.append(data)
    
    print(f"Processing {len(all_data)} simulations at t=0")
    
    # Create directory for plots
    os.makedirs('analysis_plots', exist_ok=True)
    
    # Create output directory for debug data
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Aggregate all messages and forces from all simulations
    all_msgs = []
    all_forces = []
    all_edge_dirs = []
    all_bd = []
    all_edge_index = []
    all_positions = []
    
    for data in all_data:
        all_msgs.append(data['messages'])         # [num_edges, msg_dim]
        all_forces.append(data['src_accels'])     # 
        all_edge_dirs.append(data['edge_dirs'])
        all_bd.append(data['distances'] + 1e-2)
        all_edge_index.append(data['edge_index'])
        all_positions.append(data['positions'])
    
    all_msgs = np.vstack(all_msgs)        # [total_edges, msg_dim]
    all_forces = np.vstack(all_forces)    # [total_edges, 2]
    all_edge_dirs = np.vstack(all_edge_dirs)
    all_bd = np.concatenate(all_bd)
    all_edge_index = np.hstack(all_edge_index)
    all_positions = np.vstack(all_positions)
    
    # Calculate message importance using standard deviation globally
    # print(all_msgs.shape)
    msg_importance = all_msgs.std(axis=0)
    # print(msg_importance)
    most_important = np.argsort(msg_importance)[-2:]  # Get top 2 dimensions for 2D visualization
    # print(msg_importance[most_important])
    msgs_to_compare = all_msgs[:, most_important]
    
    # Normalize messages
    msgs_to_compare = (msgs_to_compare - np.average(msgs_to_compare, axis=0)) / np.std(msgs_to_compare, axis=0)
    
    # Calculate electrostatic forces
    k = 1.0  # Coulomb constant
    charges = all_positions[:, -1]  # Assuming charge is stored in the last column
    
    # Save debugging information
    debug_data = {
        'dx': all_edge_dirs[:, 0],  # x-component of direction vectors
        'dy': all_edge_dirs[:, 1],  # y-component of direction vectors
        'source_charges': charges[all_edge_index[0]],  # charges of source nodes
        'target_charges': charges[all_edge_index[1]],  # charges of target nodes
        'msgs_to_compare': msgs_to_compare,  # the two most important message components
        'distances': all_bd,  # distances between nodes
        'edge_index': all_edge_index,  # edge connectivity
        'positions': all_positions  # node positions
    }
    
    # Save debugging data
    debug_file = os.path.join(args.output_dir, 'debug_data.npz')
    np.savez(debug_file, **debug_data)
    print(f"Debug data saved to {debug_file}")
    
    # Calculate electrostatic forces
    force_fnc = lambda msg: -(msg['bd'][:, None] - 1) **2 * np.array(msg['edge_dirs'])
#    force_fnc = lambda msg: -msg['bd'][:, None] * np.array(msg['edge_dirs']) 
    expected_forces = force_fnc({'edge_index': all_edge_index, 'edge_dirs': all_edge_dirs, 'bd': all_bd})
    
    def percentile_sum(x):
        x = x.ravel()
        bot = x.min()
        top = np.percentile(x, 90)
        msk = (x>=bot) & (x<=top)
        frac_good = (msk).sum()/len(x)
        return x[msk].sum()/frac_good
    
    from scipy.optimize import minimize
    
    def linear_transformation_2d(alpha):
        lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1]) + alpha[2]
        lincomb2 = (alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1]) + alpha[5]

        score = (
            percentile_sum(np.square(msgs_to_compare[:, 0] - lincomb1)) +
            percentile_sum(np.square(msgs_to_compare[:, 1] - lincomb2))
        )/2.0
        
        return score
    
    def out_linear_transformation_2d(alpha):
        lincomb1 = (alpha[0] * expected_forces[:, 0] + alpha[1] * expected_forces[:, 1]) + alpha[2]
        lincomb2 = (alpha[3] * expected_forces[:, 0] + alpha[4] * expected_forces[:, 1]) + alpha[5]
        return lincomb1, lincomb2
    
    # Find best linear transformation
    min_result = minimize(linear_transformation_2d, np.ones(6), method='Powell')
    print(f"Optimization score: {min_result.fun/len(all_msgs)}")
    
    # Plot results
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(2):
        px = out_linear_transformation_2d(min_result.x)[i]
        py = msgs_to_compare[:, i]
        
        # Use larger points and higher alpha for better visibility
        ax[i].scatter(px, py, alpha=0.5, s=10, color='k')
        ax[i].set_xlabel('Linear combination of forces')
        ax[i].set_ylabel(f'Message Element {i+1}')
        
        # Set plot limits using percentiles
        xlim = np.array([np.percentile(px, q) for q in [1, 99]])  # Use wider range
        ylim = np.array([np.percentile(py, q) for q in [1, 99]])  # Use wider range
        
        # Add some padding
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        xlim[0] -= x_range * 0.1
        xlim[1] += x_range * 0.1
        ylim[0] -= y_range * 0.1
        ylim[1] += y_range * 0.1
        
        ax[i].set_xlim(xlim)
        ax[i].set_ylim(ylim)
        
        # Add grid for better visibility
        ax[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_plots/force_vs_message_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    
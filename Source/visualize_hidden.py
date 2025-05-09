import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from typing import Dict, Any
import os

# obtain the prediction and the latent expression of single time step
# for the given list of initial values
def single_graph(model, graph_data): 
    model.eval()
    with torch.no_grad():
        _ = model(graph_data.x, graph_data.edge_index)
    activations = model.message_activations # hidden layer activations
    messages = activations['messages'] #  num_body x num_body message matrix -- messagesp[i,j] corresponds to message x_i -> x_j
    distances = activations['distances']
    edge_index = activations['edge_index']

    # Get the message tensor from the list
    if isinstance(messages, list):
        messages_tensor = messages[0]  # Shape: [num_edges, msg_dim]
    
    # Find the strongest message (in absolute value) for each edge
    message_strengths = torch.max(torch.abs(messages_tensor), dim=1)[0]  # Shape: [num_edges]
    
    # Get recipient nodes from edge_index
    recipient_nodes = edge_index[1]  # Shape: [num_edges]
    
    # For each recipient node, find the strongest incoming message
    num_nodes = graph_data.x.shape[0]
    strongest_messages = torch.zeros(num_nodes)
    for node in range(num_nodes):
        # Get indices of edges where this node is the recipient
        node_edge_indices = (recipient_nodes == node)
        if node_edge_indices.any():
            # Get the strongest message among incoming messages
            strongest_messages[node] = message_strengths[node_edge_indices].max()
    
    # Get accelerations for each node
    if hasattr(graph_data, 'y') and graph_data.y is not None:
        accelerations = graph_data.y.cpu().numpy()  # Shape: [num_nodes, 2]
        # Calculate magnitude of acceleration for each node
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=1)
        
        # Convert strongest messages to numpy for plotting
        strongest_messages_np = strongest_messages.cpu().numpy()
        
        return acceleration_magnitudes, strongest_messages_np
    else:
        print("No acceleration data available")
        return None, None

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

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze message passing in N-body GNN")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="analysis_output", 
                       help="Directory to save analysis outputs")
    
    args = parser.parse_args()
    
    # Load model and data
    from train import NbodyGraph  # If possible, import from train.py

    # Set the same parameters as used in training
    model = NbodyGraph(
        in_channels=6,
        hidden_dim=32,
        msg_dim=16,
        out_channels=2, 
        dt=0.01,
        nt=100,             # number of timesteps, adjust as needed
        ndim=2              # spatial dimension, adjust as needed
    )
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    trajectory_data = create_graphs(args.data_path)

    # Collect data from all simulations
    all_accelerations = []
    all_messages = []
    
    for graph in trajectory_data:
        accel, msg = single_graph(model, graph)
        if accel is not None and msg is not None:
            all_accelerations.extend(accel)
            all_messages.extend(msg)
    
    # Create scatter plot with all data
    plt.figure(figsize=(10, 6))
    plt.scatter(all_accelerations, all_messages, alpha=0.3, s=10)  # Smaller points, more transparent
    
    # Add correlation coefficient
    corr = np.corrcoef(all_accelerations, all_messages)[0,1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Acceleration Magnitude')
    plt.ylabel('Strongest Incoming Message')
    plt.title('Strongest Incoming Message vs Node Acceleration (All Simulations)')
    
    # Add a trend line
    z = np.polyfit(all_accelerations, all_messages, 1)
    p = np.poly1d(z)
    plt.plot(np.sort(all_accelerations), p(np.sort(all_accelerations)), 
             "r--", alpha=0.8, label=f'Trend line (slope: {z[0]:.3f})')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('message_vs_acceleration_all_sims.png')
    plt.close()
    
    print(f"Total number of data points: {len(all_accelerations)}")
    print(f"Correlation between strongest messages and acceleration: {corr:.3f}")


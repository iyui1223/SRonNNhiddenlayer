# latentexp.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from typing import Dict, Any
import os

class MessageActivationAnalyzer:
    def __init__(self, output_dir: str = "analysis_output"):
        """
        Initialize the analyzer with an output directory for plots.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def analyze_single_graph(self, 
                           model: torch.nn.Module, 
                           graph_data: Data, 
                           timestep: int = 0,
                           initial_values: np.ndarray = None) -> Dict[str, Any]:
        """
        Analyze message passing activations for a single graph.
        """
        model.eval()
        with torch.no_grad():
            _ = model(graph_data.x, graph_data.edge_index)
        activations = model.message_activations
        if activations is None:
            raise ValueError("No activations stored. Make sure model is modified to store activations.")
        messages = activations['messages']
        distances = activations['distances']
        edge_index = activations['edge_index']
        if isinstance(messages, list):
            messages = torch.cat(messages, dim=0)
        if isinstance(distances, list):
            distances = torch.cat(distances, dim=0)
        message_strengths = torch.max(messages, dim=1)[0]
        distances_np = distances.cpu().numpy()
        strengths_np = message_strengths.cpu().numpy()
        # Extract acceleration (per node)
        if hasattr(graph_data, 'y') and graph_data.y is not None:
            acceleration_np = graph_data.y.cpu().numpy()
        else:
            acceleration_np = None
        # Save message strengths and acceleration to .npz
        np.savez(os.path.join(self.output_dir, f"t{timestep}_msg_accel.npz"),
                 message_strengths=strengths_np, acceleration=acceleration_np)
        # Plot acceleration vs. message strength (only for final timestep)
        if timestep == -1 or initial_values is not None:
            if acceleration_np is not None:
                plt.figure(figsize=(8,6))
                # If acceleration is per node and message_strengths is per edge, just plot their distributions
                plt.scatter(np.arange(len(strengths_np)), strengths_np, label='Message Strength (max component)', alpha=0.5)
                plt.scatter(np.arange(len(acceleration_np)), np.linalg.norm(acceleration_np, axis=1), label='Acceleration (norm)', alpha=0.5)
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.title('Message Strengths vs. Acceleration (final timestep)')
                plt.legend()
                plt.savefig(os.path.join(self.output_dir, f"acceleration_vs_message_strength_t{timestep}.png"))
                plt.close()
        self._plot_distance_strength(distances_np, strengths_np, timestep, initial_values)
        stats = {
            'distance_strength_corr': np.corrcoef(distances_np, strengths_np)[0,1],
            'max_strength': message_strengths.max().item(),
            'min_strength': message_strengths.min().item(),
            'mean_strength': message_strengths.mean().item(),
            'std_strength': message_strengths.std().item()
        }
        return stats

    def _plot_distance_strength(self, 
                              distances: np.ndarray, 
                              strengths: np.ndarray, 
                              timestep: int,
                              initial_values: np.ndarray = None):
        plt.figure(figsize=(10, 6))
        plt.scatter(distances, strengths, alpha=0.5, label='Final timestep')
        z = np.polyfit(distances, strengths, 1)
        p = np.poly1d(z)
        plt.plot(distances, p(distances), "r--", alpha=0.8)
        correlation = np.corrcoef(distances, strengths)[0,1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes)
        plt.xlabel('Distance between particles')
        plt.ylabel('Message strength')
        plt.title(f'Message Strength vs Distance (t={timestep})')
        if initial_values is not None:
            # initial_values should be a (N, 2) array: [distance, strength] for all initial values
            plt.scatter(initial_values[:, 0], initial_values[:, 1], alpha=0.5, color='g', label='Initial values')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'message_strength_vs_distance_t{timestep}.png'))
        plt.close()

    def analyze_trajectory(self, 
                         model: torch.nn.Module, 
                         trajectory_data: list[Data], 
                         save_summary: bool = True):
        """
        Analyze message passing over a trajectory of graphs.
        Only plot the final timestep, but scatter all initial values in the same figure.
        """
        # Collect all initial values (from the first graph of each simulation)
        # Assume trajectory_data is a list of Data objects, one per (sim, t)
        # We'll assume the first 100 are t=0 for each sim, the last 100 are t=final for each sim
        # If not, user may need to adapt this logic
        num_sims = 100
        initial_node_strengths = []
        initial_accelerations = []
        initial_forces = []
        initial_strongest_messages = []
        for i in range(num_sims):
            graph = trajectory_data[i]  # t=0 for sim i
            model.eval()
            with torch.no_grad():
                _ = model(graph.x, graph.edge_index)
            activations = model.message_activations
            messages = activations['messages']
            distances = activations['distances']
            edge_index = activations['edge_index']
            if isinstance(messages, list):
                messages = torch.cat(messages, dim=0)
            if isinstance(distances, list):
                distances = torch.cat(distances, dim=0)
            message_strongest = torch.max(messages, dim=1)[0].cpu().numpy()  # (num_edges,)
            # For each edge, get the force from the source node
            if hasattr(graph, 'y') and graph.y is not None:
                accel = graph.y.cpu().numpy()  # (num_nodes, ndim)
                x_pos = graph.x.cpu().numpy()[:, :accel.shape[1]]  # positions (assume first ndim columns)
                src_nodes = edge_index[0].cpu().numpy()
                tgt_nodes = edge_index[1].cpu().numpy()
                # Edge direction vectors
                edge_vecs = x_pos[tgt_nodes] - x_pos[src_nodes]  # (num_edges, ndim)
                edge_vecs_norm = np.linalg.norm(edge_vecs, axis=1, keepdims=True) + 1e-8
                edge_dirs = edge_vecs / edge_vecs_norm  # normalized direction
                # Project force/accel vector of source node onto edge direction
                src_forces = accel[src_nodes]  # (num_edges, ndim)
                force_along_edge = np.sum(src_forces * edge_dirs, axis=1)  # (num_edges,)
                initial_forces.append(force_along_edge)
                initial_strongest_messages.append(message_strongest)
            # Aggregate message strengths per node (sum of incoming messages)
            num_nodes = graph.x.shape[0]
            node_strengths = np.zeros(num_nodes)
            for idx, tgt in enumerate(edge_index[1].cpu().numpy()):
                node_strengths[tgt] += messages[idx].cpu().item()
            initial_node_strengths.append(node_strengths)
            # Get acceleration for each node (can be positive or negative, use x and y components)
            if hasattr(graph, 'y') and graph.y is not None:
                accel = graph.y.cpu().numpy()  # shape (num_nodes, ndim)
                initial_accelerations.append(accel)
        initial_node_strengths = np.concatenate(initial_node_strengths)
        initial_accelerations = np.concatenate(initial_accelerations)
        initial_forces = np.concatenate(initial_forces)
        initial_strongest_messages = np.concatenate(initial_strongest_messages)
        # Plot scatter: message strength vs acceleration (x and y components)
        if initial_accelerations.shape[1] >= 2:
            plt.figure(figsize=(8,6))
            plt.scatter(initial_node_strengths, initial_accelerations[:,0], alpha=0.5, label='x-acceleration')
            plt.scatter(initial_node_strengths, initial_accelerations[:,1], alpha=0.5, label='y-acceleration')
            plt.xlabel('Aggregated Message Strength (per node)')
            plt.ylabel('Acceleration')
            plt.title('Initial: Message Strength vs Acceleration (per node)')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'initial_message_strength_vs_acceleration.png'))
            plt.close()
        else:
            plt.figure(figsize=(8,6))
            plt.scatter(initial_node_strengths, initial_accelerations[:,0], alpha=0.5, label='acceleration')
            plt.xlabel('Aggregated Message Strength (per node)')
            plt.ylabel('Acceleration')
            plt.title('Initial: Message Strength vs Acceleration (per node)')
            plt.legend()
            plt.savefig(os.path.join(self.output_dir, 'initial_message_strength_vs_acceleration.png'))
            plt.close()

        # Only analyze the final timestep (last graph in trajectory_data)
        final_graph = trajectory_data[-1]
        self.analyze_single_graph(model, final_graph, len(trajectory_data)-1)

        if initial_forces and initial_strongest_messages:
            plt.figure(figsize=(8,6))
            plt.scatter(initial_forces, initial_strongest_messages, alpha=0.5)
            plt.xlabel('Force (linear combination along edge direction)')
            plt.ylabel('Strongest Message Value (per edge)')
            plt.title('Initial: Force vs Strongest Message Value (per edge)')
            plt.savefig(os.path.join(self.output_dir, 'initial_force_vs_strongest_message.png'))
            plt.close()

def create_graphs(npz_path):
    loaded_data = np.load(npz_path)
    positions_velocities = loaded_data["data"]
    accelerations = loaded_data["accelerations"]
    num_simulations, num_timesteps, num_bodies, _ = positions_velocities.shape

    def connect_all(num_nodes):
        indices = torch.combinations(torch.arange(num_nodes), with_replacement=False).T
        return torch.cat([indices, indices.flip(0)], dim=1)  # Bidirectional edges

    graphs = []
    for sim in range(100):
        for t in range(1):
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

    # Create analyzer and run analysis
    analyzer = MessageActivationAnalyzer(args.output_dir)
    analyzer.analyze_trajectory(model, trajectory_data)
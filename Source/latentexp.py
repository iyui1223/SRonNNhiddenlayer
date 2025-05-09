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
                           timestep: int = 0) -> Dict[str, Any]:
        """
        Analyze message passing activations for a single graph.
        
        Args:
            model: Trained GraphNetwork model
            graph_data: A single graph Data object
            timestep: Current timestep for labeling
        """
        # Ensure model is in eval mode
        model.eval()
        
        # Forward pass to get activations
        with torch.no_grad():
            _ = model(graph_data.x, graph_data.edge_index)
        
        # Get stored activations
        activations = model.message_activations
        if activations is None:
            raise ValueError("No activations stored. Make sure model is modified to store activations.")
            
        messages = activations['messages']
        distances = activations['distances']
        edge_index = activations['edge_index']
        
        # Calculate message strengths
        message_strengths = torch.norm(messages, dim=1)
        
        # Convert to numpy
        distances_np = distances.cpu().numpy()
        strengths_np = message_strengths.cpu().numpy()
        
        # Generate visualizations
        self._plot_distance_strength(distances_np, strengths_np, timestep)
        self._plot_interaction_heatmap(edge_index, message_strengths, timestep)
        
        # Calculate statistics
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
                              timestep: int):
        """Plot relationship between distance and message strength."""
        plt.figure(figsize=(10, 6))
        plt.scatter(distances, strengths, alpha=0.5)
        
        # Add trend line
        z = np.polyfit(distances, strengths, 1)
        p = np.poly1d(z)
        plt.plot(distances, p(distances), "r--", alpha=0.8)
        
        # Calculate and display correlation
        correlation = np.corrcoef(distances, strengths)[0,1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes)
        
        plt.xlabel('Distance between particles')
        plt.ylabel('Message strength')
        plt.title(f'Message Strength vs Distance (t={timestep})')
        
        plt.savefig(os.path.join(self.output_dir, f'message_strength_vs_distance_t{timestep}.png'))
        plt.close()
    
    def _plot_interaction_heatmap(self, 
                                edge_index: torch.Tensor, 
                                message_strengths: torch.Tensor, 
                                timestep: int):
        """Plot heatmap of pairwise particle interactions."""
        num_nodes = len(set(edge_index[0].cpu().numpy()))
        interaction_matrix = np.zeros((num_nodes, num_nodes))
        
        for i in range(len(edge_index[0])):
            src = edge_index[0][i].item()
            dst = edge_index[1][i].item()
            interaction_matrix[src, dst] = message_strengths[i].item()
        
        plt.figure(figsize=(8, 8))
        sns.heatmap(interaction_matrix, cmap='viridis')
        plt.title(f'Pairwise Interaction Strengths (t={timestep})')
        plt.xlabel('To Particle')
        plt.ylabel('From Particle')
        
        plt.savefig(os.path.join(self.output_dir, f'interaction_heatmap_t{timestep}.png'))
        plt.close()

    def analyze_trajectory(self, 
                         model: torch.nn.Module, 
                         trajectory_data: list[Data], 
                         save_summary: bool = True):
        """
        Analyze message passing over a trajectory of graphs.
        
        Args:
            model: Trained GraphNetwork model
            trajectory_data: List of graph Data objects representing a trajectory
            save_summary: Whether to save summary statistics
        """
        all_stats = []
        for t, graph in enumerate(trajectory_data):
            stats = self.analyze_single_graph(model, graph, t)
            all_stats.append(stats)
            
        if save_summary:
            self._save_trajectory_summary(all_stats)
    
    def _save_trajectory_summary(self, stats_list: list[Dict[str, float]]):
        """Save summary statistics for the trajectory."""
        # Convert stats to arrays for easier analysis
        stats_arrays = {
            key: np.array([stats[key] for stats in stats_list])
            for key in stats_list[0].keys()
        }
        
        # Plot evolution of statistics over time
        for key, values in stats_arrays.items():
            plt.figure(figsize=(10, 6))
            plt.plot(values)
            plt.title(f'Evolution of {key}')
            plt.xlabel('Timestep')
            plt.ylabel(key)
            plt.savefig(os.path.join(self.output_dir, f'evolution_{key}.png'))
            plt.close()

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
    model = torch.load(args.model_path)
    data = torch.load(args.data_path)
    
    # Create analyzer and run analysis
    analyzer = MessageActivationAnalyzer(args.output_dir)
    analyzer.analyze_trajectory(model, data)
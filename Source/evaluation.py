import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, Dict
from datetime import datetime

def connect_all(num_nodes):
    """Create a fully connected graph."""
    indices = torch.combinations(torch.arange(num_nodes), with_replacement=False).T
    return torch.cat([indices, indices.flip(0)], dim=1)  # Bidirectional edges

def calculate_metrics(trajectory_true: np.ndarray, trajectory_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate RMSE and ACC metrics between true and predicted trajectories for each timestep.
    
    Args:
        trajectory_true: True trajectories of shape (num_timesteps, num_bodies, spatial_dim)
        trajectory_pred: Predicted trajectories of shape (num_timesteps, num_bodies, spatial_dim)
    
    Returns:
        Dictionary containing RMSE and ACC metrics as time series arrays
    """
    # Calculate RMSE for each timestep
    squared_errors = np.square(trajectory_true - trajectory_pred)
    rmse = np.sqrt(np.mean(squared_errors, axis=(1, 2)))  # mean over bodies and spatial dimensions
    
    # Calculate ACC (Accuracy) for each timestep
    # ACC = 1 - (||y_pred - y_true||_2 / ||y_true||_2)
    numerator = np.linalg.norm(trajectory_pred - trajectory_true, axis=2)  # L2 norm over spatial dimensions
    denominator = np.linalg.norm(trajectory_true, axis=2)  # L2 norm over spatial dimensions
    acc = 1 - np.mean(numerator / (denominator + 1e-10), axis=1)  # mean over bodies
    
    return {
        'rmse': rmse,
        'acc': acc
    }

@torch.no_grad()
def evaluate_and_plot(
    model: torch.nn.Module,
    positions_velocities: np.ndarray,
    accelerations: np.ndarray,
    spatial_dim: int,
    simulation_index: int = 0,
    num_timesteps: int = 50,
    device: str = 'cpu',
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluate and plot a single simulation for specified number of timesteps."""
    model.eval()
    model = model.to(device)

    num_bodies = positions_velocities.shape[2]
    dt = model.dt

    # Initialize position and velocity
    x_pred = torch.tensor(positions_velocities[simulation_index, 0], dtype=torch.float32).to(device)
    trajectory_pred = [x_pred[:, :spatial_dim].cpu().numpy()]
    constants = x_pred[:, 2 * spatial_dim:]  # Keep constants separate

    # Simulate forward
    for t in range(1, num_timesteps):
        edge_index = connect_all(num_bodies).to(device)
        acc = model(x_pred, edge_index)  # Predict accelerations

        pos = x_pred[:, :spatial_dim]
        vel = x_pred[:, spatial_dim:2*spatial_dim]

        # Euler integration step
        vel_next = vel + acc * dt
        pos_next = pos + vel_next * dt
        x_pred = torch.cat([pos_next, vel_next, constants], dim=1)  # Reuse constants
        trajectory_pred.append(pos_next.cpu().numpy())

    # Convert to numpy arrays
    trajectory_pred = np.stack(trajectory_pred)
    trajectory_true = positions_velocities[simulation_index, :num_timesteps, :, :spatial_dim]
    
    print("trajectory_pred =", trajectory_pred.shape)
    print("trajectory_true =", trajectory_true.shape)

    # Calculate metrics
    metrics = calculate_metrics(trajectory_true, trajectory_pred)

    # Plot trajectories
    plt.figure(figsize=(10, 5))
    for i in range(num_bodies):
        plt.plot(trajectory_true[:, i, 0], trajectory_true[:, i, 1], 'k--', linewidth=2.5, 
                label=f'True {i}' if i == 0 else "")
        plt.plot(trajectory_pred[:, i, 0], trajectory_pred[:, i, 1], 
                label=f'Pred {i}')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Predicted vs True Trajectories")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        # Save metrics to a text file
        metrics_file = Path(save_path).parent / "metrics.txt"
        with open(metrics_file, 'w') as f:
            f.write("Time series metrics:\n")
            f.write("RMSE over time:\n")
            for t, rmse in enumerate(metrics['rmse']):
                f.write(f"t={t}: {rmse:.6f}\n")
            f.write("\nACC over time:\n")
            for t, acc in enumerate(metrics['acc']):
                f.write(f"t={t}: {acc:.6f}\n")
            # Also write mean values
            f.write("\nMean metrics:\n")
            f.write(f"Mean RMSE: {np.mean(metrics['rmse']):.6f}\n")
            f.write(f"Mean ACC: {np.mean(metrics['acc']):.6f}\n")
        print(f"Metrics saved to {metrics_file}")
    else:
        plt.show()
    plt.close()

    return trajectory_true, trajectory_pred, metrics

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Evaluate N-body GNN model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run evaluation on")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                       help="Directory to save evaluation results")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension size")
    parser.add_argument("--msg_dim", type=int, default=16, help="Message dimension size")
    parser.add_argument("--num_timesteps", type=int, default=50, help="Number of timesteps to evaluate")
    parser.add_argument("--dt", type=float, required=True, help="Time step size used in simulation")
    parser.add_argument("--ndim", type=int, required=True, help="Number of spatial dimensions")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"trajectory_plot.png"
    
    # Load model and data
    from train import NbodyGraph
    import torch
    
    # Set the same parameters as used in training
    model = NbodyGraph(
        in_channels=6,
        hidden_dim=args.hidden_dim,
        msg_dim=args.msg_dim,
        out_channels=args.ndim, 
        dt=args.dt,
        nt=1,
        ndim=args.ndim
    )
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load data
    data = np.load(args.data_path)
    positions_velocities = data["data"]
    print(positions_velocities.shape)
    accelerations = data["accelerations"]
    
    # Run evaluation
    trajectory_true, trajectory_pred, metrics = evaluate_and_plot(
        model=model,
        positions_velocities=positions_velocities,
        accelerations=accelerations,
        spatial_dim=2,
        simulation_index=0,
        num_timesteps=args.num_timesteps,
        device=args.device,
        save_path=str(output_file)
    )

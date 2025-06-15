import argparse
import os
import re
import pathlib
import numpy as np
import pandas as pd
from pathlib import Path
from pysr import PySRRegressor

# Configure Julia environment
#os.environ["PYTHON_JULIAPKG_PROJECT"] = str(pathlib.Path.home() / "julia_pysr/share/julia/environments/v1.9")
#os.environ["JULIA_BIN"] = str(pathlib.Path.home() / "julia_pysr/bin/julia")

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
        velocities = graph_data.x[:, ndim:2*ndim].cpu().numpy()  # Shape: [num_nodes, ndim]
        accelerations = graph_data.y.cpu().numpy()     # Shape: [num_nodes, ndim]
        
        # Get source and target nodes for each edge
        src_nodes = edge_index[0].cpu().numpy()
        tgt_nodes = edge_index[1].cpu().numpy()
        
        # Calculate edge direction vectors
        edge_vecs = positions[tgt_nodes] - positions[src_nodes]  # Shape: [num_edges, ndim]
        edge_vecs_norm = np.linalg.norm(edge_vecs, axis=1, keepdims=True) + 1e-8
        edge_dirs = edge_vecs / edge_vecs_norm  # Normalized direction vectors
        
        # Get velocities of source nodes
        src_vels = velocities[src_nodes]  # Shape: [num_edges, ndim]
        
        # Get parameters from the last two positions of the state vector
        param1 = graph_data.x[:, -2].cpu().numpy()  # First parameter (e.g., charge, damping, tension)
        param2 = graph_data.x[:, -1].cpu().numpy()  # Second parameter (e.g., mass)
        
        # Get parameters for source and target nodes
        src_param1 = param1[src_nodes]  # Shape: [num_edges]
        src_param2 = param2[src_nodes]  # Shape: [num_edges]
        tgt_param1 = param1[tgt_nodes]  # Shape: [num_edges]
        tgt_param2 = param2[tgt_nodes]  # Shape: [num_edges]
        
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
            'distances': distances_tensor.cpu().numpy(),
            'src_vels': src_vels,
            'src_param1': src_param1,
            'src_param2': src_param2,
            'tgt_param1': tgt_param1,
            'tgt_param2': tgt_param2
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
        return torch.cat([indices, indices.flip(0)], dim=1)

    graphs = []
    for sim in range(2500): # range(num_simulations): # range(300):  @@@debug
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
    match = re.match(r"([a-zA-Z0-9]+)_h\d+_m\d+", filename)
    model_type = match.group(1) 
    pattern = rf"{re.escape(model_type)}_h(\d+)_m(\d+)"
    match = re.search(pattern, filename)
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

def detect_force_type(data_path):
    """Detect the type of forcing from the data path."""
    filename = os.path.basename(data_path)
    if filename.startswith('spring_'):
        return 'spring'
    elif filename.startswith('charge_'):
        return 'charge'
    elif filename.startswith('damped_'):
        return 'damped'
    elif filename.startswith('string_'):
        return 'string'
    elif filename.startswith('disc_'):
        return 'discontinuous'
    elif filename.startswith('r1_'):
        return 'r1'
    elif filename.startswith('r2_'):
        return 'r2'
    else:
        raise ValueError(f"Unknown force type in data path: {data_path}")

def get_force_function(force_type, dim):
    """Get the appropriate force function based on the force type."""
    if force_type == 'spring':
        return lambda msg: -(msg['bd'].to_numpy() - 1)[:, None] * msg[[f'd{"xyz"[i]}' for i in range(dim)]].to_numpy() / msg['bd'].to_numpy()[:, None]
    elif force_type == 'charge':
        return lambda msg: msg['param1'].to_numpy()[:, None] * msg['param2'].to_numpy()[:, None] * msg[[f'd{"xyz"[i]}' for i in range(dim)]].to_numpy() / (msg['bd'].to_numpy()[:, None] ** 2)
    elif force_type == 'damped':
        return lambda msg: (-(msg['bd'].to_numpy() - 1)[:, None] * msg[[f'd{"xyz"[i]}' for i in range(dim)]].to_numpy() / msg['bd'].to_numpy()[:, None] - 
                           msg['param1'].to_numpy()[:, None] * msg[[f'v{"xyz"[i]}' for i in range(dim)]].to_numpy())
    elif force_type == 'string':
        return lambda msg: (-(msg['bd'].to_numpy() - 1)[:, None] * msg[[f'd{"xyz"[i]}' for i in range(dim)]].to_numpy() / msg['bd'].to_numpy()[:, None] +
                           msg['param1'].to_numpy()[:, None] * msg[[f'd{"xyz"[i]}' for i in range(dim)]].to_numpy())
    elif force_type == 'discontinuous':
        return lambda msg: (
            (msg['bd'].to_numpy() < 1)[:, None] * 0.0 +
            ((msg['bd'].to_numpy() >= 1) & (msg['bd'].to_numpy() < 2))[:, None] * (-msg['param1'].to_numpy()[:, None] * msg['param2'].to_numpy()[:, None] / (msg['bd'].to_numpy()[:, None] ** 2)) +
            (msg['bd'].to_numpy() >= 2)[:, None] * (-(msg['bd'].to_numpy() - 1)[:, None] * msg[[f'd{"xyz"[i]}' for i in range(dim)]].to_numpy() / msg['bd'].to_numpy()[:, None])
        )
    elif force_type == 'r1':
        # For r1, the force is derived from potential = m1*m2*log(r)
        return lambda msg: msg['param1'].to_numpy()[:, None] * msg['param2'].to_numpy()[:, None] * msg[[f'd{"xyz"[i]}' for i in range(dim)]].to_numpy() / msg['bd'].to_numpy()[:, None]
    elif force_type == 'r2':
        # For r2, the force is derived from potential = -m1*m2/r
        return lambda msg: -msg['param1'].to_numpy()[:, None] * msg['param2'].to_numpy()[:, None] * msg[[f'd{"xyz"[i]}' for i in range(dim)]].to_numpy() / (msg['bd'].to_numpy()[:, None] ** 2)
    else:
        raise ValueError(f"Unknown force type: {force_type}")

def scatter_all_force_message(messages_over_time, msg_dim, dim=2, data_path=None, sim_type="", model_type=""):
    force_type = detect_force_type(data_path) if data_path else "[Error] cannot determine force type."
    print(f"Detected force type: {force_type}")

    # Prepare all data for saving
    force_fnc = get_force_function(force_type, dim)
    expected_forces = []
    all_messages = []

    for msgs in messages_over_time:
        msgs = copy(msgs)
        msgs['bd'] = msgs.r + 1e-2
        if "bottleneck" in model_type:
            msg_columns = [f"e{k+1}" for k in range(dim)]
        else:
            msg_columns = [f"e{k+1}" for k in range(msg_dim)]
        msg_array = np.array(msgs[msg_columns])
        
        expected_forces.append(force_fnc(msgs))
        all_messages.append(msg_array)

    # Save all data in a single file
    output_file = "model_data.npz"
    pos_cols = ['dx', 'dy', 'dz'][:dim]
    data = {
        'force_type': force_type,
        'msg_dim': msg_dim,
        'dim': dim,
        'sim_type': sim_type,
        'model_type': model_type,
        'messages': [msg.to_numpy() for msg in messages_over_time],
        'distances': [msg['r'].to_numpy() for msg in messages_over_time],
        'edge_dirs': [msg[[f'd{"xyz"[i]}' for i in range(dim)]].to_numpy() for msg in messages_over_time],
        'velocities': [msg[[f'v{"xyz"[i]}' for i in range(dim)]].to_numpy() for msg in messages_over_time],
        'param1': [msg['param1'].to_numpy() for msg in messages_over_time],
        'param2': [msg['param2'].to_numpy() for msg in messages_over_time],
        'expected_forces': expected_forces,
        'all_messages': all_messages,
        'direction_vectors': [msg[pos_cols].to_numpy() for msg in messages_over_time]
    }
    np.savez(output_file, **data)
    print(f"All data saved to {output_file}")


def load_previous_data(prev_data_path):
    """Load data from a previous run if it exists."""
    if prev_data_path and os.path.exists(prev_data_path):
        print(f"Loading previous data from {prev_data_path}")
        data = np.load(prev_data_path)
        return {
            'force_type': data['force_type'],
            'dim': data['dim'],
            'sim_type': data['sim_type'],
            'model_type': data['model_type'],
            'expected_forces': data['expected_forces'],
            'all_messages': data['all_messages']
        }
    return None

def process_data(model, model_path, data_path, ndim, dt, prev_data=None):
    """Process the data and return the results."""
    data_base = os.path.basename(data_path).replace('.npz', '')
    model_base = os.path.basename(model_path).replace('.pt', '')

    graphs = create_graphs(data_path)
    all_data = [single_graph(model, g) for g in graphs if single_graph(model, g) is not None]
    print(f"Processed {len(all_data)} simulations")

    messages_over_time = []
    for data in all_data:
        msgs = data['messages']
        df = pd.DataFrame(msgs, columns=[f"e{i+1}" for i in range(msgs.shape[1])])
        df['r'] = data['distances']
        df['bd'] = data['distances'] + 1e-2
        for i in range(ndim):
            df[f'd{"xyz"[i]}'] = data['edge_dirs'][:, i]
            df[f'v{"xyz"[i]}'] = data['src_vels'][:, i]
        df['param1'] = data['src_param1']
        df['param2'] = data['src_param2']
        messages_over_time.append(df)

    return messages_over_time, data_base, model_base

def apply_symbolic_regression(all_messages, expected_forces, dim, force_type, model_type):
    """
    Apply symbolic regression to find relationships between messages and forces.
    
    Args:
        all_messages: List of message arrays from the GNN
        expected_forces: List of expected force arrays
        dim: Number of dimensions
        force_type: Type of force being analyzed
        model_type: Type of model used
    """
    
    # Convert lists to numpy arrays
    all_messages = np.vstack(all_messages)  # Shape: [total_edges, msg_dim]
    expected_forces = np.vstack(expected_forces)  # Shape: [total_edges, dim]
    
    # Calculate norms for magnitude analysis
    message_norms = np.linalg.norm(all_messages, axis=1)
    force_norms = np.linalg.norm(expected_forces, axis=1)
    
    equations = []
    r2_scores = []
    
    # Train for each force dimension
    for i in range(dim):
        print(f"\n--- Training for Force Dimension {i+1}/{dim} ---")
        
        y = expected_forces[:, i]
        X = all_messages
        
        model = PySRRegressor(
            model_selection="best",  # Keep best single equation
            niterations=20,  # Increased from 5 for better results
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["log", "abs"],  # Added unary operators
            loss="loss(x, y) = (x - y)^2",
            maxsize=25,  # Increased for more complex equations
            populations=30,
            procs=4,  # Parallelism
            batching=True,  # Enable batching for better performance
            batch_size=50,  # Batch size for training
            warm_start=True,  # Enable warm start for faster convergence
            random_state=42  # For reproducibility
        )
        
        try:
            model.fit(X, y)
            best_equation = model.equations_.iloc[0]
            equations.append(best_equation)
            
            # Calculate R² score
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            r2_scores.append(r2)
            
            print(f"Best equation for dimension {i+1}:")
            print(f"Equation: {best_equation.equation}")
            print(f"R² score: {r2:.4f}")
            
        except Exception as e:
            print(f"Error in dimension {i+1}: {str(e)}")
            equations.append(None)
            r2_scores.append(None)
    
    # Save results
    results = {
        'force_type': force_type,
        'model_type': model_type,
        'dim': dim,
        'equations': equations,
        'r2_scores': r2_scores,
        'message_norms': message_norms,
        'force_norms': force_norms
    }
    
    output_file = f"sr_results_{force_type}_{model_type}.npz"
    np.savez(output_file, **results)
    print(f"\nResults saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--ndim', type=int, default=2, help='Number of dimensions')
    parser.add_argument('--dt', type=float, default=0.1, help='Time step')
    parser.add_argument('--prev_data', type=str, help='Path to previous data file')
    args = parser.parse_args()

    # Load the saved data for symbolic regression
    if args.prev_data and args.prev_data != "None":
        print(f"Loading previous data from {args.prev_data}")
        data = np.load(args.prev_data)
        sr_results = apply_symbolic_regression(
            data['all_messages'],
            data['expected_forces'],
            args.ndim,
            data['force_type'],
            data['model_type']
        )
    else:
        import torch
        import matplotlib.pyplot as plt
        from scipy.optimize import minimize
        from copy import copy
        from torch_geometric.data import Data
        from model_util import NbodyGraph
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        # Initialize model
        hidden_dim, msg_dim = parse_model_params(args.model_path)
        model = NbodyGraph(
            in_channels=2 * args.ndim + 2,
            hidden_dim=hidden_dim,
            msg_dim=msg_dim,
            out_channels=args.ndim,
            dt=args.dt,
            nt=1,
            ndim=args.ndim
        )
        checkpoint = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Process data and extract messages
        messages_over_time, data_base, model_base = process_data(
            model, args.model_path, args.data_path, args.ndim, args.dt, None
        )

        # Run analysis and save data
        scatter_all_force_message(
            messages_over_time, 
            msg_dim=msg_dim, 
            dim=args.ndim,
            data_path=args.data_path,
            sim_type=data_base, 
            model_type=model_base
        )

        # Load the saved data for symbolic regression
        data = np.load("model_data.npz")
        
        # Apply symbolic regression
        sr_results = apply_symbolic_regression(
            data['all_messages'],
            data['expected_forces'],
            args.ndim,
            data['force_type'],
            data['model_type']
        )
    
    # Print summary
    print("\nSymbolic Regression Summary:")
    print(f"Force Type: {sr_results['force_type']}")
    print(f"Model Type: {sr_results['model_type']}")
    print("\nEquations by dimension:")
    for i, (eq, r2) in enumerate(zip(sr_results['equations'], sr_results['r2_scores'])):
        if eq is not None:
            print(f"Dimension {i+1}:")
            print(f"  Equation: {eq.equation}")
            print(f"  R² score: {r2:.4f}")

if __name__ == "__main__":
    main()

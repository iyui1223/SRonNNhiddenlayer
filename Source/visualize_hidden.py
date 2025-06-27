import argparse
import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from celluloid import Camera
from scipy.optimize import minimize
from copy import copy
from torch_geometric.data import Data
from model_util import NbodyGraph  # Update path if needed
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
    for sim in range(10): # range(num_simulations): 300 @@@debug
        for t in range(250):  # Take ALL timesteps
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

    pos_cols = ['dx', 'dy', 'dz'][:dim]
    fig, axes = plt.subplots(1, dim, figsize=(5 * dim, 5))
    if dim == 1:
        axes = [axes]

    force_fnc = get_force_function(force_type, dim)

    # --- Sparsity analysis (keep as is, per simulation) ---
    sparsity_ratios = []
    for msgs in messages_over_time:
        if "bottleneck" in model_type:
            msg_columns = [f"e{k+1}" for k in range(dim)]
        else:
            msg_columns = [f"e{k+1}" for k in range(msg_dim)]
        msg_array = np.array(msgs[msg_columns])
        overall_var = np.var(msg_array, axis=0)
        overall_var_sum = np.sum(overall_var)
        msg_importance = np.var(msg_array, axis=0)
        most_important = np.argsort(msg_importance)[-dim:]
        top_msgs_var = np.var(msg_array[:, most_important], axis=0)
        top_msgs_var_sum = np.sum(top_msgs_var)
        sparsity_ratio = top_msgs_var_sum / overall_var_sum
        sparsity_ratios.append(sparsity_ratio)
    mean_sparsity = np.mean(sparsity_ratios)
    sparsity_file = "message_sparsity.txt"
    with open(sparsity_file, 'w') as f:
        f.write(f"Mean sparsity ratio: {mean_sparsity:.4f}\n")
        f.write(f"Number of simulations analyzed: {len(sparsity_ratios)}\n")
        f.write(f"Top {dim} messages variance / Overall messages variance\n")
        f.write(f"Individual simulation ratios:\n")
    print(f"Sparsity analysis saved to {sparsity_file}")

    # --- NEW: Use all messages at once for the rest ---
    all_msgs_df = pd.concat(messages_over_time, ignore_index=True)
    all_msgs_df['bd'] = all_msgs_df.r + 1e-2
    if "bottleneck" in model_type:
        msg_columns = [f"e{k+1}" for k in range(dim)]
    else:
        msg_columns = [f"e{k+1}" for k in range(msg_dim)]
    msg_array = np.array(all_msgs_df[msg_columns])
    msg_importance = msg_array.std(axis=0)
    most_important = np.argsort(msg_importance)[-dim:]
    msgs_to_compare = msg_array[:, most_important]
    msgs_to_compare = (msgs_to_compare - np.mean(msgs_to_compare, axis=0)) / np.std(msgs_to_compare, axis=0)
    expected_forces = force_fnc(all_msgs_df)

    def linear_transformation(alpha):
        lincombs = []
        for i in range(dim):
            lin = sum(alpha[i * dim + j] * expected_forces[:, j] for j in range(dim)) + alpha[dim * dim + i]
            lincombs.append(lin)
        return np.mean([percentile_sum((msgs_to_compare[:, i] - lincombs[i]) ** 2) for i in range(dim)])

    def out_linear_transformation(alpha):
        lincombs = []
        for i in range(dim):
            lin = sum(alpha[i * dim + j] * expected_forces[:, j] for j in range(dim)) + alpha[dim * dim + i]
            lincombs.append(lin)
        return lincombs

    init_params = np.ones(dim * (dim + 1))
    min_result = minimize(linear_transformation, init_params, method='Powell')
    lincombs = out_linear_transformation(min_result.x)

    all_force_proj = [[] for _ in range(dim)]
    all_msg_comp = [[] for _ in range(dim)]
    for i in range(dim):
        all_force_proj[i].append(lincombs[i])
        all_msg_comp[i].append(msgs_to_compare[:, i])

    r2_scores = []
    for i in range(dim):
        px = np.concatenate(all_force_proj[i]).reshape(-1, 1)  # Independent variable
        py = np.concatenate(all_msg_comp[i])                   # Dependent variable
        mask = (px[:, 0] >= -1) & (px[:, 0] <= 1) & (py >= -1) & (py <= 1)
        px = px[mask]
        py = py[mask]
        reg = LinearRegression().fit(px, py)
        y_pred = reg.predict(px)
        r2 = r2_score(py, y_pred)
        r2_scores.append(r2)
        axes[i].scatter(px, py, s=1, alpha=0.1, color='black')
        axes[i].plot(px, y_pred, color='red', linewidth=1.0, label=f"R2={r2:.4f}")
        axes[i].set_xlabel('Linear combination of forces')
        axes[i].set_ylabel(f'Message Element {i+1}')
        axes[i].set_title(f"Message {i+1} vs Force", fontsize=10)
        fig.text(0.01, 0.99, f"Sim: {sim_type}\nModel: {model_type}", ha='left', va='top', fontsize=6)
        axes[i].grid(True)
        axes[i].set_xlim(-1, 1)
        axes[i].set_ylim(-1, 1)
        axes[i].legend()
    plt.tight_layout()
    plt.savefig("force_message_relation.png")
    plt.show()
    r2_file = "message_r2_scores.txt"
    with open(r2_file, 'w') as f:
        f.write("R2 Scores for Linear Fit between Force Projection and Message Elements:\n")
        for i, r2 in enumerate(r2_scores):
            f.write(f"Message Element {i+1}: {r2:.4f}\n")
    print(f"R2 scores saved to {r2_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--ndim", type=int, required=True, help="Number of spatial dimensions")
    parser.add_argument("--dt", type=float, required=True, help="Time step size")
    args = parser.parse_args()

    data_base = os.path.basename(args.data_path).replace('.npz', '')
    model_base = os.path.basename(args.model_path).replace('.pt', '')

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

    graphs = create_graphs(args.data_path)
    all_data = [single_graph(model, g) for g in graphs if single_graph(model, g) is not None]
    print(f"Processed {len(all_data)} simulations")

    messages_over_time = []
    for data in all_data:
        msgs = data['messages']
        df = pd.DataFrame(msgs, columns=[f"e{i+1}" for i in range(msgs.shape[1])])
        df['r'] = data['distances']
        df['bd'] = data['distances'] + 1e-2
        for i in range(args.ndim):
            df[f'd{"xyz"[i]}'] = data['edge_dirs'][:, i]
            df[f'v{"xyz"[i]}'] = data['src_vels'][:, i]
        df['param1'] = data['src_param1']
        df['param2'] = data['src_param2']
        messages_over_time.append(df)

    scatter_all_force_message(messages_over_time, msg_dim=msg_dim, dim=args.ndim,
                              data_path=args.data_path,
                              sim_type=data_base, model_type=model_base)

if __name__ == "__main__":
    main()

import argparse
import numpy as np
import simulate

def main():
    parser = argparse.ArgumentParser(description="Run an n-body simulation with user-defined parameters.")

    # Define the command-line arguments
    parser.add_argument("--sim", type=str, required=True, help="Type of simulation (e.g., spring, r1, r2, charge, etc.)")
    parser.add_argument("--n", type=int, required=True, help="Number of nodes")
    parser.add_argument("--dim", type=int, required=True, choices=[2, 3], help="Dimension (2D or 3D)")
    parser.add_argument("--nt", type=int, required=True, help="Number of time steps")
    parser.add_argument("--ns", type=int, default=10000, help="Number of simulations to run (default: 10000)")

    args = parser.parse_args()

    # Simulation options
    sim_sets = [
        {'sim': 'r1', 'dt': [5e-3], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
        {'sim': 'r2', 'dt': [1e-3], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
        {'sim': 'spring', 'dt': [1e-2], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
        {'sim': 'string', 'dt': [1e-2], 'nt': [1000], 'n': [30], 'dim': [2]},
        {'sim': 'charge', 'dt': [1e-3], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
        {'sim': 'superposition', 'dt': [1e-3], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
        {'sim': 'damped', 'dt': [2e-2], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
        {'sim': 'discontinuous', 'dt': [1e-2], 'nt': [1000], 'n': [4, 8], 'dim': [2, 3]},
    ]

    # Get appropriate dt for the selected simulation
    dt = next((ss['dt'][0] for ss in sim_sets if ss['sim'] == args.sim), None)

    if dt is None:
        print(f"Error: Simulation type '{args.sim}' is not recognized.")
        return

    # Generate a unique title based on parameters
    title = f"{args.sim}_n{args.n}_dim{args.dim}_nt{args.nt}_dt{dt}"
    print(f"Running simulation: {title}")

    # Run the simulation
    s = simulate.SimulationDataset(args.sim, n=args.n, dim=args.dim, nt=args.nt//2, dt=dt)
    s.simulate(args.ns)

    # Save simulation data
    accel_data = s.get_acceleration()
    np.save(f"data_{title}.npy", s.data)
    np.save(f"accel_{title}.npy", accel_data)

    print(f"Simulation completed. Data saved as 'data_{title}.npy' and 'accel_{title}.npy'.")

if __name__ == "__main__":
    main()

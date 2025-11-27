# load_controller.py

import torch
import argparse
import csv
from datetime import datetime
from pathlib import Path
from mlp import ParameterController
from ants_env import SteppableACO
import job_shop_lib.benchmarking as benchmarking
import utils

def load_model(model_path, device):
    """
    Load trained model from either state_dict or checkpoint format.
    """
    controller = ParameterController(input_dim=6, action_dim=3).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            controller.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"Loaded checkpoint from epoch {epoch}")
        else:
            controller.load_state_dict(checkpoint)
            print(f"Loaded model state_dict")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    controller.eval()
    return controller

def run_inference(model_path, instance_name, num_ants=200, iterations_per_batch=50, num_batches=10, device=None, log_dir="logs"):
    """
    Run inference with dynamic parameter updates every batch.
    Logs results to CSV file.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    controller = load_model(model_path, device)
    print(f"Loaded model from {model_path}\n")
    
    instance = benchmarking.load_benchmark_instance(instance_name)
    utils.inspect_instance(instance_name)
    
    aco = SteppableACO(
        instance=instance,
        num_ants=num_ants,
        iterations=0,
        alpha=1.0,
        beta=1.0,
        rho=0.1
    )
    
    # Build initial solution
    sched, seq = aco._build_ant_solution()
    aco.global_best_schedule = sched
    aco.global_best_seq = seq
    initial_makespan = aco.global_best_schedule.makespan()
    print(f"Initial Makespan: {initial_makespan}\n")
    
    # Setup CSV logging
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).stem
    csv_filename = f"{log_dir}/{instance_name}_{model_name}_{timestamp}.csv"
    
    # Open CSV file and write header
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'batch', 'total_iterations', 'makespan', 'improvement', 'avg_chaos',
        'alpha', 'beta', 'rho', 'reduction_from_initial', 'percent_improvement'
    ])
    
    print(f"Running {num_batches} batches of {iterations_per_batch} iterations each")
    print(f"Logging to: {csv_filename}")
    print("=" * 70)
    
    for batch_idx in range(num_batches):
        # Controller predicts new parameters based on current state
        with torch.no_grad():
            state = aco.get_state()
            # Check if state is already a tensor
            if isinstance(state, torch.Tensor):
                state = state.detach().clone().to(device)
            else:
                state = torch.tensor(state, dtype=torch.float32).to(device)
            
            # Squeeze to remove batch dimension if present
            if state.ndim == 2 and state.shape[0] == 1:
                state = state.squeeze(0)
            
            action_dict, log_prob = controller.get_action(state)
        
        # Apply predicted parameters
        aco.set_params(action_dict['alpha'], action_dict['beta'], action_dict['rho'])
        
        # Run batch with these parameters
        improvement, avg_chaos = aco.run_batch(num_iterations=iterations_per_batch)
        current_makespan = aco.global_best_schedule.makespan()
        total_iters = (batch_idx + 1) * iterations_per_batch
        reduction = initial_makespan - current_makespan
        percent_imp = 100 * reduction / initial_makespan
        
        # Write to CSV
        csv_writer.writerow([
            batch_idx + 1,
            total_iters,
            current_makespan,
            improvement,
            f"{avg_chaos:.4f}",
            f"{action_dict['alpha']:.4f}",
            f"{action_dict['beta']:.4f}",
            f"{action_dict['rho']:.4f}",
            reduction,
            f"{percent_imp:.2f}"
        ])
        csv_file.flush()  # Ensure data is written immediately
        
        print(f"Batch {batch_idx+1:2d} | "
              f"Makespan: {current_makespan:4d} | "
              f"α: {action_dict['alpha']:.3f} | "
              f"β: {action_dict['beta']:.3f} | "
              f"ρ: {action_dict['rho']:.3f} | "
              f"Chaos: {avg_chaos:.2f}")
    
    csv_file.close()
    
    final_makespan = aco.global_best_schedule.makespan()
    total_reduction = initial_makespan - final_makespan
    percent_improvement = 100 * total_reduction / initial_makespan
    
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Initial Makespan: {initial_makespan}")
    print(f"  Final Makespan:   {final_makespan}")
    print(f"  Reduction:        {total_reduction} ({percent_improvement:.2f}%)")
    print(f"  Total Iterations: {num_batches * iterations_per_batch}")
    print(f"\nResults saved to: {csv_filename}")
    
    return {
        'initial_makespan': initial_makespan,
        'final_makespan': final_makespan,
        'reduction': total_reduction,
        'percent_improvement': percent_improvement,
        'csv_file': csv_filename
    }

def main():
    parser = argparse.ArgumentParser(description='Load and evaluate trained ACO parameter controller')
    parser.add_argument('--model_path', type=str, default='parameter_controller_final.pth',
                        help='Path to trained model weights')
    parser.add_argument('--instance', type=str, default='ft10',
                        help='Job-shop benchmark instance name')
    parser.add_argument('--num_ants', type=int, default=200,
                        help='Number of ants in ACO')
    parser.add_argument('--iterations_per_batch', type=int, default=50,
                        help='Number of ACO iterations per batch')
    parser.add_argument('--num_batches', type=int, default=10,
                        help='Number of batches (parameter updates)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save CSV logs')
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else None
    
    results = run_inference(
        model_path=args.model_path,
        instance_name=args.instance,
        num_ants=args.num_ants,
        iterations_per_batch=args.iterations_per_batch,
        num_batches=args.num_batches,
        device=device,
        log_dir=args.log_dir
    )
    
    return results

if __name__ == "__main__":
    main()

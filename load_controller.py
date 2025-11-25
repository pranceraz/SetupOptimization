# load_trained_controller.py

import torch
from mlp import ParameterController
from ants_env import SteppableACO
import job_shop_lib.benchmarking as benchmarking
import utils

# -------------------------
# Configuration
# -------------------------
MODEL_PATH = "parameter_controller_final.pth"  # Change if using a checkpoint
INSTANCE_NAME = "ft10"                          # Job-shop instance to run

# -------------------------
# Device setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Load trained model
# -------------------------
controller = ParameterController(input_dim=6, action_dim=3).to(device)
controller.load_state_dict(torch.load(MODEL_PATH, map_location=device))
controller.eval()  # evaluation mode
print(f"Loaded model weights from {MODEL_PATH}")

# -------------------------
# Load benchmark instance
# -------------------------
instance = benchmarking.load_benchmark_instance(INSTANCE_NAME)
utils.inspect_instance(INSTANCE_NAME)

# -------------------------
# Initialize ACO environment
# -------------------------
aco = SteppableACO(
    instance=instance,
    num_ants=200,
    iterations=0,
    alpha=1.0,
    beta=1.0,
    rho=0.1
)

# Build initial solution
sched, seq = aco._build_ant_solution()
aco.global_best_schedule = sched
aco.global_best_seq = seq
print(f"Initial Makespan: {aco.global_best_schedule.makespan()}")

# -------------------------
# Run the controller once
# -------------------------
state = torch.tensor(aco.get_state(), dtype=torch.float32).to(device)
action_dict, log_prob = controller.get_action(state)

# Apply the predicted parameters
aco.set_params(action_dict['alpha'], action_dict['beta'], action_dict['rho'])

print("Controller suggested parameters:")
print(f"Alpha: {action_dict['alpha']:.2f}, Beta: {action_dict['beta']:.2f}, Rho: {action_dict['rho']:.2f}")

# Run one batch to see result
improvement, avg_chaos = aco.run_batch(num_iterations=50)
print(f"New Makespan after applying controller: {aco.global_best_schedule.makespan()}")
print(f"Improvement: {improvement}, Avg Chaos: {avg_chaos:.2f}")

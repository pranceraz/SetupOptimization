import job_shop_lib.benchmarking as benchmarking
from ants_env import SteppableACO
from mlp import ParameterController
import torch
import torch.optim as optim
import logging
import utils
# Configure logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

def train_nn_aco(instance_name):
    # 1. Setup
    instance = benchmarking.load_benchmark_instance(instance_name)
    utils.inspect_instance(instance_name)
    # Initialize Environment
    aco = SteppableACO(
        instance=instance, 
        num_ants=100, 
        iterations=0, 
        alpha=1.0, beta=1, rho=0.1        
    )

    # We must build one solution so 'global_best_schedule' exists
    sched, seq = aco._build_ant_solution()
    aco.global_best_schedule = sched
    aco.global_best_seq = seq
    # ---------------------------------------------

    # Initialize Controller
    controller = ParameterController(input_dim=6, action_dim=3)
    optimizer = optim.Adam(controller.parameters(), lr=0.001)

    MAX_STEPS = 20
    BATCH_ITERS = 50

    print(f"Initial Makespan: {aco.global_best_schedule.makespan()}")
    
    for step in range(MAX_STEPS):
        # A. Get State
        state = aco.get_state()
        
        # B. Get Action
        action_dict, log_prob = controller.get_action(state)
        
        # C. Apply Params
        aco.set_params(action_dict['alpha'], action_dict['beta'], action_dict['rho'])
        
        # D. Run Batch
        improvement, avg_chaos = aco.run_batch(num_iterations=BATCH_ITERS)
        current_best = aco.global_best_schedule.makespan()
        
        # E. Calculate Reward
        if improvement > 0:
            # SUCCESS
            percent_imp = (improvement / (current_best + improvement)) * 100.0
            reward = 1.0 + percent_imp
            log.info(f"IMPROVED: {percent_imp:.2f}% | New Best: {current_best} | Reward: {reward:.2f}")

        else:
            # STAGNATION
            if avg_chaos > 0.85:
                 # High Chaos (Exploration) -> Small Reward to encourage searching
                 reward = 0.1
                 log.info(f"SEARCHING: Chaos {avg_chaos:.2f} | Reward: {reward:.2f}")

            elif avg_chaos < 0.3:
                 # Low Chaos (Convergence) -> Penalty for being stuck
                 reward = -1.0
                 log.warning(f"STUCK (Converged): Chaos {avg_chaos:.2f} | Reward: {reward:.2f}")
                 
            else:
                 # Transitioning -> Scaled Penalty
                 reward = -1.0 + avg_chaos
                 log.info(f"STAGNATING: Chaos {avg_chaos:.2f} | Reward: {reward:.2f}")

        # F. Update Network
        if log_prob is not None:
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            loss = -(reward_tensor * log_prob)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Periodic Print
        if step % 1 == 0:
            print(f"Step {step} | Best: {current_best} | "
                  f"Params: A={action_dict['alpha']:.2f} B={action_dict['beta']:.2f} R={action_dict['rho']:.2f} | "
                  f"Reward: {reward:.2f}")

    print(f"Final Makespan: {aco.global_best_schedule.makespan()}")

if __name__ == "__main__":
    train_nn_aco(instance_name="ft10")
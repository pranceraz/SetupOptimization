import job_shop_lib.benchmarking as benchmarking
from ants_env import SteppableACO
from mlp import ParameterController
import torch
import torch.optim as optim
import logging, os
import utils
# Configure logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
log = logging.getLogger(__name__)

def train_nn_aco(instance_name,LOAD_CHECKPOINT = True ):
    # 1. Setup
    instance = benchmarking.load_benchmark_instance(instance_name)
    utils.inspect_instance(instance_name)
    # Initialize Environment
    aco = SteppableACO(
        instance=instance, 
        num_ants=200, 
        iterations=0, 
        alpha=1.0, beta=1, rho=0.1,q= 0.05,
        elitist= True, elitist_factor= .1  
    )

    # We must build one solution so 'global_best_schedule' exists
    sched, seq = aco._build_ant_solution()
    aco.global_best_schedule = sched
    aco.global_best_seq = seq
    # ---------------------------------------------

    # Initialize Controller

   

    controller = ParameterController(input_dim=6, action_dim=3)
    optimizer = optim.Adam(controller.parameters(), lr=0.001)

      
    checkpoint_file = "150_steps.pth"
    start_step = 0
    if LOAD_CHECKPOINT and os.path.exists(checkpoint_file):
        print(f"Checkpoint found! Loading {checkpoint_file} ...")
        checkpoint = torch.load(checkpoint_file, map_location="cpu")

        controller.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1

        print(f"Resuming training from step {start_step}")
        
    else:
        print("No checkpoint found. Starting fresh training.")

    MAX_STEPS = 200
    BATCH_ITERS = 50

    print(f"Initial Makespan: {aco.global_best_schedule.makespan()}")
    
    for step in range(start_step, MAX_STEPS):
        # A. Get State
        state = aco.get_state()
        
        # B. Get Action
        action_dict, log_prob = controller.get_action(state)
        
        # C. Apply Params
        aco.set_params(action_dict['alpha'], action_dict['beta'], action_dict['rho'])
        
        # D. Run Batch
        improvement, avg_chaos= aco.run_batch(num_iterations=BATCH_ITERS)
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
        if (step + 1) % 25 == 0:
            checkpoint = {
                "model_state_dict": controller.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step
            }
            torch.save(checkpoint, checkpoint_file)
            print(f"Checkpoint saved at step {step+1} -> {checkpoint_file}")

        if step % 1 == 0:
            print(f"Step {step} | Best: {current_best} | "
                f"Params: A={action_dict['alpha']:.2f} B={action_dict['beta']:.2f} R={action_dict['rho']:.2f} | "
                f"Reward: {reward:.2f}")

    print(f"Final Makespan: {aco.global_best_schedule.makespan()}")
    torch.save(controller.state_dict(), "parameter_controller_final.pth")
    print("Final model saved -> parameter_controller_final.pth")


if __name__ == "__main__":
    train_nn_aco(instance_name="ta01", LOAD_CHECKPOINT= True)
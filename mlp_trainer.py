import job_shop_lib.benchmarking as benchmarking
import csv
from ants_env import SteppableACO
from mlp import ParameterController
import torch
import torch.optim as optim
import logging, os
import utils
import time
import numpy as np 

# Configure logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
log = logging.getLogger(__name__)



WARMUP_STEPS = 5

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
        elitist= True, elitist_factor= .5  
    )

    # We must build one solution so 'global_best_schedule' exists
    sched, seq = aco._build_ant_solution()
    aco.global_best_schedule = sched
    aco.global_best_seq = seq
    # ---------------------------------------------

    # Initialize Controller

   

    controller = ParameterController(input_dim=6, action_dim=3)
    optimizer = optim.Adam(controller.parameters(), lr=0.001)

      
    checkpoint_file = "fresh_point.pth"
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

    MAX_STEPS = 1000
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

        if step < WARMUP_STEPS:
            reward = 0.0
            log.info(f"[WARM-UP] Step {step} | Reward suppressed.")

        else:
            old_best = current_best + improvement
            new_best = current_best

            if improvement > 1e-12:
                # SUCCESS: reward = log improvement
                reward = np.log((old_best + 1e-12) / (new_best + 1e-12))
                log.info(f"IMPROVED: {improvement:.2f} | New Best: {current_best} | Reward: {reward:.3f}")

                # Penalize if chaos is way too high
                if avg_chaos > 0.9:  # adjust threshold as needed
                    chaos_penalty = -0.5 * (avg_chaos - 0.9)
                    reward += chaos_penalty
                    log.info(f"High chaos penalty applied: {chaos_penalty:.3f} | Adjusted Reward: {reward:.3f}")

            else:
                # STAGNATION: reward mostly based on chaos
                if avg_chaos > 0.9:
                    reward = -0.5 * (avg_chaos - 0.9)  # only penalize extreme chaos
                    log.warning(f"STAGNATION + HIGH CHAOS {avg_chaos:.2f} | Reward: {reward:.3f}")
                else:
                    reward = 0.0
                    log.info(f"STAGNATION: Chaos {avg_chaos:.2f} | Reward: {reward:.3f}")

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
    train_nn_aco(instance_name="ft06", LOAD_CHECKPOINT= False)
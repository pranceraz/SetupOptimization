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



WARMUP_STEPS = 3

def train_nn_aco(instance_name,LOAD_CHECKPOINT = True ):
    # 1. Setup
    instance = benchmarking.load_benchmark_instance(instance_name)
    utils.inspect_instance(instance_name)

    c_opt = instance.metadata.get('optimum')
    # Initialize Environment
    aco = SteppableACO(
        instance=instance, 
        num_ants=75, 
        iterations=0, 
        alpha=1.0, beta=2, rho=0.1,q= 1,
        elitist= True, elitist_factor= 20
    )

    # We must build one solution so 'global_best_schedule' exists
    sched, seq = aco._build_ant_solution()
    aco.global_best_schedule = sched
    aco.global_best_seq = seq
    # ---------------------------------------------

    # Initialize Controller

   

    controller = ParameterController(input_dim=4, action_dim=3)
    optimizer = optim.Adam(controller.parameters(), lr=0.001)

      
    checkpoint_file = "scratch24base.pth"
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

    MAX_STEPS = 100000
    BATCH_ITERS = 10

    # Tunable parameters
    IMPROVEMENT_SCALE = 2.0     # how strongly improvement is rewarded
    STAGNATION_PENALTY = -0.1   # mild, not catastrophic
    CHAOS_LOW_PENALTY = -0.4
    CHAOS_HIGH_PENALTY = -0.2

    CHAOS_LOW = 0.1
    CHAOS_GOOD_LOW = 0.3
    CHAOS_GOOD_HIGH = 0.40
    CHAOS_HIGH = 0.70
    no_improvement_for = 0
    warmup = 0

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




    # --- Reward Calculation ---
        old_best = current_best + improvement
        new_best = current_best

        if warmup < WARMUP_STEPS:
            reward = 0.0
            log.info(f"[WARM-UP] Step {step} | Reward suppressed.")
            warmup +=1

        else:

            # ===========================
            # 1. TRUE IMPROVEMENT REWARD
            # ===========================
            if improvement > 1e-6:
                # Log improvement ensures small but smooth gains
                imp_reward = np.log((old_best + 1e-6) / (new_best + 1e-6))
                no_improvement_for = 0

                # Stronger positive scaling
                reward = 0.1 + IMPROVEMENT_SCALE * imp_reward

                log.info(
                    f"IMPROVED: Δ={improvement:.2f} | Chaos={avg_chaos:.2f} | "
                    f"Reward={reward:.3f}"
                )

            else:
                # No improvement (stagnation)
                reward = 0.0  # start neutral
                no_improvement_for += 1
                # =======================
                # 2. CHAOS REWARD SHAPING
                # =======================

                if avg_chaos < CHAOS_LOW:
                    # Strong convergence (could be stuck)
                    reward += CHAOS_LOW_PENALTY
                    log.warning(
                        f"LOW CHAOS (possible stagnation): Chaos={avg_chaos:.2f} | "
                        f"Reward={reward:.2f}"
                    )

                elif CHAOS_LOW <= avg_chaos < CHAOS_GOOD_LOW:
                    # Very stable, probably converged but not bad
                    reward += +0.05
                    log.info(
                        f"GOOD-LOW CHAOS (stable): Chaos={avg_chaos:.2f} | "
                        f"Reward={reward:.2f}"
                    )

                elif CHAOS_GOOD_LOW <= avg_chaos <= CHAOS_GOOD_HIGH:
                    # Ideal exploration window
                    reward += +0.02
                    log.info(
                        f"HEALTHY CHAOS: Chaos={avg_chaos:.2f} | "
                        f"Reward={reward:.2f}"
                    )

                elif avg_chaos > CHAOS_HIGH:
                    # Too chaotic — penalize
                    reward += CHAOS_HIGH_PENALTY
                    log.warning(
                        f"HIGH CHAOS: Chaos={avg_chaos:.2f} | Reward={reward:.2f}"
                    )

                # ===========================
                # 3. MILD STAGNATION PENALTY
                # ===========================
                # Only if no improvement for multiple steps
                if no_improvement_for >= 50:
                    reward += STAGNATION_PENALTY
                    log.warning(
                        f"STAGNATION: No improvement for {no_improvement_for} steps | "
                        f"Reward={reward:.2f}"
                    )




        # F. Update Network
        if log_prob is not None:
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            loss = -(reward_tensor * log_prob)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        # Periodic Print
        if (step + 1) % 5 == 0:
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
    train_nn_aco(instance_name="ft10", LOAD_CHECKPOINT= True)
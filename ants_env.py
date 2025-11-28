#ants_env.py

from ants import ACO_Solver
import numpy as np
import torch, csv
import logging,random
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import time

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
log = logging.getLogger(__name__)


class SteppableACO(ACO_Solver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # RL Tracking (Observation Only)
        self.stagnation_counter = 0
        self.last_best_makespan = float('inf')
        self.num_ops = self.instance.num_operations
        self.pheromone = np.full((self.num_ops + 1, self.num_ops),0.5, dtype=np.float64)
        self.initial_makespan = 5000.0

        timestamp = int(time.time())
        self.log_path = f"aco_training_log_{timestamp}.csv"

        # Write CSV header once
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration",
                "best_makespan",
                "global_best_makespan",
                "chaos_score",
                "pheromone_entropy",
                "pheromone_sum",
                "alpha",
                "beta",
                "rho",
                "stagnation"
            ])
        self.iteration_counter = 0



    def set_params(self, alpha, beta, rho):
        """Update parameters dynamically."""
        self.alpha = float(np.clip(alpha, 0.1, 5.0))
        self.beta = float(np.clip(beta, 0.1, 5.0))
        self.rho = float(np.clip(rho, 0.01, 0.6))


    def get_pheromone_entropy(self):
        P = self.pheromone
        S = np.sum(P)
        if S < 1e-12:  
            return 0.0
        P = (P / S).flatten()
        return float(-(P * np.log(P + 1e-12)).sum())


    def _calculate_chaos_score(self):
        """
        Calculates the 'State' of the colony (Converged vs. Chaotic).
        Uses Coefficient of Variation (CV) to measure how 'spiky' the pheromones are.
        """
        mean_phero = np.mean(self.pheromone)
        std_phero = np.std(self.pheromone)
        
        # CV = Standard Deviation / Mean
        # Low CV (0.0) = Flat/Random (Chaos)
        # High CV (>1.0) = Spiky/Converged (Order)
        cv = std_phero / (mean_phero + 1e-9)
        
        # Squash to 0-1 range using Tanh
        # Result: 1.0 = Chaos/Exploration, 0.0 = Converged/Stagnation
        order_score = np.tanh(cv) 
        return 1.0 - order_score
    
    def _get_max_pheromone(self):
        """Returns the maximum pheromone value, normalized by the upper clip limit (5.0)."""
        # The upper clip limit is defined as 5.0 in ACO_Solver._update_pheromones
        max_phero = np.max(self.pheromone)
        log.debug(f"the max pheromone is {max_phero}")
        return float(max_phero / 5.0)
        
    def get_state(self):
        """
        Extract features for the Neural Network.
        Returns 4 features: [Norm_Makespan, Chaos_Score, Max_Pheromone_Norm, Norm_Stagnation]
        
        This aligns with ParameterController(input_dim=4).
        """
        # Feature 1: Normalized Makespan (Absolute Quality)
        current_makespan = (
            self.global_best_schedule.makespan() 
            if self.global_best_schedule is not None 
            else self.c_opt * 2.0 # Fallback safety value
        )
        
        # CRITICAL FIX: Ensure C_opt is set and is non-zero
        safe_c_opt = max(self.c_opt, 1e-9)
        norm_makespan = current_makespan / safe_c_opt
        
        # Feature 2: Chaos Score
        chaos_score = self._calculate_chaos_score()
        
        # Feature 3: Max Pheromone Concentration (Exploitation Strength)
        max_phero_norm = self._get_max_pheromone()

        # Feature 4: Stagnation Counter (Normalized)
        norm_stagnation = min(self.stagnation_counter / 50.0, 1.0)

        return torch.tensor([
            norm_makespan,
            chaos_score,
            max_phero_norm,
            norm_stagnation 
        ], dtype=torch.float32)
    
    def run_batch(self, num_iterations=10):
        """
        Runs the ACO for a batch sequentially.
        Returns: (improvement, avg_chaos, batch_best_makespan)
        """
        start_makespan = self.global_best_schedule.makespan() if self.global_best_schedule else float('inf')
        batch_chaos_scores = []
        iter_best_makespans = []
        

        log.info(f"Start of batch: pheromone sum = {np.sum(self.pheromone):.2f}")

        for _ in range(num_iterations):
            ant_schedules = []
            ant_sequences = []

            # --- Build solutions sequentially ---
            for _ in range(self.num_ants):
                sched, seq = self._build_ant_solution()
                ant_schedules.append(sched)
                ant_sequences.append(seq)
        
            # --- Update pheromones ---
            self._update_pheromones(ant_schedules, ant_sequences)

            # --- Measure chaos ---
            chaos = self._calculate_chaos_score()
            batch_chaos_scores.append(chaos)

            # --- Track batch best ---
            iter_best_idx = int(np.argmin([s.makespan() for s in ant_schedules]))
            iter_best_schedule = ant_schedules[iter_best_idx]
            iter_best_seq = ant_sequences[iter_best_idx]

            iter_best_makespans.append(iter_best_schedule.makespan())



            # CSV LOGGING FOR THIS ITERATION

            entropy = SteppableACO.get_pheromone_entropy(self)
            pheromone_sum = float(np.sum(self.pheromone))
            best_mk = iter_best_schedule.makespan()
            global_best = self.global_best_schedule.makespan()

            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.iteration_counter,
                    best_mk,
                    global_best,
                    chaos,
                    entropy,
                    pheromone_sum,
                    self.alpha,
                    self.beta,
                    self.rho,
                    self.stagnation_counter
                ])

            self.iteration_counter += 1

            # --- Update global best ---
            if self.global_best_schedule is None or iter_best_schedule.makespan() < self.global_best_schedule.makespan():
                self.global_best_schedule = iter_best_schedule
                self.global_best_seq = iter_best_seq

        # --- Post-batch analysis ---
        new_makespan = self.global_best_schedule.makespan()
        avg_chaos = np.mean(batch_chaos_scores)
        batch_best_makespan = min(iter_best_makespans)
        log.info(f"End of batch: pheromone sum = {np.sum(self.pheromone):.2f}")

        # Update stagnation counter
        if new_makespan < self.last_best_makespan:
            self.stagnation_counter = 0
            self.last_best_makespan = new_makespan
        else:
            self.stagnation_counter += 1

        log.info(f"[Sequential] Batch Best Makespan: {batch_best_makespan}, Avg Chaos: {avg_chaos:.3f}")

        improvement = start_makespan - new_makespan
        return improvement, avg_chaos

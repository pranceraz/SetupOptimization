#ants_env.py

from ants import ACO_Solver
import numpy as np
import torch
import logging,random
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor


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
    def set_params(self, alpha, beta, rho):
        """Update parameters dynamically."""
        self.alpha = float(np.clip(alpha, 0.1, 5.0))
        self.beta = float(np.clip(beta, 0.1, 5.0))
        self.rho = float(np.clip(rho, 0.01, 0.6))

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

    def get_state(self):
        """
        Extract features for the Neural Network.
        Returns: [Norm_Makespan, Chaos_Score, Alpha, Beta, Rho, Stagnation]
        """
        # Feature 1: Normalized Makespan
        current_makespan = self.global_best_schedule.makespan() if self.global_best_schedule else 5000
        if self.initial_makespan == 5000.0 and self.global_best_schedule:
             self.initial_makespan = current_makespan
        # Use first makespan (or known upper bound for this problem size)
        norm_makespan = current_makespan / self.initial_makespan

        
        # Feature 2: Chaos Score (The "State" you want to see)
        chaos_score = self._calculate_chaos_score()
        
        # Feature 3: Stagnation Counter (Normalized)
        norm_stagnation = min(self.stagnation_counter / 50.0, 1.0)

        return torch.tensor([
            norm_makespan,
            chaos_score,
            self.alpha / 5.0,
            self.beta / 5.0,
            self.rho,
            norm_stagnation 
        ], dtype=torch.float32)
    

    @staticmethod
    def _build_solution_static(args):
        """
        Static method for parallel building. Supports deterministic per-thread seeds.
        """
        self_obj, seed = args
        np.random.seed(seed)
        random.seed(seed)
        return self_obj._build_ant_solution()

    def _parallel_build_solutions(self):
        """
        Builds solutions in parallel using threads (lighter than multiprocessing on Windows).
        """
        seeds = list(range(self.num_ants))
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                SteppableACO._build_solution_static,
                [(self, s) for s in seeds]
            ))

        ant_schedules, ant_sequences = zip(*results)
        return list(ant_schedules), list(ant_sequences)

    def run_batch(self, num_iterations=10):
        """
        Runs the ACO for a batch sequentially.
        Returns: (improvement, avg_chaos, batch_best_makespan)
        """
        start_makespan = self.global_best_schedule.makespan() if self.global_best_schedule else float('inf')
        batch_chaos_scores = []
        iter_best_makespans = []

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

            # --- Update global best ---
            if self.global_best_schedule is None or iter_best_schedule.makespan() < self.global_best_schedule.makespan():
                self.global_best_schedule = iter_best_schedule
                self.global_best_seq = iter_best_seq

        # --- Post-batch analysis ---
        new_makespan = self.global_best_schedule.makespan()
        avg_chaos = np.mean(batch_chaos_scores)
        batch_best_makespan = min(iter_best_makespans)

        # Update stagnation counter
        if new_makespan < self.last_best_makespan:
            self.stagnation_counter = 0
            self.last_best_makespan = new_makespan
        else:
            self.stagnation_counter += 1

        log.info(f"[Sequential] Batch Best Makespan: {batch_best_makespan}, Avg Chaos: {avg_chaos:.3f}")

        improvement = start_makespan - new_makespan
        return improvement, avg_chaos

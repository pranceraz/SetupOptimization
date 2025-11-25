#ants_env.py

from ants import ACO_Solver
import numpy as np
import torch
import logging

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
        self.rho = float(np.clip(rho, 0.01, 0.99))

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

    def run_batch(self, num_iterations=10):
        """
        Runs the ACO for a batch.
        Returns: (improvement, average_chaos_score)
        """
        start_makespan = self.global_best_schedule.makespan() if self.global_best_schedule else float('inf')
        batch_chaos_scores = []
        iter_best_makespans = []   
        
        for _ in range(num_iterations):
            ant_schedules = []
            ant_sequences = []

            # 1. Build Solutions
            for _ in range(self.num_ants):
                sched, seq = self._build_ant_solution()
                ant_schedules.append(sched)
                ant_sequences.append(seq)

            # 2. Update Pheromones (Normal ACO logic)
            self._update_pheromones(ant_schedules, ant_sequences)

            # 3. Measure State (Chaos)
            chaos = self._calculate_chaos_score()
            batch_chaos_scores.append(chaos)

            # 4. Update Best
            iter_best_idx = int(np.argmin([s.makespan() for s in ant_schedules]))
            iter_best_schedule = ant_schedules[iter_best_idx]
            iter_best_seq = ant_sequences[iter_best_idx]
            iter_best_makespans.append(iter_best_schedule.makespan())

            if self.global_best_schedule is None or iter_best_schedule.makespan() < self.global_best_schedule.makespan():
                self.global_best_schedule = iter_best_schedule
                self.global_best_seq = iter_best_seq
        
        # --- Post-Batch Analysis ---
        new_makespan = self.global_best_schedule.makespan()
        avg_chaos = np.mean(batch_chaos_scores)
        
        # Update Stagnation Counter (Just for tracking/NN input)
        if new_makespan < self.last_best_makespan:
            self.stagnation_counter = 0
            self.last_best_makespan = new_makespan
        else:
            self.stagnation_counter += 1
           

        improvement = start_makespan - new_makespan
        
        return improvement, avg_chaos,iter_best_makespans
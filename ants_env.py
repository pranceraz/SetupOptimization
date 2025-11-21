from ants import ACO_Solver
import numpy as np
import torch

class SteppableACO(ACO_Solver):
    def set_params(self, alpha, beta, rho):
        """Update parameters dynamically."""
        self.alpha = float(np.clip(alpha, 0.1, 5.0))
        self.beta = float(np.clip(beta, 0.1, 5.0))
        self.rho = float(np.clip(rho, 0.01, 0.99))

    def get_state(self):
        """
        Extract features for the Neural Network.
        Returns a tensor representing the current optimization state.
        """
        # Feature 1: Normalized Makespan (lower is better, so we invert or scale)
        # We use the current best vs a baseline (heuristic or initial)
        current_makespan = self.global_best_schedule.makespan() if self.global_best_schedule else 10000
        
        # Feature 2: Pheromone Entropy (Are we converging?)
        # High entropy = confused/exploring, Low entropy = converged
        pheromones = self.pheromone / (self.pheromone.sum() + 1e-6)
        entropy = -np.sum(pheromones * np.log(pheromones + 1e-6))
        
        # Feature 3: Current Parameters
        return torch.tensor([
            current_makespan,
            entropy,
            self.alpha,
            self.beta,
            self.rho
        ], dtype=torch.float32)

    def run_batch(self, num_iterations=10):
        """
        Runs the ACO for a small batch of iterations using current params.
        Returns the improvement in makespan (Reward).
        """
        initial_makespan = self.global_best_schedule.makespan() if self.global_best_schedule else float('inf')
        
        # --- Logic lifted from your solve() method ---
        # We assume self.global_best_schedule is initialized externally or in __init__
        
        for _ in range(num_iterations):
            ant_schedules = []
            ant_sequences = []

            for _ in range(self.num_ants):
                sched, seq = self._build_ant_solution()
                ant_schedules.append(sched)
                ant_sequences.append(seq)

            self._update_pheromones(ant_schedules, ant_sequences)

            iter_best_idx = int(np.argmin([s.makespan() for s in ant_schedules]))
            iter_best_schedule = ant_schedules[iter_best_idx]
            iter_best_seq = ant_sequences[iter_best_idx]

            if self.global_best_schedule is None or iter_best_schedule.makespan() < self.global_best_schedule.makespan():
                self.global_best_schedule = iter_best_schedule
                self.global_best_seq = iter_best_seq
        
        new_makespan = self.global_best_schedule.makespan()
        
        # Reward: Positive if we improved, 0 otherwise (or small negative for time cost)
        improvement = initial_makespan - new_makespan
        return improvement
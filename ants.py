import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time

class Job:
    """Represents a job with processing time and due date"""
    def __init__(self, job_id: int, processing_time: float, due_date: float = None):
        self.id = job_id
        self.processing_time = processing_time
        self.due_date = due_date if due_date is not None else processing_time * 2 #weird auto assign watch out  

    def __repr__(self):
        return f"Job({self.id}, p={self.processing_time}, d={self.due_date})"

class ACO_Scheduler:
    """
    Ant Colony Optimization for Single Machine Scheduling 
    with Sequence-Dependent Setup Times (SMSP-SDST)
    """
    def __init__(self, 
                 jobs: List[Job],
                 setup_times: np.ndarray,
                 num_ants: int = 20,
                 num_iterations: int = 100,
                 alpha: float = 1.0,  # pheromone importance
                 beta: float = 2.0,   # heuristic importance
                 rho: float = 0.1,    # evaporation rate
                 q0: float = 0.9,     # exploitation vs exploration
                 initial_pheromone: float = 0.1):
        """
        Initialize ACO scheduler

        Args:
            jobs: List of Job objects
            setup_times: Matrix of setup times (n x n)
            num_ants: Number of ants in colony
            num_iterations: Number of iterations
            alpha: Pheromone trail importance
            beta: Heuristic information importance
            rho: Pheromone evaporation rate
            q0: Exploitation parameter (0-1)
            initial_pheromone: Initial pheromone value
        """
        self.jobs = jobs
        self.n_jobs = len(jobs)
        self.setup_times = setup_times
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0

        # Initialize pheromone matrix #TODO
        self.pheromone = np.ones((self.n_jobs, self.n_jobs)) * initial_pheromone

        # Best solution tracking
        self.best_solution = None
        self.best_cost = float('inf')
        self.cost_history = []

    def calculate_heuristic(self) -> np.ndarray:
        """
        Calculate heuristic information matrix
        Using a combination of processing time and setup time
        """
        eta = np.zeros((self.n_jobs, self.n_jobs))

        for i in range(self.n_jobs):
            for j in range(self.n_jobs):
                if i != j:
                    # Heuristic: prefer jobs with shorter processing + setup time
                    total_time = self.jobs[j].processing_time + self.setup_times[i, j]
                    eta[i, j] = 1.0 / (total_time + 1e-10)

        return eta

    def calculate_makespan(self, sequence: List[int]) -> float:
        """Calculate total completion time (makespan) for a sequence"""
        current_time = 0
        makespan = 0

        for idx, job_idx in enumerate(sequence):
            # Add setup time if not first job
            if idx > 0:
                prev_job_idx = sequence[idx - 1]
                current_time += self.setup_times[prev_job_idx, job_idx]

            # Add processing time
            current_time += self.jobs[job_idx].processing_time
            makespan = current_time

        return makespan

    def calculate_total_tardiness(self, sequence: List[int]) -> float:
        """Calculate total tardiness for a sequence"""
        current_time = 0
        total_tardiness = 0

        for idx, job_idx in enumerate(sequence):
            # Add setup time if not first job
            if idx > 0:
                prev_job_idx = sequence[idx - 1]
                current_time += self.setup_times[prev_job_idx, job_idx]

            # Add processing time
            current_time += self.jobs[job_idx].processing_time

            # Calculate tardiness
            tardiness = max(0, current_time - self.jobs[job_idx].due_date)
            total_tardiness += tardiness

        return total_tardiness

    def calculate_total_completion_time(self, sequence: List[int]) -> float:
        """Calculate sum of completion times for a sequence"""
        current_time = 0
        total_completion_time = 0

        for idx, job_idx in enumerate(sequence):
            # Add setup time if not first job
            if idx > 0:
                prev_job_idx = sequence[idx - 1]
                current_time += self.setup_times[prev_job_idx, job_idx]

            # Add processing time
            current_time += self.jobs[job_idx].processing_time

            # Add to total completion time
            total_completion_time += current_time

        return total_completion_time

    def construct_solution(self, eta: np.ndarray) -> Tuple[List[int], float]:
        """
        Construct a solution using ACO rules
        Returns: (sequence, cost)
        """
        sequence = []
        available_jobs = set(range(self.n_jobs))

        # Start with random job
        current_job = np.random.choice(list(available_jobs))
        sequence.append(current_job)
        available_jobs.remove(current_job)

        # Build rest of sequence
        while available_jobs:
            # Calculate probabilities for next job selection
            probabilities = []
            available_list = list(available_jobs)

            for next_job in available_list:
                tau = self.pheromone[current_job, next_job] ** self.alpha
                eta_val = eta[current_job, next_job] ** self.beta
                probabilities.append(tau * eta_val)

            # Normalize probabilities
            prob_sum = sum(probabilities)
            if prob_sum > 0:
                probabilities = [p / prob_sum for p in probabilities]
            else:
                probabilities = [1.0 / len(available_list)] * len(available_list)

            # Exploitation vs exploration
            if np.random.random() < self.q0:
                # Exploitation: choose best
                next_idx = np.argmax(probabilities)
            else:
                # Exploration: probabilistic selection
                next_idx = np.random.choice(len(available_list), p=probabilities)

            next_job = available_list[next_idx]
            sequence.append(next_job)
            available_jobs.remove(next_job)
            current_job = next_job

        # Calculate cost (using total completion time as default)
        cost = self.calculate_total_completion_time(sequence)

        return sequence, cost

    def update_pheromones(self, all_solutions: List[Tuple[List[int], float]]):
        """Update pheromone trails based on solutions"""
        # Evaporation
        self.pheromone *= (1 - self.rho)

        # Add pheromone from best solution (elitist strategy)
        best_sequence, best_cost = min(all_solutions, key=lambda x: x[1])

        if best_cost < self.best_cost:
            self.best_solution = best_sequence
            self.best_cost = best_cost

        # Deposit pheromone on best path
        delta_tau = 1.0 / (best_cost + 1e-10)

        for i in range(len(best_sequence) - 1):
            current_job = best_sequence[i]
            next_job = best_sequence[i + 1]
            self.pheromone[current_job, next_job] += delta_tau

        # Optional: Add pheromone from all ants (weighted)
        for sequence, cost in all_solutions:
            delta = 0.1 / (cost + 1e-10)  # Smaller contribution
            for i in range(len(sequence) - 1):
                current_job = sequence[i]
                next_job = sequence[i + 1]
                self.pheromone[current_job, next_job] += delta

    def optimize(self, verbose: bool = True) -> Tuple[List[int], float]:
        """
        Run ACO optimization

        Returns:
            best_sequence: Optimal job sequence
            best_cost: Cost of best sequence
        """
        eta = self.calculate_heuristic()

        for iteration in range(self.num_iterations):
            # Generate solutions for all ants
            solutions = []
            for ant in range(self.num_ants):
                sequence, cost = self.construct_solution(eta)
                solutions.append((sequence, cost))

            # Update pheromones
            self.update_pheromones(solutions)

            # Track progress
            self.cost_history.append(self.best_cost)

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.num_iterations}, "
                      f"Best Cost: {self.best_cost:.2f}")

        return self.best_solution, self.best_cost

    def print_solution(self):
        """Print detailed solution information"""
        if self.best_solution is None:
            print("No solution found yet. Run optimize() first.")
            return

        print("\n" + "="*60)
        print("OPTIMAL SCHEDULE")
        print("="*60)
        print(f"Job Sequence: {[self.jobs[i].id for i in self.best_solution]}")
        print(f"Total Cost (Sum of Completion Times): {self.best_cost:.2f}")
        print(f"Total Tardiness: {self.calculate_total_tardiness(self.best_solution):.2f}")
        print(f"Makespan: {self.calculate_makespan(self.best_solution):.2f}")

        print("\nDetailed Schedule:")
        print("-"*60)
        current_time = 0

        for idx, job_idx in enumerate(self.best_solution):
            job = self.jobs[job_idx]

            # Setup time
            if idx > 0:
                prev_job_idx = self.best_solution[idx - 1]
                setup = self.setup_times[prev_job_idx, job_idx]
                current_time += setup
                print(f"  Setup time from Job {self.jobs[prev_job_idx].id} "
                      f"to Job {job.id}: {setup:.2f}")

            # Processing
            start_time = current_time
            current_time += job.processing_time
            completion_time = current_time
            tardiness = max(0, completion_time - job.due_date)

            print(f"Job {job.id}: Start={start_time:.2f}, "
                  f"Process={job.processing_time:.2f}, "
                  f"Complete={completion_time:.2f}, "
                  f"Due={job.due_date:.2f}, "
                  f"Tardiness={tardiness:.2f}")

        print("="*60)

    def plot_convergence(self):
        """Plot convergence curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Cost', fontsize=12)
        plt.title('ACO Convergence Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('aco_convergence.png', dpi=300)
        print("Convergence plot saved as 'aco_convergence.png'")


def main():
    """Example usage"""
    print("ACO for Single Machine Scheduling with Sequence-Dependent Setup Times")
    print("="*70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate test jobs
    n_jobs = 10
    jobs = []
    for i in range(n_jobs):
        processing_time = np.random.uniform(5, 20)
        due_date = np.random.uniform(30, 100)
        jobs.append(Job(i, processing_time, due_date))

    print(f"\nTest Problem: {n_jobs} jobs")
    print("\nJob Details:")
    for job in jobs:
        print(f"  {job}")

    # Generate sequence-dependent setup times
    setup_times = np.random.uniform(1, 10, (n_jobs, n_jobs))
    np.fill_diagonal(setup_times, 0)  # No setup when same job

    print("\nSequence-Dependent Setup Time Matrix:")
    print(setup_times.round(2))

    # Run ACO optimization
    print("\n" + "="*70)
    print("Running ACO Optimization...")
    print("="*70)

    aco = ACO_Scheduler(
        jobs=jobs,
        setup_times=setup_times,
        num_ants=15,
        num_iterations=50,
        alpha=1.0,
        beta=2.5,
        rho=0.1,
        q0=0.9
    )

    start_time = time.time()
    best_sequence, best_cost = aco.optimize(verbose=True)
    end_time = time.time()

    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")

    # Print solution
    aco.print_solution()


if __name__ == "__main__":
    main()

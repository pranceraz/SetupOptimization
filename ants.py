import numpy as np
import random
from job_shop_lib import (
    JobShopInstance, 
    SolutionSchedule, 
    benchmarking
)

class ACO_Solver:
    """
    A skeleton class for an Ant Colony Optimization solver that is
    compatible with job_shop_lib.
    
    "Compatible" means it takes a JobShopInstance as input and
    returns a SolutionSchedule as output.
    alpha : pheromone weight, beta: heuristic weight, rho(evaporation rate)
    """
    def __init__(self, num_ants: int, iterations: int, 
                 alpha: float, beta: float, rho: float):
        """
        Initializes the ACO solver with its hyperparameters.
        
        Args:
            num_ants (int): Number of ants (solutions) to build per iteration.
            iterations (int): Number of "generations" to run.
            alpha (float): Pheromone influence factor.
            beta (float): Heuristic influence factor.
            rho (float): Pheromone evaporation rate (e.g., 0.1).
        """
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha  # Pheromone influence
        self.beta = beta    # Heuristic influence
        self.rho = rho      # Evaporation rate
        
        # This will hold our pheromone values.
        # We use a simple 1D array: one value per *operation*.
        # This represents the "desirability" of scheduling an operation *now*.
        self.pheromone_matrix = None
        
        self.global_best_schedule = None

    def solve(self, instance: JobShopInstance) -> SolutionSchedule:
        """
        Main solving loop. This is the "compatible" entry point.
        """
        # Initialize pheromones (e.g., all to 1.0)
        self._initialize_pheromones(instance)
        
        for _ in range(self.iterations):
            all_ant_schedules = []
            
            # 1. CONSTRUCT SOLUTIONS (All Ants)
            for _ in range(self.num_ants):
                # This is the core logic: a single ant builds one
                # full, valid schedule.
                schedule = self._build_ant_solution(instance)
                all_ant_schedules.append(schedule)
            
            # 2. UPDATE PHEROMONES
            self._update_pheromones(all_ant_schedules)
            
            # 3. UPDATE GLOBAL BEST
            # Keep track of the best solution found so far
            for schedule in all_ant_schedules:
                if (self.global_best_schedule is None or 
                    schedule.makespan < self.global_best_schedule.makespan):
                    self.global_best_schedule = schedule
            
            print(f"Iteration {_}: Best Makespan = {self.global_best_schedule.makespan}")

        return self.global_best_schedule

    def _initialize_pheromones(self, instance: JobShopInstance):
        """
        Initializes the pheromone matrix.
        A simple model is one pheromone value per operation.
        """
        # A 1D array of size [num_operations]
        self.pheromone_matrix = np.ones(instance.num_operations)

    def _build_ant_solution(self, instance: JobShopInstance) -> SolutionSchedule:
        """
        **THIS IS THE HARDEST AND MOST IMPORTANT PART**
        Simulates a single ant building a complete schedule.
        
        This requires a discrete-event simulation.
        
        The ant must:
        1. Keep track of "ready time" for each machine and each job.
        2. Maintain a set of "ready operations" (operations whose
           job-predecessor is done).
        3. At each "step", choose one operation from the ready set.
        4. The *choice* is probabilistic, based on:
           P(op_i) ~ (pheromone[op_i])^alpha * (heuristic[op_i])^beta
        5. "Schedule" the chosen op, update machine/job ready times,
           and add the *next* op from that job to the ready set.
        6. Store the sequence of operations for each machine.
        """
        
        # --- This is a simplified placeholder ---
        # The real implementation is a complex loop.
        
        # The key to "compatibility" is to return a SolutionSchedule.
        # The easiest way is to build the machine-level sequences.
        
        # `job_sequences[m]` = list of op_ids run on machine `m`
        job_sequences = [[] for _ in range(instance.num_machines)]
        
        # ---
        # --- YOUR CORE ANT SIMULATION LOGIC GOES HERE ---
        # This logic would populate the `job_sequences` list.
        # This is a complex simulation that is the heart of your
        # ACO implementation.
        # ---
        
        # --- Placeholder: Just make a random (but valid) schedule ---
        # This uses a simple "dispatching rule" as a stand-in
        # to show how to build the SolutionSchedule object.
        
        # This helper structure tracks the *next* operation for each job
        next_op_idx = [0] * instance.num_jobs
        
        # Pointers to the actual Operation objects
        job_op_pointers = [instance.jobs[j][0] for j in range(instance.num_jobs)]
        
        # Keep track of when machines and jobs are free
        machine_available_time = [0] * instance.num_machines
        job_available_time = [0] * instance.num_jobs
        
        num_scheduled = 0
        
        # This set holds the op_ids that are "ready" to be scheduled
        ready_ops = set(op.operation_id for op in job_op_pointers)
        
        while num_scheduled < instance.num_operations:
            # --- This is where your ACO logic would be ---
            # 1. Get all ops in `ready_ops`
            # 2. Calculate transition probabilities for each
            #    (using self.pheromone_matrix and a heuristic)
            # 3. Probabilistically *choose* one operation
            
            # --- Placeholder "ant" (just picks randomly) ---
            chosen_op_id = random.choice(list(ready_ops))
            ready_ops.remove(chosen_op_id)
            
            op = instance.get_operation(chosen_op_id)
            job_id = op.job_id
            machine_id = op.machine_id
            
            # Calculate start time
            start_time = max(machine_available_time[machine_id], 
                             job_available_time[job_id])
            end_time = start_time + op.duration
            
            # Update "clocks"
            machine_available_time[machine_id] = end_time
            job_available_time[job_id] = end_time
            
            # Add to the machine's sequence
            job_sequences[machine_id].append(op.operation_id)
            
            # Add the *next* operation from this job to the ready set
            next_op_idx[job_id] += 1
            if next_op_idx[job_id] < instance.num_operations_per_job(job_id):
                next_op = instance.jobs[job_id][next_op_idx[job_id]]
                ready_ops.add(next_op.operation_id)
                
            num_scheduled += 1

        # This is the "compatible" part. We use the library's
        # constructor to build a valid SolutionSchedule object
        # from the sequences our ant generated.
        return SolutionSchedule.from_job_sequences(
            instance=instance, 
            job_sequences=job_sequences
        )

    def _update_pheromones(self, all_ant_schedules: list[SolutionSchedule]):
        """
        Updates the pheromone matrix based on the ants' performance.
        """
        # 1. Evaporation
        self.pheromone_matrix *= (1 - self.rho)
        
        # 2. Deposition
        # Add pheromone based on the *quality* of the schedules
        for schedule in all_ant_schedules:
            # Reward is 1 / makespan (we want to minimize makespan)
            reward = 1.0 / schedule.makespan
            
            # ---
            # --- YOUR DEPOSITION LOGIC HERE ---
            # This logic needs to update self.pheromone_matrix
            # based on the `schedule` (e.g., which operations
            # were used in this good/bad schedule)
            # ---
            
            # Example (if you deposit on all ops in the schedule):
            for op in schedule.instance.operations:
                 # This is a naive update. A better one would
                 # reward operations that were scheduled *early*.
                 self.pheromone_matrix[op.operation_id] += reward
                 
        # (Optional) Add extra "elitist" pheromone 
        # for the single best schedule found.
        pass


if __name__ == "__main__":
    # Load the instance
    instance = benchmarking.load_benchmark_instance("ft06")
    
    # Create the solver
    aco_solver = ACO_Solver(
        num_ants=10, 
        iterations=5,  # Keep low for a quick test
        alpha=1.0, 
        beta=1.0, 
        rho=0.1
    )
    
    # Solve
    # The 'solve' method is compatible and returns a SolutionSchedule
    print("Starting ACO Solver...")
    best_solution = aco_solver.solve(instance)
    
    print("\n--- Solver Finished ---")
    print(f"Best makespan found: {best_solution.makespan}")
    print(f"(Optimal makespan for 'ft06' is 55)")
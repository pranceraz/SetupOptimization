#ants.py
import numpy as np
import random
import logging
from typing import List
from multiprocessing import Pool, cpu_count
from job_shop_lib import (
    JobShopInstance,
    benchmarking,
    ScheduledOperation,
    Schedule,
    Operation
)




logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

log = logging.getLogger(__name__)
np.random.seed(42)

# MACHINE_1 = 0
# MACHINE_2 = 1
# MACHINE_3 = 2

# job_1 = [Operation(MACHINE_1, 3), Operation(MACHINE_2, 3), Operation(MACHINE_3, 3)]
# job_2 = [Operation(MACHINE_1, 2), Operation(MACHINE_3, 3), Operation(MACHINE_2, 4)]
# job_3 = [Operation(MACHINE_2, 3), Operation(MACHINE_1, 2), Operation(MACHINE_3, 1)]

# jobs = [job_1, job_2, job_3]





class ACO_Solver:
    """
    Ant Colony Optimization solver compatible with job_shop_lib.
    alpha: pheromone weight; beta: heuristic weight; rho: evaporation rate.
    """
    def __init__(self, instance: JobShopInstance, num_ants: int, iterations: int, 
                 alpha: float, beta: float, rho: float, q: float = 1.0,
                 elitist: bool = False, elitist_factor: int = 1):
        self.instance = instance
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q  # scaling for deposition
        self.elitist = elitist
        self.elitist_factor = elitist_factor
        self.num_ops = self.instance.num_operations

        self.pheromone = np.full((self.num_ops + 1, self.num_ops),0.5, dtype=np.float64)
        self.op_list: List[Operation] = []
        for job in self.instance.jobs:
            for op in job:
                self.op_list.append(op)
        self.heuristic_cache = np.zeros(self.num_ops, dtype=np.float32)
        
        for job in self.instance.jobs:
            accumulated_duration = 0
            # Walk backwards from the last op of the job to the first
            for op in reversed(job):
                accumulated_duration += op.duration
                self.heuristic_cache[op.operation_id] = accumulated_duration

        self.global_best_schedule: Schedule | None = None
        self.global_best_seq: list[int] | None = None  # sequence of op_ids in scheduling order

    def solve(self) -> Schedule:
        """
        Main solving loop. Returns a Schedule.
        """
        # Initialize with one ant
        init_schedule, init_seq = self._build_ant_solution()
        self.global_best_schedule = init_schedule
        self.global_best_seq = init_seq
        print(f"Initial Makespan: {self.global_best_schedule.makespan()}")

        for i in range(self.iterations):

            ant_schedules, ant_sequences = self._parallel_build_solutions()
            # Construct solutions
            for _ in range(self.num_ants):
                sched, seq = self._build_ant_solution()
                ant_schedules.append(sched)
                ant_sequences.append(seq)

            # Update pheromones
            self._update_pheromones(ant_schedules, ant_sequences)

            # Iteration best
            iter_best_idx = int(np.argmin([s.makespan() for s in ant_schedules]))
            iter_best_schedule = ant_schedules[iter_best_idx]
            iter_best_seq = ant_sequences[iter_best_idx]

            # Update global best
            if iter_best_schedule.makespan() < self.global_best_schedule.makespan():
                self.global_best_schedule = iter_best_schedule
                self.global_best_seq = iter_best_seq

            log.debug(f"Iteration {i+1}: Best Makespan = {iter_best_schedule.makespan()}, "
                  f"Global Best = {self.global_best_schedule.makespan()}")

        return self.global_best_schedule

    def _visibility(self, op: Operation) -> float:
        # Greedy heuristic: inverse of duration (avoid div by zero)
        #return 1.0 / (op.duration + 1e-6)
        return self.heuristic_cache[op.operation_id]

    # def _ant_brain(self, ready_ops: list[int],last_op:int) -> int:
    #     """
    #     Selects one operation id from ready_ops using pheromone and heuristic.
    #     """
    #     # Compute scores
    #     scores = []
    #     log_scores=[]
    #     for op_id in ready_ops:
    #         tau = self.pheromone[last_op,op_id]
    #         eta = self._visibility(self.op_list[op_id])

    #         log_tau = np.log(tau + 1e-9)
    #         log_eta = np.log(eta + 1e-9)

    #         log_score = (self.alpha * log_tau) + (self.beta * log_eta)
    #         log_scores.append(log_score)

    #     log_scores = np.array(log_scores)
    #     log_scores_shifted = log_scores - np.max(log_scores)
    #     unnormalized_probs = np.exp(log_scores_shifted)
    #     probs = unnormalized_probs / unnormalized_probs.sum()
    #     return int(np.random.choice(ready_ops, p=probs))

    def _ant_brain(self, ready_ops: list[int], last_op: int) -> int:
        """
        Vectorized selection of an operation from ready_ops using pheromone and heuristic.
        """
        ready_ops_arr = np.array(ready_ops, dtype=np.int32)

        # Pheromones for last_op -> ready_ops
        taus = self.pheromone[last_op, ready_ops_arr]
        tau_min, tau_max = np.min(taus), np.max(taus)
        taus_scaled = 0.5 + 0.5 * (taus - tau_min) / max(1e-12, (tau_max - tau_min))
        # Heuristic values (cached)
        etas = self.heuristic_cache[ready_ops_arr]
        eta_min, eta_max = np.min(etas), np.max(etas)
        etas_scaled = 0.5 + 0.5 * (etas - eta_min) / max(1e-12, (eta_max - eta_min))        #log.debug(taus)   

       # log.debug(taus)
        # Log-space scoring
        log_scores = self.alpha * np.log(taus_scaled + 1e-9) + self.beta * np.log(etas_scaled + 1e-9)

        # Numerical stability: shift by max
        log_scores -= np.max(log_scores)

        # Convert to probabilities
        probs = np.exp(log_scores)
        probs /= probs.sum()

        # Choose operation
        choice = np.random.choice(ready_ops_arr, p=probs)
        return int(choice)



    def _build_ant_solution(self) -> tuple[Schedule, list[int]]:
        """
        Builds a feasible schedule by repeatedly picking from ready ops.
        Returns (Schedule, sequence_of_op_ids).
        """
        scheduled_ops: list[ScheduledOperation] = []
        seq: list[int] = []

        # Track next operation index per job
        next_op_idx = [0] * self.instance.num_jobs
        job_op_pointers = [self.instance.jobs[j][0] for j in range(self.instance.num_jobs)]

        machine_available_time = [0] * self.instance.num_machines
        job_available_time = [0] * self.instance.num_jobs

        num_scheduled = 0
        ready_ops = set(op.operation_id for op in job_op_pointers)

        last_op = self.num_ops

        while num_scheduled < self.num_ops:
            chosen_op_id = self._ant_brain(list(ready_ops),last_op=last_op)
            ready_ops.remove(chosen_op_id)

            op = self.op_list[chosen_op_id]
            job_id = op.job_id
            machine_id = op.machine_id

            # Earliest start respecting machine and job availability
            start_time = max(machine_available_time[machine_id], job_available_time[job_id])
            end_time = start_time + op.duration

            machine_available_time[machine_id] = end_time
            
            job_available_time[job_id] = end_time

            scheduled_ops.append(ScheduledOperation(
                operation=op,
                start_time=start_time,
                machine_id= op.machine_id # watch out for bug if multiple machines can perform the same problem
            ))
            seq.append(op.operation_id)

            last_op = chosen_op_id

            # Advance job pointer and add next op if any
            next_op_idx[job_id] += 1
            if next_op_idx[job_id] < len(self.instance.jobs[job_id]):
                nxt = self.instance.jobs[job_id][next_op_idx[job_id]]
                ready_ops.add(nxt.operation_id)

            num_scheduled += 1
        # Build machine-wise schedule
        machine_schedules = [[] for _ in range(self.instance.num_machines)]
        for sop in scheduled_ops:
            machine_schedules[sop.machine_id].append(sop)

        sched = Schedule(instance=self.instance, schedule=machine_schedules)
        
        # for sop in scheduled_ops:
        #     print(f"Op {sop.operation.operation_id}: start={sop.start_time}, end={sop.end_time}, job={sop.operation.job_id}, machine={sop.operation.machine_id}, duration={sop.operation.duration}")

        return sched, seq
    
    # def _update_pheromones(self, ant_schedules: list[Schedule], ant_sequences: list[list[int]]):
    #     """
    #     Standard Ant System update:
    #     - Evaporation: tau = (1 - rho) * tau
    #     - Deposition from all ants
    #     - Extra reinforcement for iteration-best and global-best
    #     """
    #     # 1) Evaporation
    #     self.pheromone *= (1.0 - self.rho)

    #     # 2) Deposition from all ants
    #     batch_best_idx = int(np.argmin([s.makespan() for s in ant_schedules]))
    #     batch_best_schedule = ant_schedules[batch_best_idx]
    #     batch_best_seq = ant_sequences[batch_best_idx]
    #     batch_best_makespan = batch_best_schedule.makespan()

    #     dynamic_Q = (self.q * batch_best_makespan) / self.num_ants

    #     for sched, seq in zip(ant_schedules, ant_sequences):
    #         current_makespan = max(1e-6, float(sched.makespan()))
    #         reward = dynamic_Q / current_makespan
    #         curr = self.num_ops  # dummy start

    #         for next_op in seq:
    #             self.pheromone[curr, next_op] += reward
    #             #log.debug(f"reinforcing decision by {reward}")
    #             curr = next_op

    #     # 3) Reinforce iteration-best
    #     iter_reward = self.q / max(1e-6, batch_best_makespan)
    #     log.debug(f"reinforcing iteration by {iter_reward}")
    #     curr = self.num_ops
    #     for op_id in batch_best_seq:
    #         self.pheromone[curr, op_id] += iter_reward
    #         curr = op_id

    #     # 4) Reinforce global-best (elitist)
    #     if self.elitist and self.global_best_seq is not None:
    #         global_reward = self.q / max(1e-6, self.global_best_schedule.makespan())
    #         log.debug(f"reinforcing global by {global_reward}")
    #         curr = self.num_ops
    #         for op_id in self.global_best_seq:
    #             self.pheromone[curr, op_id] += global_reward
    #             curr = op_id

    #     # 5) Clip pheromones
    #     np.clip(self.pheromone, 0.01, 10.0, out=self.pheromone)
    def _update_pheromones(self, ant_schedules: list[Schedule], ant_sequences: list[list[int]]):
        """
        Pheromone update with old-style loops, but using the new scaled deposition.
        """
        # 1) Evaporation
        self.pheromone *= (1.0 - self.rho)

        # 2) Find iteration-best
        makespans = [s.makespan() for s in ant_schedules]
        batch_best_idx = int(np.argmin(makespans))
        batch_best_seq = ant_sequences[batch_best_idx]
        batch_best_makespan = makespans[batch_best_idx]

        dynamic_Q = self.q  # scaled deposition base

        # 3) Per-ant deposition
        for seq, mk in zip(ant_sequences, makespans):
            reward = dynamic_Q / max(1e-6, mk)
            curr = self.num_ops  # dummy start
            for op_id in seq:
                self.pheromone[curr, op_id] += reward
                curr = op_id
                # log.debug(f"desision {reward}")
        # 4) Iteration-best reinforcement
        iter_reward = dynamic_Q / max(1e-6, batch_best_makespan)
        curr = self.num_ops
        for op_id in batch_best_seq:
            self.pheromone[curr, op_id] += iter_reward
            curr = op_id
            #log.debug(f"desision {iter_reward}")


        # 5) Global-best reinforcement
        if self.elitist and self.global_best_seq is not None:
            global_reward = dynamic_Q / max(1e-6, self.global_best_schedule.makespan())
            curr = self.num_ops
            for op_id in self.global_best_seq:
                self.pheromone[curr, op_id] += global_reward
                curr = op_id

        # 6) Clip pheromones
        np.clip(self.pheromone, 0.01, 5.0, out=self.pheromone)


    @staticmethod
    def _build_solution_static(args):
        self_obj, seed = args
        
        np.random.seed(seed)
        random.seed(seed)

        return self_obj._build_ant_solution()


    def _parallel_build_solutions(self):
        from multiprocessing import Pool, cpu_count
        
        # deterministic seed for each ant
        seeds = [12345 + i for i in range(self.num_ants)]
        
        with Pool(cpu_count()) as pool:
            results = pool.map(
                ACO_Solver._build_solution_static,
                [(self, seeds[i]) for i in range(self.num_ants)]
            )
        
        ant_schedules, ant_sequences = zip(*results)
        return list(ant_schedules), list(ant_sequences)


if __name__ == "__main__":
    instance_name: str = "ta02"
    #instance = benchmarking.load_benchmark_instance(instance_name)
    instance = benchmarking.load_benchmark_instance(instance_name)
    aco_solver = ACO_Solver(
        instance=instance,
        num_ants=200,
        iterations=1200,
        alpha=1,
        beta=2,
        rho=0.5,
        q=1.0,
        elitist=True,
        elitist_factor=1
    )
    print("Starting ACO Solver...")
    best_solution = aco_solver.solve()
    print("\n--- Solver Finished ---")
    print(f"Best makespan found: {best_solution.makespan()}")
    # print(f"for {instance_name} is {instance.metadata})")

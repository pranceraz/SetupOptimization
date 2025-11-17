import numpy as np
import random
from typing import List
from job_shop_lib import (
    JobShopInstance,
    benchmarking,
    ScheduledOperation,
    Schedule,
    Operation
)

class ACO_Solver:
    """
    Ant Colony Optimization solver compatible with job_shop_lib.
    alpha: pheromone weight; beta: heuristic weight; rho: evaporation rate.
    """
    def __init__(self, instance: JobShopInstance, num_ants: int, iterations: int, 
                 alpha: float, beta: float, rho: float, q: float = 1.0,
                 elitist: bool = True, elitist_factor: int = 1):
        self.instance = instance
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q  # scaling for deposition
        self.elitist = elitist
        self.elitist_factor = elitist_factor
        self.op_list: List[Operation] = []
        for job in self.instance.jobs:
            for op in job:
                self.op_list.append(op)

        # One pheromone value per operation id (0..num_operations-1)
        self.pheromone = np.ones(self.instance.num_operations, dtype=np.float64)

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
            ant_schedules = []
            ant_sequences = []

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

            print(f"Iteration {i+1}: Best Makespan = {iter_best_schedule.makespan()}, "
                  f"Global Best = {self.global_best_schedule.makespan()}")

        return self.global_best_schedule

    def _visibility(self, op: Operation) -> float:
        # Greedy heuristic: inverse of duration (avoid div by zero)
        return 1.0 / (op.duration + 1e-6)

    def _ant_brain(self, ready_ops: list[int]) -> int:
        """
        Selects one operation id from ready_ops using pheromone and heuristic.
        """
        # Compute scores
        scores = []
        for op_id in ready_ops:
            tau = self.pheromone[op_id]
            eta = self._visibility(self.op_list[op_id])
            # Compute desirability
            score = (tau ** self.alpha) * (eta ** self.beta)
            scores.append(score)

        total = float(np.sum(scores))
        if total <= 0.0 or not np.isfinite(total):
            # Fallback: uniform random among ready ops
            return random.choice(ready_ops)

        probs = np.array(scores, dtype=np.float64) / total
        # Numerical safety
        probs = probs / probs.sum()
        return int(np.random.choice(ready_ops, p=probs))

    def _build_ant_solution(self) -> tuple[Schedule, list[int]]:
        """
        Builds a feasible semi-active schedule by repeatedly picking from ready ops.
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

        while num_scheduled < self.instance.num_operations:
            chosen_op_id = self._ant_brain(list(ready_ops))
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

    def _update_pheromones(self, ant_schedules: list[Schedule], ant_sequences: list[list[int]]):
        """
        Standard Ant System update:
        - Evaporation: tau = (1 - rho) * tau
        - Deposition: for each ant, for each op in its sequence, add q / makespan
        - Optional elitist: add extra deposition on iteration-best/global-best
        """
        # 1) Evaporation
        self.pheromone *= (1.0 - self.rho)

        # 2) Deposition from all ants
        for sched, seq in zip(ant_schedules, ant_sequences):
            reward = self.q / max(1e-6, float(sched.makespan()))
            # Deposit on operations that were used (sequence encodes order)
            for op_id in seq:
                self.pheromone[op_id] += reward

        # 3) Elitist reinforcement (optional)
        if self.elitist and ant_schedules:
            # Iteration best
            idx = int(np.argmin([s.makespan() for s in ant_schedules]))
            iter_best = ant_schedules[idx]
            iter_seq = ant_sequences[idx]
            iter_reward = self.q / max(1e-6, float(iter_best.makespan()))
            for _ in range(self.elitist_factor):
                for op_id in iter_seq:
                    self.pheromone[op_id] += iter_reward

            # Global best
            if self.global_best_schedule is not None and self.global_best_seq is not None:
                glob_reward = self.q / max(1e-6, float(self.global_best_schedule.makespan()))
                for _ in range(self.elitist_factor):
                    for op_id in self.global_best_seq:
                        self.pheromone[op_id] += glob_reward


if __name__ == "__main__":
    instance_name: str = "ft06"
    instance = benchmarking.load_benchmark_instance(instance_name)
    aco_solver = ACO_Solver(
        instance=instance,
        num_ants=20,
        iterations=100,
        alpha=1.0,
        beta=1.0,
        rho=0.1,
        q=1.0,
        elitist=True,
        elitist_factor=1
    )
    print("Starting ACO Solver...")
    best_solution = aco_solver.solve()
    print("\n--- Solver Finished ---")
    print(f"Best makespan found: {best_solution.makespan()}")
    print(f"for {instance_name} is {instance.metadata})")

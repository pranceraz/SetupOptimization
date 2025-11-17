from job_shop_lib import JobShopInstance,Operation
import job_shop_lib.benchmarking as benchmarking 
#from job_shop_lib.constraint_programming import ORToolsSolver
from job_shop_lib.graphs import build_disjunctive_graph
from job_shop_lib.visualization.graphs import plot_disjunctive_graph
import matplotlib.pyplot as plt

from ants import ACO_Solver




def inspect_instance(instance_name:str = "ft06"):
    """
    Loads the 'ft06' benchmark instance (6 jobs, 6 machines) and
    prints its structure to the console.
    """
    print("Loading 'ft06' benchmark instance...")
    
    # This is the line that loads the problem.
    # 'ft06' is a classic 6x6 problem.
    instance = benchmarking.load_benchmark_instance(instance_name)
    
    print("\n--- Instance Overview ---")
    print(f"Instance Name: {instance.name}")
    print(f"Number of Jobs: {instance.num_jobs}")
    print(f"Number of Machines: {instance.num_machines}")
    print(f"Total Operations: {instance.num_operations}")
    
    # The optimal known makespan for 'ft06' is 55
    print(f"Known Optimal Makespan: {instance.metadata.get('optimum')}")
    
    print("\n--- Job-Operation Structure ---")
    
    # instance.jobs is a list of lists.
    # Each inner list is a job, containing its operations in order.
    for job_id, job in enumerate(instance.jobs):
        print(f"--- Job {job_id} ---")
        
        for op in job:
            # Each 'op' is an Operation object
            print(f"  Operation {op.operation_id:02d} (Job {op.job_id}, Pos {op.position_in_job}): "
                  f"Runs on Machine {op.machine_id} for {op.duration} units")

instance_name: str = "ft06"
instance = benchmarking.load_benchmark_instance(instance_name)
# aco_solver = ACO_Solver(
#     instance=instance,
#     num_ants=20,
#     iterations=1000,
#     alpha=1.0,
#     beta=1.0,
#     rho=0.1,
#     q=1.0,
#     elitist=False,
#     elitist_factor=1
#     )
# print("Starting ACO Solver...")
# best_solution = aco_solver.solve()
# print("\n--- Solver Finished ---")
# print(f"Best makespan found: {best_solution.makespan()}")
# print(f"for {instance_name} is {instance.metadata})")

graph = build_disjunctive_graph(instance)
# print(graph.nodes_by_type)


fig, ax = plot_disjunctive_graph(graph, figsize=(6, 4))

fig.savefig("disjunctive.png", dpi=200, bbox_inches="tight")
plt.close(fig)
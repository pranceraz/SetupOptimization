from job_shop_lib import JobShopInstance,Operation
import job_shop_lib.benchmarking as benchmarking 
from job_shop_lib.constraint_programming import ORToolsSolver

instance = benchmarking.load_benchmark_instance("ta01")

# MACHINE_A = 1
# MACHINE_B= 2
# MACHINE_C = 3

# job_1 = [Operation]

print(f"Loaded {instance.name}: {instance.num_jobs} jobs, {instance.num_machines} machines")


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
inspect_instance("ta01")
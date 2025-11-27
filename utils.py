
import job_shop_lib.benchmarking as benchmarking

def inspect_instance(instance_name:str = "ft06",verbose = False):
    """
    Loads the benchmark instance and
    prints its structure to the console.
    """
    print(f"Loading '{instance_name}' benchmark instance...")
    
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
    if verbose:
        print("\n--- Job-Operation Structure ---")
        
        # instance.jobs is a list of lists.
        # Each inner list is a job, containing its operations in order.
        for job_id, job in enumerate(instance.jobs):
            print(f"--- Job {job_id} ---")
            
            for op in job:
                # Each 'op' is an Operation object
                print(f"  Operation {op.operation_id:02d} (Job {op.job_id}, Pos {op.position_in_job}): "
                    f"Runs on Machine {op.machine_id} for {op.duration} units")
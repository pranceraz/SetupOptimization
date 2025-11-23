import job_shop_lib,utils
from job_shop_lib import JobShopInstance,Operation
import job_shop_lib.benchmarking as benchmarking 
#from job_shop_lib.constraint_programming import ORToolsSolver
from job_shop_lib.graphs import build_disjunctive_graph
from job_shop_lib.visualization.graphs import plot_disjunctive_graph
import matplotlib.pyplot as plt
from job_shop_lib.reinforcement_learning import (
    # MakespanReward,
    SingleJobShopGraphEnv,
    ObservationSpaceKey,
    IdleTimeReward,
    ObservationDict,
)
from job_shop_lib.dispatching.feature_observers import (
    FeatureObserverType,
    FeatureType,
)
from job_shop_lib.dispatching import DispatcherObserverConfig
from ants import ACO_Solver
# import torch
import gymnasium
from gymnasium.spaces import Box
import utils

instance_name: str = "ta02"
instance = benchmarking.load_benchmark_instance(instance_name)

#inspect_instance()
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


# env = SingleJobShopGraphEnv(instance)
# obs, _ = env.reset()
# env.action_space = Box(low=np.array([0.1, 5, 0.1]), high=np.array([0.9, 100, 0.9]), shape=(3,), dtype=np.float32)


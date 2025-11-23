import torch.nn as nn

class GNNACOParamPolicy(nn.Module):
    def __init__(self, graph_input_dim):
        super().__init__()
        # GNN layers using PyTorch Geometric or DGL here
        # replace this with graph conv layers from the original codebase
        self.fc = nn.Linear(graph_input_dim, 3)  # 3 ACO params

    def forward(self, graph_features):
        # Output: [evaporation_rate, num_ants, exploitation_rate]
        out = torch.sigmoid(self.fc(graph_features))
        # Scale and shift to parameter bounds
        return out * torch.tensor([0.8, 95, 0.8]) + torch.tensor([0.1, 5, 0.1])

    def environment_step(env, gnn_output):
        aco_params = gnn_output.detach().cpu().numpy()
        # Pass params into your ACO implementation
        schedule, makespan = run_aco(env.current_instance, aco_params)
        # Compute reward (e.g., -makespan, improvement vs baseline, etc.)
        reward = -makespan  # Or whatever is suitable
        # Update RL environment state as needed (Pablo’s code updates graph based on schedule)
        obs = get_graph_observation(env)
        done = check_if_completed(env, schedule)
        return obs, reward, done
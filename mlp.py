import torch
import torch.nn as nn
import torch.optim as optim 
from torch.distributions import Normal
import logging
logging.basicConfig(level=logging.INFO)
torch.manual_seed(42)

log = logging.getLogger(__name__)
class ParameterController(nn.Module):
    def __init__(self,input_dim, action_dim):
        super(ParameterController,self).__init__()# same as super()
        self.fc1 = nn.Linear(input_dim,64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64,64)
        self.mu_head = nn.Linear(64, action_dim)

        self.sigma_head= nn.Linear(64, action_dim)
        # with torch.no_grad():
        #     # 1. Alpha & Beta: Bias 0.0 -> Sigmoid(0) = 0.5 -> Scaled to ~2.5 (Middle ground)
        #     self.mu_head.bias[0] = 0.0 
        #     self.mu_head.bias[1] = 0.0
            
        #     # 2. Rho: Bias -2.0 -> Sigmoid(-2.0) = 0.12 -> Scaled to ~0.12 (High Memory)
        #     # This forces the agent to start by "remembering" trails.
        #     self.mu_head.bias[2] = -2.0


    def forward(self, x):
        x = torch.relu(self.fc1(x)) #layer 1 
        x = torch.relu(self.fc2(x)) #layer 2 
        x = torch.relu(self.fc3(x)) #layer 3

        mu = self.mu_head(x)
        sigma = torch.nn.functional.softplus(self.sigma_head(x)) + 0.01 #constrain to positive vals for sd
        return mu, sigma

    def get_action(self, state):
        mu,sigma = self.forward(state)
        dist = Normal(mu,sigma)
        action = dist.sample()

        #action = torch.clamp(action, 0.0, 1.0)

        log.debug(f"action type is {type(action)}")
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        decoded_action = {
            'alpha': (torch.sigmoid(action[0]).item() * 4.9) + 0.1, 
            'beta':  (torch.sigmoid(action[1]).item() * 4.9) + 0.1,
            'rho': (torch.sigmoid(action[2]).item() * 0.98) + 0.01
        }
        return decoded_action, log_prob


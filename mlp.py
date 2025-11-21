import torch
import torch.nn as nn
import torch.optim as optim 
from torch.distributions import Normal

class ParameterController(nn.Module):
    def __init__(self,input_dim, action_dim):
        super(ParameterController,self).__init__()# same as super()
        self.fc1 = nn.Linear(input_dim,64)
        self.fc2 = nn.Linear(64, 64)
        self.mu_head = nn.Linear(64, action_vec)

        self.sigma_head= nn.Linear(64, action_vec)


    def foward(self, x):
        x = torch.relu(self.fc1(x)) #layer 1 
        x = torch.relu(self.fc2(x)) #layer 2 

        mu = torch.sigmoid(self.mu_head(x))
        sigma = torch.nn.functional.softplus(self.sigma_head(x)) +0.01 #constrain to positive vals for sd
        return mu, sigma
        
        


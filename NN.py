import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Constant import *

# Need some adjustment to the game features.
class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNEL_NUM, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        
        # Note this conversion from CNN to Linear is so hand wavy, there is probably some way to make this easier that I should look into.
        # there is a + 1 in the end to make the agent aware of what side it is on.
        self.fc1 = nn.Linear((MAP_W - 4) * (MAP_H - 4) * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, MAP_W * MAP_H * NUM_ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) 
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.2)
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.2)
        
        x = self.fc3(x)
        
        x = x.view(MAP_W, MAP_H, NUM_ACTIONS)
        
        return x
class CriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNEL_NUM, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        
        self.fc1 = nn.Linear((MAP_W - 4) * (MAP_H - 4) * 16 + 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x, kills):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1) 
        
        x = torch.cat((x, kills.unsqueeze(0)), dim = 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.2)
        
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.2)
        
        x = self.fc3(x)
        
        return x
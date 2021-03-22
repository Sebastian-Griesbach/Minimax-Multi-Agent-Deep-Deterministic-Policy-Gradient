import torch
from abc import ABC, abstractmethod

class Multiagent_critic(ABC, torch.nn.Module):
    def __init__(self):
        super(Multiagent_critic, self).__init__()

    @abstractmethod
    def forward(self, state, *actions) -> torch.tensor:
        pass

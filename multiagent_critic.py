import torch
from abc import ABC, abstractmethod

class Multiagent_critic(ABC, torch.nn.Module):
    def __init__(self, model):
        super(Multiagent_critic, self).__init__()
        self.model = model

    def forward(self, state, *actions):
        combined_input = self._build_input(state, *actions)
        return self.model(combined_input)

    @abstractmethod
    def _build_input(self, state, *actions) -> torch.tensor:
        pass
import torch
from abc import ABC, abstractmethod

class Multiagent_critic(ABC, torch.nn.Module):
    def __init__(self, model):
        super(Multiagent_critic, self).__init__()
        self.model = model

    def forward(self, **kwargs):
        combined_input = self._build_input(**kwargs)
        return self.model(combined_input)

    @abstractmethod
    def _build_input(self, **kwargs) -> torch.tensor:
        pass
import torch
from abc import ABC, abstractmethod
from typing import List

class Multiagent_critic(ABC, torch.nn.Module):
    def __init__(self):
        """Initialise the Multiagent critic
        """
        super(Multiagent_critic, self).__init__()

    @abstractmethod
    def forward(self, state: torch.tensor, actions: List[torch.tensor]) -> torch.tensor:
        """This method needs to forward the state and the actions through the models such that they have a pytorch traceable gradient.

        Args:
            state (torch.tensor): Total state of the underlying environment
            actions (List[torch.tensor]): List of actions according to the order of agents

        Returns:
            torch.tensor: The critics evalutaion of the Q-value of this state action pair for a specific agent.
        """
        pass

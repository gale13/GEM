import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union
from tianshou.data import Batch, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR

class DiffusionOPT(BasePolicy):

    def __init__(
            self,
            actor: Optional[torch.nn.Module],
            actor_optim: Optional[torch.optim.Optimizer],
            action_dim: int,
            device: torch.device,
            reward_normalization: bool = False,
            lr_decay: bool = False,
            lr_maxt: int = 1000,
            exploration_noise: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Initialize actor network and optimizer if provided
        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor  # Actor network
            self._actor_optim: torch.optim.Optimizer = actor_optim  # Optimizer for the actor network
            self._action_dim = action_dim  # Dimensionality of the action space

        # If learning rate decay is applied, initialize learning rate schedulers for both actor and critic
        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)

        # Initialize other parameters and configurations
        self._rew_norm = reward_normalization  # If true, normalize rewards
        self._lr_decay = lr_decay  # If true, apply learning rate decay
        self._device = device  # Device to run computations on
        self.noise_generator = GaussianNoise(sigma=exploration_noise)


    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            model: str = "actor"
    ) -> Batch:
        # Convert batch observations to PyTorch tensors
        obs_ = to_torch(batch[input], device=self._device, dtype=torch.float32)
        # Use actor or target actor based on provided model argument
        model_ = self._actor
        # Feed observations through the selected model to get action logits
        logits, hidden = model_(obs_), None

        acts = logits
        dist = None  # does not use a probability distribution for actions

        return Batch(logits=logits, act=acts, state=obs_, dist=dist)



    def _update(self, batch: Batch, update: bool = False) -> torch.Tensor:
        # Compute the behavior cloning loss
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        expert_actions = torch.Tensor([info["expert_action"] for info in batch.info]).to(self._device)

        bc_loss = self._actor.loss(expert_actions, obs_).mean()

        if update:  # Update actor parameters if update flag is True
            self._actor_optim.zero_grad()  # Zero the actor optimizer's gradients
            bc_loss.backward()  # Backpropagate the loss
            self._actor_optim.step()  # Perform a step of optimization
        return bc_loss


    def learn(
            self,
            batch: Batch,
            **kwargs: Any
    ) -> Dict[str, List[float]]:
        # Update actor network. Here, we first calculate the policy gradient (pg_loss) and
        # behavior cloning loss (bc_loss) but we do not update the actor network yet.
        # The overall loss is a weighted combination of policy gradient loss and behavior cloning loss.
        overall_loss = self._update(batch, update=False)

        self._actor_optim.zero_grad()
        overall_loss.backward()
        self._actor_optim.step()
        return {
            'overall_loss': overall_loss.item()  # Returns the overall loss as part of the results
        }

class GaussianNoise:
    def __init__(self, mu=0.0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def generate(self, shape):
        """
        :param shape: Shape of the noise to generate, typically the action's shape.
        :return: Numpy array with Gaussian noise.
        """
        noise = np.random.normal(self.mu, self.sigma, shape)
        return noise

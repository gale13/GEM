import torch
import os
import pprint

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import offpolicy_trainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import WandbLogger

from env.distEnergy import DistEnergy
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import LstmRNN
 
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env_e = DistEnergy()

train_envs = DummyVectorEnv([lambda: DistEnergy() for _ in range(10)])
test_envs = DummyVectorEnv([lambda: DistEnergy() for _ in range(10)])


#======================net========================================#

# create actor
actor_net = LstmRNN(
    action_dim=env_e.action_space.shape[0],
    state_dim = env_e.observation_space.shape[0]
)

# Actor is a Diffusion model
actor = Diffusion(
    state_dim=env_e.observation_space.shape[0],
    action_dim=env_e.action_space.shape[0],
    model=actor_net,
    max_action=1,
    beta_schedule='vp',
    n_timesteps=8, #diffusion steps 8
).to(device)
actor_optim = torch.optim.AdamW(
    actor.parameters(),
    lr=1e-5, #1e-4 1e-5
    weight_decay=1e-5 #1e-4
) #optimizer


log_path = 'log'
logger = WandbLogger()
logger.load(SummaryWriter(log_path))

# Define policy
policy = DiffusionOPT(
    actor,
    actor_optim,
    env_e.action_space.shape[0],
    device,
    lr_decay=False,
    lr_maxt=1000,
    action_space=env_e.action_space,
    exploration_noise = 0.1
)

train_collector = Collector(policy, train_envs, VectorReplayBuffer(1e6, len(train_envs)))
test_collector = Collector(policy, test_envs)

def save_best_fn(policy):
    torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

result = offpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    10000, #30 1000
    100,
    1000,
    1,
    512,
    save_best_fn=save_best_fn,
    logger=logger,
    test_in_train=False
)

pprint.pprint(result)

# Generate the topology
Requirement:
> pip install networkx==3.1
  
File: network/gennet.py

Output: network.gml

> python gennet.py

Parameters: N (the number of nodes), p (the probability of connecting any two nodes)

# Parameter Settings
batch_size = 512

Dummy_envs = 10

learning_rate = 1e-5

weight_decay = 1e-5

denoising_steps = 8

network architecture

![Real-world](/image/network.png "network")

# Main Process
> python main.py

Requirements:

Solver: Mosek (https://www.mosek.com/products/academic-licenses/)

Logger: wandb (https://wandb.ai/)

Python:
> pip install cvxpy==1.5.2
>
> pip install wandb==0.17.3
> 
> pip install tianshou==0.4.11
>
> pip install matplotlib==3.7.3
> 
> pip install scipy==1.10.1

# Performance
The gap between GEM and the optimal solution (OPT).

> Real-world Scenario
![Real-world](/image/real.png "Real-world")

> Simulation Scenario
![Simulation](/image/sim.png "Simulation")

# Acknowledgements
This project uses some of the source code from the following repositories:

https://github.com/hojonathanho/diffusion

https://github.com/HongyangDu/GDMOPT


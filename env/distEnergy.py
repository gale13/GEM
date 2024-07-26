import gym
from gym import spaces
import numpy as np
import networkx as nx
import cvxpy as cp

def convexprogram(Demand,Generate,Storage,Graph,unitcost,a,b):
    nodes = list(Graph.nodes)
    neighbors=[list(Graph.neighbors(_)) for _ in nodes]
    L = sum([len(sub_list) for sub_list in neighbors])+len(nodes)

    x = cp.Variable(L)

    Supplyto = np.zeros((len(nodes),L)) #from j to i
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            for l in range(len(neighbors[j])):
                if (nodes[i]==neighbors[j][l]):
                    Supplyto[i][sum([len(sub_list) for sub_list in neighbors[:j]])+l]=1
    for k in range(len(nodes)):
        Supplyto[k][-len(nodes)+k]=1


    Supplyfrom = np.zeros((len(nodes),L)) #from i to j
    Supplyfromno = np.zeros((len(nodes),L)) #from i to j no i to i
    for i in range(len(neighbors)):
        for j in range(len(neighbors[i])):
            Supplyfrom[i][sum([len(sub_list) for sub_list in neighbors[:i]])+j]=1
            Supplyfromno[i][sum([len(sub_list) for sub_list in neighbors[:i]])+j]=1
    for k in range(len(nodes)):
        Supplyfrom[k][-len(nodes)+k]=1

    demand = np.array(Demand)
    generate = np.array(Generate)
    storage = np.array(Storage)

    objective = cp.Minimize(a*cp.sum(cp.maximum(demand-Supplyto @ x,0))+b*Supplyfromno@x@unitcost) #unitcost@x
    constraints = [Supplyfrom@x <= generate+storage,x>=0]
    prob = cp.Problem(objective, constraints)

    res = prob.solve()

    optaction = x.value
    #print(optaction)
    weightedaction = [0]*len(optaction)
    for i in range(len(nodes)):
        temp = sum(optaction[sum([len(sub_list) for sub_list in neighbors[:i]]):sum([len(sub_list) for sub_list in neighbors[:i]])+len(neighbors[i])])+optaction[-len(nodes)+i]
        if (temp>0):
            for j in range(len(neighbors[i])):
                weightedaction[sum([len(sub_list) for sub_list in neighbors[:i]])+j]=optaction[sum([len(sub_list) for sub_list in neighbors[:i]])+j]/temp
            weightedaction[-len(nodes)+i]=optaction[-len(nodes)+i]/temp
        else:
            for j in range(len(neighbors[i])):
                weightedaction[sum([len(sub_list) for sub_list in neighbors[:i]])+j]=optaction[sum([len(sub_list) for sub_list in neighbors[:i]])+j]
            weightedaction[-len(nodes)+i]=optaction[-len(nodes)+i]

    return res,weightedaction

class DistEnergy(gym.Env):

    def __init__(self):

        self.Graph=nx.read_gml("./network/network.gml")
        #self.Graph = nx.fast_gnp_random_graph(self.nodes_num, self.p) #topology of the distributed energy network
        self.nodes = list(self.Graph.nodes)
        self.nodes_num = len(self.nodes)
        self.neighbors = [list(self.Graph.neighbors(_)) for _ in self.nodes]
        #self.L = sum([len(_) for _ in self.neighbors])+len(self.nodes)
        self.L = len(self.nodes)
        self.a = 1
        self.b = 0.1

        self.maxpower = np.array([100]*(self.nodes_num*3+self.L)) #max generated energy at each node
        self.lowpower = np.array([0]*(self.nodes_num*3+self.L))

        self.maxsupply = np.array([1]*(sum([len(sub_list) for sub_list in self.neighbors])+self.nodes_num))
        self.lowsupply = np.array([-1]*(sum([len(sub_list) for sub_list in self.neighbors])+self.nodes_num))

        self.observation_space = spaces.Box(shape=(3*self.nodes_num+self.L,),low=self.lowpower,high=self.maxpower,dtype=np.float32) # generated energy (N) + energy demand (N)+storage(N)
        self.action_space = spaces.Box(shape=(sum([len(sub_list) for sub_list in self.neighbors])+self.nodes_num,),dtype=np.float32,low=self.lowsupply,high=self.maxsupply)

        self.terminated = False
        self._state = None
        self._steps_per_episode = 24
        self._num_steps = 0
 
        self.seed()

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        # Reset the environment to its initial state
        self.terminated = False
        state = self.state()
        self._num_steps = 0
        return state, {}


    def state(self):
        # Provide the current state to the agent
        states = np.random.uniform(0, self.maxpower[0], 2*self.nodes_num)
        zeros = np.zeros((self.nodes_num,)) #for storage
        unitcost = np.random.uniform(0,1,self.L)
        #unitcost[-len(self.nodes):]=[0]*len(self.nodes)
        res = np.concatenate([states, zeros,unitcost])
        self._state = res
        return res
    

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action)
        )
        assert not self.terminated, "One episodic has terminated"
 
        Generate = self._state[:self.nodes_num]
        Demand = self._state[self.nodes_num:2*self.nodes_num]
        Storage = self._state[2*self.nodes_num:-self.L]
        Unitcost = self._state[-self.L:]

        supply = [0]*self.nodes_num
        weightedSupply = dict(zip(self.nodes,supply)) # total energy from j to i

        action = abs(action)
        cost = 0

        op,expert_action = convexprogram(Demand,Generate,Storage,self.Graph,Unitcost,self.a,self.b)

        for i in range(self.nodes_num):
            # total weight
            temp = sum(action[sum([len(sub_list) for sub_list in self.neighbors[:i]]):sum([len(sub_list) for sub_list in self.neighbors[:i]])+len(self.neighbors[i])])+action[-self.nodes_num+i]
            for j in range(len(self.neighbors[i])):
                # energy from i to j
                weightedSupply[self.neighbors[i][j]]+=action[sum([len(sub_list) for sub_list in self.neighbors[:i]])+j]/temp*(Generate[i]+Storage[i])
                cost+=action[sum([len(sub_list) for sub_list in self.neighbors[:i]])+j]/temp*(Generate[i]+Storage[i])*Unitcost[i]
            weightedSupply[self.nodes[i]]+=action[-self.nodes_num+i]/temp*(Generate[i]+Storage[i]) #from i to i
            #cost+=action[-self.nodes_num+i]/temp*(Generate[i]+Storage[i])*Unitcost[i]

        #op = 0
        reward = 0.0
        overflow = []
        for i in range(self.nodes_num):
            reward+=np.max([Demand[i] - weightedSupply[self.nodes[i]],0])
            if (weightedSupply[self.nodes[i]] >Demand[i]-0.001):
                overflow.append(np.min([weightedSupply[self.nodes[i]]-Demand[i],self.maxpower[0]]))
            else:
                overflow.append(0)

        reward = op - self.a*reward - self.b*cost

        self._num_steps += 1
        if self._num_steps >= self._steps_per_episode:
            self.terminated = True
 
        state = self.state()
        self._state[2*self.nodes_num:-self.L]=overflow
        state[2*self.nodes_num:-self.L]=overflow
        info = {'reward':reward,'opt':op,'expert_action':expert_action}
        return state, reward, self.terminated, info





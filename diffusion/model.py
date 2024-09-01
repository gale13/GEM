import torch
import torch.nn as nn
from .helpers import SinusoidalPosEmb
    

class LstmRNN(nn.Module):
    """
        Parameters：
        - hidden_size: number of hidden units
        - num_layers: layers of LSTM to stack
    """
 
    def __init__(self, action_dim,state_dim, hidden_size=256, num_layers=3):
        super().__init__()
        hidden_dim=256
        t_dim=16
        self.state_dim = state_dim
        self.action_dim=action_dim
        self.input_size=hidden_dim + action_dim + t_dim
        self.num_layers=num_layers
        self.hidden_size=hidden_size
 
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, action_dim) # fc

        _act = nn.ReLU
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
 
    def forward(self, x, time, state):
        processed_state = self.state_mlp(state)
        t = self.time_mlp(time)
        x = torch.cat([x, t, processed_state], dim=1)
        batch_size, seq_len = x.shape[0], x.shape[1]


        x = x.reshape((-1,batch_size,self.input_size))
        x, _ = self.lstm(x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = self.linear1(x)
        return x[-1]



  

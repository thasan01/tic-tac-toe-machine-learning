import torch
from torch import nn
import torch.nn.functional as F


# Tic-Tac-Toe Deep Q Network Model
class T3DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

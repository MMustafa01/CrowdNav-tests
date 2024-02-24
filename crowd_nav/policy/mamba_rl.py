import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL

''''
So from my understanding the MambaRL class wont be the class where I define the model, 
Instead I should make a valuenetwork class to define the value network and just store it in self.model
'''

class ValueNetork(nn.module):
    def __init__(self, input_dim, mlp_dims, dim, d_state, d_conv, expand):
        super().__init__()
        self.value_network = mlp(input_dim, mlp_dims)
        self.mambaLayer = MambaRL(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=dim, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )

class MambaRL(MultiHumanRL):
    def __init__(self):
        self.name = 'MambaRL'
        return

    def configure(self):
        '''
        Eventually set everything in this one'
        Need to read MLP/Value network dimensions 

        '''
        
        
        super().__init__()
        
        
        self.mlp = mlp(
            '''Gotta figure out '''
        )
        return
    def forward(self):
        pass
    '''
    Check if 
    '''
    def predict(self, state):
        return super().predict(state)


import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL
from mamba_ssm import Mamba

''''
So from my understanding the MambaRL class wont be the class where I define the model, 
Instead I should make a valuenetwork class to define the value network and just store it in self.model
'''

class ValueNetork(nn.Module):
    def __init__(self, input_dim, mlp_dims, d_state, d_conv, expand):
        super().__init__()
        '''
        Maybe use a sequential layer here
        '''
        print(f'input dimension = {input_dim}')
        self.mambaLayer = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=input_dim, # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        self.value_network = mlp(input_dim, mlp_dims)

    def forward(self,state):
        # print('State')
        # print(state.shape)
        x = self.mambaLayer(state)
        # print('Mamba Layer')
        
        x = x.squeeze()
        # print(x.shape)
        # joint_state = torch.cat([state, x], dim=1)
        # print(joint_state.shape)
        value = self.value_network(x)
        # print('Value network')
        # print(value.shape)

        # print(value.shape)

        return value

class MambaRL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'MambaRL'

    def configure(self,config):
        d_state = 16
        d_conv=4
        expand=2
        '''
        Eventually set everything in this one'
        Need to read MLP/Value network dimensions 

        Need to understand the parameters of 
        mamba, i.e. d_state, d_conv,expand

        '''
        self.set_common_parameters(config) #check what this does
        mlp_dims = [int(x) for x in config.get('MambaRL', 'mlp_dims').split(', ')]
        self.multiagent_training = config.getboolean('MambaRL', 'multiagent_training')
        
        self.model = ValueNetork(self.input_dim(), #joint state dimensions
                                 mlp_dims=mlp_dims,
                                 d_state=d_state,
                                 d_conv=d_conv,
                                 expand=expand)
        print(self.model.eval())
        logging.info(f'Policy:{self.name}')
        return
    '''
    Check if something different would need to be done to change the MultiHumanRL functions. For example, in LstmRL they implement predict because they want to set the state in a particular order
    '''
    def predict(self, state):
        return super().predict(state)


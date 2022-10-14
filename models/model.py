import torch
from layers import * ## import everything from layers, so its compatible with eval 


class CNN_model(nn.Module):
    def __init__(self, mconfig):
        super(CNN_model, self).__init__()
        conv_block1_fn = eval(mconfig['model']['conv_block1_fn']) # get the name of the layer you want to use, then load it with eval 
        self.conv_block1 = conv_block1_fn(**mconfig['model']['conv_block1_kwargs']) ## everything in this section of the config needs to match the args in the layer, no more, no less 
        
        conv_block2_fn = eval(mconfig['model']['conv_block2_fn'])
        self.conv_block2 = conv_block2_fn(**mconfig['model']['conv_block2_kwargs'])
        
        mlp_block_fn = eval(mconfig['model']['mlp_block_fn'])
        self.mlp = mlp_block_fn(**mconfig['model']['mlp_block_kwargs'])
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x 
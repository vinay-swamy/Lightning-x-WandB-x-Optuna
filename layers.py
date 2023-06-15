from turtle import forward
import torch 
from torch import nn 


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size, stride, padding,maxpool_kernel_size ):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(         
            nn.Conv2d(
                in_channels=in_channels,              
                out_channels=out_channels,            
                kernel_size=conv_kernel_size,              
                stride=stride,                   
                padding=padding,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=maxpool_kernel_size),    
        )
    def forward(self, x):
        x = self.net(x)
        return x 

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MultiLayerPerceptron, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,out_dim)
        )
        self.sm = nn.Softmax()
    def forward(self, x):
        logits = self.MLP(x)
        probs=  self.sm(logits)
        return logits, probs 
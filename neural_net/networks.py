import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net3(torch.nn.Module):
    """
    A class which defines model architecture with number of hidden layers equal
    to ´depth´.
    Xavier uniform initialization of weights.
    Initialization of bias to ones. ReLu activation function.
    """
    def __init__(self, n_feature, n_hidden, n_output, depth, init = 0.1):
        super(Net3, self).__init__()
        
        self.input = torch.nn.Linear(n_feature, n_hidden).float().to(device)
        nn.init.xavier_uniform_(self.input.weight,gain=1)
        nn.init.ones_(self.input.bias)

        self.layers = nn.ModuleDict()
        self.bn = nn.BatchNorm1d(n_hidden).to(device)
        self.dropout = torch.nn.Dropout(0.05) 

        for i in range(1, depth):
            self.layers['hidden_'+str(i)] = torch.nn.Linear(n_hidden,n_hidden).float().to(device)
            self.layer_initialisation(self.layers['hidden_'+str(i)], weight_init = 'xavier', bias_init = 'ones')
    
        self.predict = torch.nn.Linear(n_hidden, n_output).float().to(device)
        nn.init.xavier_uniform_(self.predict.weight,gain=1)
        nn.init.ones_(self.predict.bias)

    def forward(self, x):
        x = F.relu(self.input(x))     
        x = self.bn(x)
        for layer in self.layers:
            x = F.relu(self.layers[layer](x))
            x = self.bn(x)
            x = self.dropout(x)
        x = self.predict(x)           
        return x

    def layer_initialisation(self, incoming_layer, weight_init = 'xavier', bias_init = 'ones'):
        if weight_init == 'xavier':
            nn.init.xavier_uniform_(incoming_layer.weight,gain=1)
            nn.init.ones_(incoming_layer.bias) 
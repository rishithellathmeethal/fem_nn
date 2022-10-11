import torch
from loss_functions import *
'''
This file contain different training methods for neural network. eg: train pinn alone, train pinn and fem together etc.
'''
device = "cuda" if torch.cuda.is_available() else "cpu" 

class train_model_base():
    def __init__(self, model):
        self.model =  model.to(device)
        self.train_data = None
    
    def _calculate_loss(self, u):
        pass 
    
    def train(self, data, epochs=1000, opt=None):
        self.train_data = data.to(device)
        self.model.train()
        if opt is None:
            opt = torch.optim.Adam(self.model.parameters(), lr = 0.0003)
        self.loss_h = []
        for epoch in range(epochs):
            u = self.model(self.train_data)
            loss_epoch = self._calculate_loss(u)
            opt.zero_grad()
            loss_epoch.backward()
            opt.step()
            if epoch%1000==0:
                print(epoch, loss_epoch.detach().cpu().numpy())
            self.loss_h.append(loss_epoch.detach().cpu().numpy())
        return self.loss_h
        
    def predict(self, p_in_data):
        self.model.eval() 
        return self.model(p_in_data)
    
    def plot_training(self):
        plt.plot(self.loss_h)

class train_pinn_1d(train_model_base):
    def __init__(self, model, f, l_bc, r_bc):
        train_model_base.__init__(self, model)
        self.force    = f 
        self.left_bc  = l_bc
        self.right_bc = r_bc
    def _calculate_loss(self, u):
        l = loss_f(u, self.train_data, self.force) + loss_mse(-1.0, self.left_bc, 1.0, self.right_bc, self.model)
        return l

class train_fem_loss_1d(train_model_base):
    def __init__(self, model, A, F):
        train_model_base.__init__(self, model)
        self.A =  A
        self.F =  F
    def _calculate_loss(self, u):
        l = loss_fem(self.A, self.F, u)
        return l
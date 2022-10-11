import torch
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu" 

def get_der(u, t, order=2):
    '''
    calculating derivation using AD 
    '''
    ones = torch.ones_like(u)
    der, = torch.autograd.grad(u, t, create_graph=True, grad_outputs=ones, allow_unused=True)
    if der is None:
        print("Returning zero gradients")
        return torch.zeros_like(t, requires_grad=True)
    else:
        der.requires_grad_()
    for i in range(1, order):
        ones = torch.ones_like(der)
        der, = torch.autograd.grad(der, t, create_graph=True, grad_outputs=ones, allow_unused=True)
        if der is None:
            return torch.zeros_like(t, requires_grad=True)
        else:
            der.requires_grad_()
    
    return der 

def loss_f(u, t, f, order=2):
    ''' 
    u, t, f are in (number_of_samples,1) shape
    '''
    der = get_der(u, t, order=2)
    
    f = f.repeat(t.shape[0],1)
    return torch.abs(torch.mean(der + f))


def loss_mse(left, lv, right, rv, model):
    return torch.sqrt((torch.tensor([lv]).to(device)-model(torch.tensor([left]).double().to(device)))**2 + (torch.tensor([rv]).to(device)-model(torch.tensor([right]).double().to(device)))**2)

def loss_fem(A, F, u):
    
    # print(A.shape, F.shape)
    l =  torch.sqrt(torch.mean((torch.matmul(torch.tensor(A).to(device), u)-torch.tensor(F).to(device))**2))
    return l
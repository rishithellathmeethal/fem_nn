from torch.autograd import Function
import torch.nn as nn
import torch
import numpy as np
import sys
sys.path.append('../')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Inherited from torch autograd Function to write custom backward function 
class Linear_residual_loss(Function):
    @staticmethod
    def forward(ctx, output, k_all, f_all): 
        """
        Takes neural network output, stiffness matrices and force vectors in minibatch as input
        """
        #k, f in the shape (number of sample * matrix/vector)
        output = output.reshape(output.shape[0], output.shape[1], 1) 
        loss = torch.zeros(1).requires_grad_() 
        # K = torch.from_numpy(np.array(k_all)).float().to(device)
        # F = torch.from_numpy(np.array(f_all)).float().to(device)
        K = k_all
        F = f_all
        R = K.matmul(output) - F
        loss = torch.norm(R) /output.shape[0]
        final_loss = loss.requires_grad_()
        ctx.save_for_backward(K, R, final_loss)
        
        return final_loss

    @staticmethod
    def backward(ctx, grad_output):
        K,R,final_loss = ctx.saved_tensors
        grad_input = grad_output = None 
        R_t = R.permute(0,2,1)
        grad_output_c = R_t.matmul(K)/(final_loss)
        grad_out = grad_output_c.permute(0,2,1)
        grad_out2 = grad_out.reshape(grad_out.shape[0],grad_out.shape[1]) / grad_out.shape[0]

        return grad_out2 , None, None

# Inherited from torch autograd Function to write custom backward function 
class Linear_residual_loss(Function):
    @staticmethod
    def forward(ctx, net_out, K, F): 
        """
        Takes neural network output, stiffness matrices and force vectors in minibatch as input
        """
        #k, f in the shape (number of sample * matrix/vector)
        net_out = net_out.reshape(net_out.shape[0], net_out.shape[1], 1) 
        loss = torch.zeros(1).requires_grad_() 
        R = K.matmul(net_out) - F
        loss = torch.norm(R) /net_out.shape[0]
        final_loss = loss.requires_grad_()
        ctx.save_for_backward(K, R, final_loss)
        
        return final_loss

    @staticmethod
    def backward(ctx, grad_output):

        K,R,final_loss = ctx.saved_tensors
        grad_net_out = grad_k = grad_f = None 

        R_t = R.permute(0,2,1)
        grad_output_c = R_t.matmul(K)/(final_loss)
        grad_out = grad_output_c.permute(0,2,1)

        grad_out2 = grad_out.reshape(grad_out.shape[0],grad_out.shape[1]) / grad_out.shape[0]

        print(grad_out2.shape, grad_output.shape)
        grad_net_out = grad_net_out.mm(grad_out2)

        return grad_out2 , None, None 
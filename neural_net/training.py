import torch
import numpy as np
from matplotlib import pyplot as plt
import time as time
from decimal import Decimal
torch.manual_seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

import sys
sys.path.append("../")
from neural_net.loss_functions import Linear_residual_loss

def train_with_loader(net, dataloader_train, dataloader_test, l_rate , epoch = 20000, loss_f = None, plot = False):
    try:
        begin = time.time()
        print("training begins")
        optimizer = torch.optim.Adam(net.parameters(), lr= l_rate)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 300, gamma = 0.95)
        linear_res    = Linear_residual_loss.apply
        
        epoch_plot = []
        train_plot = []
        test_plot  = []
        sum_loss_train = 0.0
        sum_loss_test  = 0.0

        for t in range(epoch):
            sum_loss_train = 0.0
            for d, k, f in dataloader_train:
                prediction = net(d)     
                loss = linear_res(prediction, k, f) 
                optimizer.zero_grad()   
                loss.backward()         
                optimizer.step()        
                sum_loss_train += loss.item()
            if(t%10 ==0):
                print(f"Epoch {t}: loss = {Decimal(sum_loss_train/len(dataloader_train)):.5E}", end='')
            if t >5:
                epoch_plot.append(t)
                train_plot.append(sum_loss_train/len(dataloader_train))
            
            sum_loss_test  = 0.0
            for d, k, f in dataloader_test:
                prediction = net(d)     
                loss = linear_res(prediction, k, f) 
                sum_loss_test += loss.item()
            if(t%10 ==0):
                print(f"\ttest_loss = {Decimal(sum_loss_test/len(dataloader_test)):.5E}")
            if t >5:
                test_plot.append(sum_loss_test/len(dataloader_test))
            scheduler.step() 

    except KeyboardInterrupt:
        pass
        
    end = time.time()

    print("training ends")
    print("Total time for training is",end-begin)
    # if plot == True:
    #     plt.figure()
    #     plt.plot(epoch_plot, train_plot)
    #     plt.plot(epoch_plot, test_plot)
    #     timestr = time.strftime("%Y%m%d-%H%M%S")
    #     plt.savefig('output/'+timestr+".png")
    #     plt.show()

    return sum_loss_train/len(dataloader_train), sum_loss_test/len(dataloader_test) 
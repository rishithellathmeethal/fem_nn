import numpy as np
from matplotlib import pyplot as plt
import json, torch, sys, optuna
from tqdm import tqdm
import time as time
import torch as T
from torch.utils.data import DataLoader
from sklearn import preprocessing
from pickle import dump

## following are for connecting to FEM code
sys.path.append('../../')
import KratosMultiphysics
from fem_interfaces.kratos.Kratos_Struct_Linear_Sudret_Truss import *

## neural network realated functions 
from neural_net.training import train_with_loader
from neural_net.data_utilities import loader_creation
from neural_net.networks import Net3

# For repeatability
np.random.seed(42)
torch.manual_seed(0)

# For working in both gpu and cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
# making training data of input space
n_samples = 15000

try:
    k_all = np.load(f'data/k_all_{n_samples}'+'.npy')
    f_all = np.load(f'data/f_all_{n_samples}'+'.npy')
    data_in = np.load(f'data/data_in_{n_samples}'+'.npy')
    loaded = True
except: 
    loaded = False

if not loaded:
    E_low = 2.1e10
    A_Ver_low = 1e-4
    A_Hor_low = 1e-4 
    F_low = -1e4

    E_high = 2.1e12
    A_Ver_high = 1e-3
    A_Hor_high = 1e-3
    F_high = 7.5e3

    E_Hor_rv = np.random.uniform(E_low,E_high, size = n_samples).reshape(n_samples,1)
    E_Ver_rv = np.random.uniform(E_low,E_high, size = n_samples).reshape(n_samples,1)
    A_Hor_rv = np.random.uniform(A_Hor_low,A_Hor_high, size = n_samples).reshape(n_samples,1)
    A_Ver_rv = np.random.uniform(A_Ver_low,A_Ver_high, size = n_samples).reshape(n_samples,1)
    F1_rv = np.random.uniform(F_low,F_high, size = n_samples).reshape(n_samples,1)
    F2_rv = np.random.uniform(F_low,F_high, size = n_samples).reshape(n_samples,1)
    F3_rv = np.random.uniform(F_low,F_high, size = n_samples).reshape(n_samples,1)
    F4_rv = np.random.uniform(F_low,F_high, size = n_samples).reshape(n_samples,1)
    F5_rv = np.random.uniform(F_low,F_high, size = n_samples).reshape(n_samples,1)
    F6_rv = np.random.uniform(F_low,F_high, size = n_samples).reshape(n_samples,1)

    data_in_str = np.concatenate((E_Ver_rv, A_Ver_rv, E_Hor_rv, A_Hor_rv), axis = 1)
    data_in_load = np.concatenate((F1_rv,F2_rv,F3_rv,F4_rv,F5_rv,F6_rv), axis = 1)
    data_in = np.concatenate((data_in_str, data_in_load), axis=1)
    k_all = []
    f_all = []

    ############ Data creation
    print("Started making data")
    with open("sim_parameters/ProjectParameters.json",'r') as parameter_file:
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    start_time = time.time()
    qoi_all = []

    ### call Kratos and create matrices  here
    for data_str, data_load in zip(data_in_str, data_in_load):  
        model = KratosMultiphysics.Model()
        simulation = StructMechAnaWithVaryingParameters(model,parameters,data_str, data_load)
        k, f  = simulation.Run()
        k_all.append(k)
        f_all.append(f)
    
    end_time = time.time()
    print("time taken to make data is", end_time-start_time)

    k_all = np.asarray(k_all)
    f_all = np.asarray(f_all)
    data_in = np.asarray(data_in)
        
    np.save(f'data/k_all_{n_samples}', k_all)
    np.save(f'data/f_all_{n_samples}', f_all)
    np.save(f'data/data_in_{n_samples}', data_in)

scaler = preprocessing.MinMaxScaler()
data_in_s = scaler.fit_transform(data_in)
dump(scaler, open('scaler.pkl', 'wb'))

## data_in_s is the scaled input data to the neural network 
data_in_torch = torch.from_numpy(data_in_s).float().to(device)

## loader to load data in batches , batch size can be given in argument 
lc = loader_creation(data_in_torch, k_all, f_all, n_samples)
train_loader, test_loader = lc.get_loaders(2, True)
feature_size = data_in.shape[1]

def objective(trial):
    # Suggest values of the hyperparameters using a trial object.
    n_layers = trial.suggest_int('n_layers', 1, 20)
    n_units  = trial.suggest_int('n_units', 5, 500)
    init     =  trial.suggest_float("init", 10, 1000.0)
    lr       =  trial.suggest_float("lr", 0.0001, 100.0)

    # Creation of network
    net = Net3(n_feature = feature_size, n_hidden= n_units, n_output = 39, depth = n_layers, init = init )
    train_loss, test_loss = train_with_loader(net, train_loader, test_loader, lr, 3, plot = False)

    return train_loss

#Create a study object and optimize the objective function to find the best hyperparameters.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print(study.best_params["n_units"], study.best_params["n_layers"], study.best_params["init"], study.best_params["lr"])

net_final = Net3(n_feature = feature_size, n_hidden= study.best_params["n_units"], n_output = 39, 
                depth = study.best_params["n_layers"], init = study.best_params["init"] )

train_loss, test_loss = train_with_loader(net_final, train_loader, test_loader, study.best_params["lr"], 25000, plot = True)

# saving the model
modelname = 'model'+'.pt'
torch.save(net_final, modelname)
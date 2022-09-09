import numpy as np 
from matplotlib import pyplot as plt 
from scipy import stats 
from scipy import linalg 
from scipy.stats import gaussian_kde, tmean, tstd, skew, kurtosis, mode
import numpy as np
import json, sys, torch, pickle
from tqdm import tqdm 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## following are for connecting to FEM code
import KratosMultiphysics
sys.path.append('../../')
from fem_interfaces.kratos.Kratos_Struct_Linear_Sudret_Truss import *

from utilities.plot_utilities import plot_data_general 
from neural_net.networks import Net3
np.random.seed(25)

# material properties
E_Hor = 2.1e11
E_Ver = 2.32e11
A_Hor = 9.2e-4
A_Ver = 1.89e-3

# forces
P1 = -5.2e4
P2 = -5.2e4
P3 = -5.4e4
P4 = -3.6e4
P5 = -6.5e4
P6 = -4.4e4

print("Predicting for the inputs", E_Hor, E_Ver, A_Hor, A_Ver, P1, P2, P3, P4, P5, P6)

data_in_str  = np.array([[E_Ver, A_Ver, E_Hor, A_Hor]])
data_in_load = np.array([[P1, P2, P3, P4, P5, P6]])
data_in      = np.concatenate((data_in_str, data_in_load), axis=1)

scalar = pickle.load(open('scaler.pkl', 'rb'))

data_in_scaled = scalar.transform(data_in)
train_in = torch.from_numpy(data_in_scaled).float().to(device).requires_grad_(True)

### NN surrogate model
feature_size = data_in.shape[1] 

# creation of network
model = torch.load("model.pt")
# loading the parameters fram the saved model
model.eval()

pred = model(train_in)

with open("sim_parameters/ProjectParameters.json",'r') as parameter_file:
    parameters = KratosMultiphysics.Parameters(parameter_file.read())

for data_str, data_load in zip(data_in_str, data_in_load):  # replace it with all varible you wanto to change
    model_kratos = KratosMultiphysics.Model()
    simulation = StructMechAnaWithVaryingParameters_qoi(model_kratos, parameters, data_str, data_load)
    simulation.Run()

    x_act = simulation.qoi_x
    y_act = simulation.qoi_y
    z_act = simulation.qoi_z

pred_reshaped =  pred.reshape((13,3))
x_axis =  np.array([0, 2, 4, 6, 8, 10,  12, 14,  16, 18, 20, 22, 24])
print(x_axis.shape)
plot_data_general(x_axis, np.asarray(y_act), pred_reshaped.detach().numpy()[:,1],"sudret",  ["FEM", "FEM-NN"],["x (m)", "displacement (mm)"])
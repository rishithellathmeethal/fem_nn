from __future__ import print_function, absolute_import, division #makes KratosMultiphysics backward compatible with python 2.6 and 2.7
from KratosMultiphysics import *
import KratosMultiphysics
import KratosMultiphysics as km
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_static_solver import StaticMechanicalSolver
import numpy as np
from scipy import io
import KratosMultiphysics.scipy_conversion_tools
import os

"""
For user-scripting it is intended that a new class is derived
from StructuralMechanicsAnalysis to do modifications

The following script is meant to extract matrices from Kratoos simulation after the application of boundary conditions, but before going to solver. 
Hence we never the solver. 

"""

class StaticMechanicalSolverWithSystemMatrixAccess(StaticMechanicalSolver):
    def SolveSolutionStep(self):
        self.space_utils = UblasSparseSpace()
        a = self._GetBuilderAndSolver()
        b = self._GetSolutionStrategy()
        c = self._GetScheme()
        k = self._GetSolutionStrategy().GetSystemMatrix()
        f = self._GetSolutionStrategy().GetSystemVector()
        t = f

        self.space_utils.SetToZeroMatrix(k)
        self.space_utils.SetToZeroVector(f)
        self.space_utils.SetToZeroVector(t)

        self._GetBuilderAndSolver().Build(c,self.main_model_part,k,f)
        self._GetBuilderAndSolver().ApplyDirichletConditions(c,self.main_model_part,k,t,f)

        k = self._GetSolutionStrategy().GetSystemMatrix()
        f2 = self._GetSolutionStrategy().GetSystemVector()

        ###code to save and retrieve matric and vector
        km.WriteMatrixMarketMatrix("s_mat",k,False)
        km.WriteMatrixMarketVector("s_vec",f2)

        # work around to fix the comma issue 
        k_dummy_file = open("s_mat", "r+")
        k_dummy = k_dummy_file.read()
        k_dummy = k_dummy.replace(',','.')
        k_write = open("s_mat_fin", "w+")
        k_write.write(k_dummy)
        k_write.close()

        f_dummy_file = open("s_vec", "r+")
        f_dummy = f_dummy_file.read()
        f_dummy = f_dummy.replace(',','.')
        f_write = open("s_vec_fin", "w+")
        f_write.write(f_dummy)
        f_write.close()

        k_matrix = io.mmread("s_mat_fin").A
        f_vector = io.mmread("s_vec_fin")

        os.remove("s_mat")
        os.remove("s_vec")
        os.remove("s_mat_fin")
        os.remove("s_vec_fin")

        print("File has been deleted")

        return k_matrix, f_vector


class StructMechAnaWithVaryingParameters(StructuralMechanicsAnalysis):
    def __init__(self,model,project_parameters,mat_par,load_par):
        super(StructMechAnaWithVaryingParameters,self).__init__(model,project_parameters)
        self.mat_par = mat_par
        self.load_par = load_par
        print("Calling derived class to run many simulations")
        self.model = model
    
    def ChangeMaterialProperties(self):  # changing parameters here changes them in the simulation
        self.model["Structure"].GetSubModelPart("Parts_V").GetProperties()[1].SetValue(KratosMultiphysics.YOUNG_MODULUS,self.mat_par[0])
        self.model["Structure"].GetSubModelPart("Parts_V").GetProperties()[1].SetValue(StructuralMechanicsApplication.CROSS_AREA,self.mat_par[1])
        self.model["Structure"].GetSubModelPart("Parts_H").GetProperties()[2].SetValue(KratosMultiphysics.YOUNG_MODULUS,self.mat_par[2])
        self.model["Structure"].GetSubModelPart("Parts_H").GetProperties()[2].SetValue(StructuralMechanicsApplication.CROSS_AREA,self.mat_par[3]) 
        # loads 
        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(1).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[0],0])
        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(2).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[1],0])
        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(3).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[2],0])
        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(4).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[3],0])
        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(5).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[4],0])
        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(6).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[5],0])

    def _CreateSolver(self):
        """ Create the Solver (and create and import the ModelPart if it is not alread in the model) """
        ## Solver construction
        solver_settings = self.project_parameters["solver_settings"]
        return StaticMechanicalSolverWithSystemMatrixAccess(self.model, solver_settings)
    
    def Run(self):
        self.Initialize()
        k,f = self.RunSolutionLoop()
        print("in derived class")
        return k, f

    def RunSolutionLoop(self):
        """
        This function executes the solution loop of the AnalysisStage
        It can be overridden by derived classes
        """
        while self.KeepAdvancingSolutionLoop():
            self.time = self._GetSolver().AdvanceInTime(self.time)
            self.InitializeSolutionStep()
            self._GetSolver().Predict()
            k,f = self._GetSolver().SolveSolutionStep()
        print("in derived class of RunSolutionLoop")
        return k, f 


class StructMechAnaWithVaryingParameters_qoi(StructuralMechanicsAnalysis):
    def __init__(self,model,project_parameters,mat_par,load_par):
        super(StructMechAnaWithVaryingParameters_qoi,self).__init__(model,project_parameters)
        self.mat_par = mat_par
        self.load_par = load_par
        print("Calling derived class to run many simulations")
        self.model = model
    
    def ChangeMaterialProperties(self):  # changing parameters here changes them in the simulation
        self.model["Structure"].GetSubModelPart("Parts_V").GetProperties()[1].SetValue(KratosMultiphysics.YOUNG_MODULUS,self.mat_par[0])
        self.model["Structure"].GetSubModelPart("Parts_V").GetProperties()[1].SetValue(StructuralMechanicsApplication.CROSS_AREA,self.mat_par[1])
        self.model["Structure"].GetSubModelPart("Parts_H").GetProperties()[2].SetValue(KratosMultiphysics.YOUNG_MODULUS,self.mat_par[2])
        self.model["Structure"].GetSubModelPart("Parts_H").GetProperties()[2].SetValue(StructuralMechanicsApplication.CROSS_AREA,self.mat_par[3]) 

        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(1).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[0],0])
        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(2).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[1],0])
        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(3).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[2],0])
        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(4).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[3],0])
        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(5).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[4],0])
        self.model["Structure"].GetSubModelPart("PointLoad2D_Load_on_points_Auto2").GetCondition(6).SetValue(StructuralMechanicsApplication.POINT_LOAD, [0,self.load_par[5],0])

    def FinalizeSolutionStep(self):
        super(StructMechAnaWithVaryingParameters_qoi,self).FinalizeSolutionStep()
        self.qoi_x = []
        self.qoi_y = []
        self.qoi_z = []
        for node in self._GetSolver().main_model_part.Nodes:
            self.qoi_x.append(node.GetSolutionStepValue(DISPLACEMENT_X))
            self.qoi_y.append(node.GetSolutionStepValue(DISPLACEMENT_Y))
            self.qoi_z.append(node.GetSolutionStepValue(DISPLACEMENT_Z))


class StructMechAnaWithVaryingParameters_qoi_check(StructuralMechanicsAnalysis):
    def __init__(self,model,project_parameters,mat_par,load_par):
        super(StructMechAnaWithVaryingParameters_qoi_check,self).__init__(model,project_parameters)
        self.mat_par = mat_par
        self.load_par = load_par
        print("Calling derived class to run many simulations")
        self.model = model
    
    def FinalizeSolutionStep(self):
        super(StructMechAnaWithVaryingParameters_qoi_check,self).FinalizeSolutionStep()
        self.qoi = self._GetSolver().main_model_part.GetNode(7).GetSolutionStepValue(DISPLACEMENT_Y)
        
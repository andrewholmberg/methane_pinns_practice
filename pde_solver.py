# import torch
import torch 
#import torch.tanh, tensor, square, mean,concat, Tensor, torch.autograd, torch.ones_like, transpose
import torch.nn as nn
import torch.optim as optim
import inspect
from collections.abc import Iterable
from pde_strategy import pde_strategy # Neural Network Architecture
import copy
from numpy import sort
import numpy as np


class PINN(nn.Module):
    '''
    Initialize PINN
    @param layers - list of int - size of input, hidden layers, output. ex: [3,40,40,40,1]
    @param physics_loss - contains weights, points, and physics equation - see documentation for requirements/syntax
    @param boundary_loss - contains weights, points for boundary / initial condition - see documentation for requirements/syntax
    @param data_loss - for inverse problem. Trains PINN to observed data.
    '''
    def __init__(self, layers, physics_loss, boundary_loss, data_loss = None,device = None):
        self.input_dim = layers[0]
        self.input_names = [f'x{i+1}' for i in range(self.input_dim)]
        self.check_physics_loss(physics_loss)
        self.physics_conditions = physics_loss['conditions']
        self.physics_condition_weights = physics_loss['condition_weights']

        self.physics_points = physics_loss['points']     
        self.physics_point_weights = physics_loss['point_weights']
        self.check_boundary_loss(boundary_loss)
        self.boundary_points = boundary_loss['points']
        self.boundary_point_weights = boundary_loss['point_weights']
        self.use_data_loss = False
        if(data_loss != None):
            self.use_data_loss = True
            self.check_boundary_loss(data_loss)
            self.data_points = data_loss['points']
            self.data_point_weights = data_loss['point_weights']

        self.format_tensors()
        #unique pde's that will need to be calculated
        self.unique_pdes, self.pde_order_dict = self.set_up_pdes()
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

    '''
    Evaluate PINN for given inputs
    @param inputs 2d tensor to evaluate.
    '''
    def __call__(self, inputs):
        for i, layer in enumerate(self.layers):
            inputs = torch.tanh(layer(inputs)) if i < len(self.layers) - 1 else layer(inputs)
        return inputs
    
    '''
    helper function - TO COMPLETE LATER
    '''
    def order_lambda_variable(self,string):
        x= ['u'] + [string[2*i+1:2*i+3] for i in range(len(string)//2)] if len(string) > 2 else [string]
        x.sort()
        return "".join(x)
    '''
    set up optimal pde differentiation strategy.
    @return unique_pdes list of each pde to be calculated
    @return pde_map dict of string, list - telling which pdes to be calculated from which variable.
    '''
    def set_up_pdes(self):
        pdes = {'u'}
        for f in self.physics_conditions:
            pdes.update(set(inspect.signature(f).parameters))
        ls = [x for x in list(pdes) if len(x) > 2]
        xx = [self.order_lambda_variable(x) for x in ls]
        m = max([len(x) for x in xx])
        unique_pdes, pde_map = pde_strategy(xx, self.input_dim, m)
        return unique_pdes, pde_map


    '''
    list of assertions for boundary loss input. Make sure datatypes and sizes of tensors are valid.
    @param boundary loss - same as in init.
    '''
    def check_boundary_loss(self, boundary_loss):
        assert "point_weights" in boundary_loss.keys() and "points" in boundary_loss.keys()
        points = boundary_loss['points']
        point_weights = boundary_loss['point_weights']

        #points contains all x inputs AND u
        assert len(points.keys()) == self.input_dim + 1
        #assert length of weights and all points lengths equal. so there are the same number of data points in points and weights.
        assert isinstance(points, dict)

        length = len(point_weights)
        for key in points.keys():
            #points contains all x inputs AND u
            assert length == len(points[key])
            for i in range(len(points[key])):
                #assert each point in each tensor is int or float
                assert isinstance(points[key][i].item(), (int, float))
        #weights is a list
        assert isinstance(point_weights, Iterable)
        for i in range(len(point_weights)):
            assert isinstance(point_weights[i], (int,float))

        #boundary points taxonomy
        for el in points.keys():
            self.check_variable_taxonomy(el)

    '''
    list of assertions for physics loss input. Make sure data types and sizes of tensors are valid.
    @param physics loss - same as in init.
    '''
    def check_physics_loss(self,physics_loss):
        assert "conditions" in physics_loss.keys() and "condition_weights" in physics_loss.keys() and "points" in physics_loss.keys() and "point_weights" in physics_loss.keys()
        assert len(physics_loss['conditions']) == len(physics_loss['condition_weights'])
        conditions = physics_loss['conditions']
        unique_vars = {}
        '''
        Assert conditions are formatted correctly
        '''
        for condition in conditions:
            ls = inspect.signature(condition).parameters
            unique_vars.update(dict(zip(ls,[0 for i in range(len(ls))])))

        #assert correct variable taxonomy in lambda functions.
        for key in unique_vars.keys():
            self.check_variable_taxonomy(key)
        '''
        Assert weights are formatted correctly
        '''
        condition_weights = physics_loss['condition_weights']
        assert len(condition_weights) == len(conditions)
        for el in condition_weights:
            assert isinstance(el,(int, float))
        
        '''
        Assert points set up correctly
        Revision: make points a dict of str:tensor? ex: "x1" : tensor
        For now, assume it's in order: x1,x2,x3,etc.
        '''
        points = physics_loss['points']
        point_weights = physics_loss['point_weights']

        #number of input tensors must equal number of input dimensions
        assert len(points) == self.input_dim
        length = len(point_weights)
        for el in points.keys():
            assert isinstance(points[el], torch.Tensor)
            assert len(points[el]) == length
            for t in points[el] :
                assert isinstance(t.item(), (int,float))        


        for el in point_weights:
            assert isinstance(el,(int, float))

        #physics points taxonomy
        for el in points.keys():
            self.check_variable_taxonomy(el)

    #assert variable is named properly.
    '''
    Assert variable is named properly. examples of valid variables: 'x1', 'x2', 'ux1x2', 'u', 'ux1'.
    This is essential for interpreting physics loss function and input loss point tensors found in boundary loss, 
        physics loss, and data loss.
    @param: var the variable to check for taxonomy. ex: 'ux1'. !Note! this cannot be a list of strings
    '''
    def check_variable_taxonomy(self,var:str):
        #var is not an empty string.
        assert len(var) > 0
        add = 1 if var[0] == 'u' else 0
        #establish a length limit. if starts with x, it can only be two characters. else, could be longer, depending on the physics loss equation.
        len_limit = 100 if var[0] == 'u' else 2
        assert len(var) <= len_limit
        #character 0 is either u or x
        assert var[0] == 'u' or var[0] == 'x'
        #variable has an appropriate number of characters. ex. ux is not appropriate, as it is missing a number after x.
        assert (len(var) + add) % 2 == 0
        #assert that all characters (after an optional u as first character) contribute to valid independent variable names ex: ux1 or x1 or ux1x2
        for i in [2*i for i in range((len(var))//2)]:
            #all characters (after an optional u as first character) are x
            assert var[i+add] == 'x'
            #all numbers after x are between 1 and input_dim
            assert var[i+add+1].isdigit() and int(var[i+add+1]) <= self.input_dim and int(var[i+add+1]) > 0

    '''
    Make sure input tensors are set up correctly.
    1) set requires_grad = True
    2) make sure tensors are 2d and of type float

    '''
    def format_tensors(self):
        for k in self.physics_points.keys():
            self.physics_points[k].requires_grad = True
            self.physics_points[k] = self.physics_points[k].view(-1,1).float()
        
        for k in self.boundary_points.keys():
            self.boundary_points[k].requires_grad = True
            self.boundary_points[k] = self.boundary_points[k].view(-1,1).float()

        if self.use_data_loss == True:
            # print('data loss format')
            for k in self.data_points.keys():
                self.data_points[k].requires_grad = True
                self.data_points[k] = self.data_points[k].view(-1,1).float()

    '''
    Compute total loss.
    Call physics loss, boundary loss, and data loss if applicable.
    @return total loss.
    ''' 
    def total_loss(self):
        return self.physics_loss() + self.boundary_loss() + (self.data_loss() if self.use_data_loss == True else 0)
    
    '''
    Compute data loss.
    This is practically identical to boundary_loss.
    @return data loss.
    ''' 
    def data_loss(self):
        u_pred =  self(torch.torch.concat([self.data_points[self.input_names[i]] for i in range(len(self.input_names))],dim=1))
        u = self.data_points['u']
        return torch.mean(torch.square(u_pred-u) * torch.tensor(self.data_point_weights))

    '''
    Compute physics loss.
    This is practically identical to boundary_loss.
    @return data loss.
    ''' 
    def physics_loss(self):
        #concatenate input tensors to compute physics values
        inputs = torch.concat([self.physics_points[self.input_names[i]] for i in range(len(np.sort(self.input_names)))],dim=1)
        #partial derivatives - dict of string -> tensor, to store partial derivative tensors by name.
        partial_derivatives = {}
        partial_derivatives['u'] = self(inputs)
        #store values of x1, x2, x3, etc.
        for n in self.input_names:
            partial_derivatives[n] = self.physics_points[n]
        #important that pde order dict is sorted chronologically. I believe it is...
        '''
        iterate through the pde_order_dict, and compute derivatives based on instructions stored inside. 
        Currently it just computes all pdes for each tensor that is derived. probably can change this to
        improve efficiency in the future. Right now it's pretty fast, though.
        For each element to be differentiated:
        '''
        for el in self.pde_order_dict.keys():
            '''compute partial derivative'''
            tensor = torch.autograd.grad(outputs=partial_derivatives[el], inputs = inputs, grad_outputs=torch.ones_like(partial_derivatives[el]), create_graph=True)[0]

            '''add each partial derivative that was computed to partial_derivatives dict'''
            for val in np.sort(self.pde_order_dict[el]):
                i = int(val[-1])-1
                partial_derivatives[f'{el}{val}'] = tensor[:,i]
                i+=1
        sum = 0
        '''
        For each physics condition specified in setup, compute PINN physics value from given physics equation.
        By the way the user should have set up the problem, each condition is set to 0, so we use this value
        as the loss.
        '''
        for i in range(len(self.physics_conditions)):
            parameters = [self.order_lambda_variable(el) for el in list(inspect.signature(self.physics_conditions[i]).parameters)]
            tensor = torch.transpose(torch.concat([partial_derivatives[parameters[j]].view(-1,1) for j in range(len(parameters))], dim=1),0,1)
            x = self.physics_conditions[i](*tensor) 
            sum += torch.mean(torch.square(x) * torch.tensor(self.physics_point_weights)) * self.physics_condition_weights[i]
        '''return weight per number of conditions'''
        return sum/len(self.physics_conditions)

    '''
    Compute boundary loss.
    This is practically identical to boundary_loss.
    @return data loss.
    ''' 
    def boundary_loss(self):
        u_pred =  self(torch.concat([self.boundary_points[self.input_names[i]] for i in range(len(self.input_names))],dim=1))
        u = self.boundary_points['u']
        return torch.mean(torch.square(u_pred-u) * torch.tensor(self.boundary_point_weights))
        #do i enforce that 'u' is always in boundary points?
    '''
    Train the network based on total loss. user can either use this method as default, or create their own training method.
    When using, I will probably just create my own training loop for better customization. but this is an option for 
    basic tasks! or if user doesn't know much about training.
    '''
    def train(self,optimizer,max_epochs = 20000, scheduler = None):
        for epoch in range(max_epochs+1):
            optimizer.zero_grad()
            loss_value = self.total_loss()
            loss_value.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss_value.item()}')

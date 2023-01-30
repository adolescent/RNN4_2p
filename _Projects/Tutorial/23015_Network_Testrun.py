'''
All dnn network works on torchzr env. This env include cuda,cudnn and multiple useful pkgs.

This script will build a vanilla network and learn how to use it.
'''

#%% Import.
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#%% define a 3 level MST model.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()# the super function transmit parameter() into nn.Module.
        self.flatten = nn.Flatten()# define flatten function.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )# define processing sequence

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#%% This defines how to use this model.
# Put model into network.
model = NeuralNetwork().to('cuda')
print(model)
#generate random 28*28 single frame.
X = torch.rand(1, 28, 28, device='cuda')# gpu can run, but only cpu can use numpy.
logits = model(X)# this actually use forward, but do not run forward directly.
pred_probab = nn.Softmax(dim=1)(logits)# this get the probabilty of this graph's belongnings.
y_pred = pred_probab.argmax(1)# find the most possible catagory.
print(f"Predicted class: {y_pred.to('cpu')}")# return 
#%% Here we understqand each layer of this model by minibatches.
# define a minibatch of 3 random images.
input_image = torch.rand(3,28,28) # in shape (minibatchsize,height,length)
print(input_image.size()) 
# nn.flatten
# define flatten function
flatten = nn.Flatten()# flatten is still a function!
flat_image = flatten(input_image) # Use flaten function to vert image into 3*1D graph.
print(flat_image.size())
# nn.Linear
# this function calculate one step projection from input and output, all connected.
layer1 = nn.Linear(in_features=28*28, out_features=20) # define input and output cell num,also a function.
hidden1 = layer1(flat_image) # calculate layer 1, change 768 pix into 20 neuron response.
print(hidden1.size()) # batchsize =3, so generate 3 projection.
#nn.ReLU
# this function is used to rectal input data to avoid negtive output.
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
#nn.Sequential
# this function process commands step by step.
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
# nn.Softmax
# softmax function change value between 0 to 1 and have a logistic-like value plot.
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
#%% Model Paremeters
# This part will show the parameter of model.
print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


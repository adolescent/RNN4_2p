'''
This is the toy model of a vanilla full connection network.
Use FashionMIST data set,
path = r'D:\ZR\_My_Codes\RNN4_2p\_Projects\Tutorial\data'

'''
#%% Import 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
data_root = r'D:\ZR\_My_Codes\RNN4_2p\_Projects\Tutorial\data'
#%% data acquaire, Use Fasion MNIST model, 10000 test and 60000 training.
training_data = datasets.FashionMNIST(
    root=data_root,
    train=True, # this is training set
    download=False, # no need to redownload 
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root=data_root,
    train=False,# test set.
    download=False,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


#%% Define model, still a flatten and 3 layer MLP
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Use the model.
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model = NeuralNetwork().to('cuda')# start with no parameters

#%% Hyper parameters
# Hyper parameters are the key to best model function.choose carefully.
learning_rate = 1e-3 # bigger learning rate train faster, but may lose best point.
batch_size = 64 # size of batch, parameter update only batch is over.
epochs = 5 # the number times to iterate over the dataset
#%% Loss Function
# loss function to evaluate model effect, basically the difference between test and predict.

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

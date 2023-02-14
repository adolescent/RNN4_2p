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
#%% Optimizer
# This is actually 
# Optimization is the process of adjusting model parameters to reduce model error in each training step
# In this example we use Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%% Training loop
# This is how we train a model. 
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset) # size of dataset, how many datas you have.
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(device)) # use current parameter to generate current prediction
        loss = loss_fn(pred, y.to(device)) # get prediction error. 
        # Here, pred is a proba-distribution, but y is class incidies.
        # loss_fn also accept y in type of proba-distribution.
        

        # Backpropagation, this is standard operation.
        optimizer.zero_grad() # reset grad to avoid double counting
        loss.backward() # BP loss function
        optimizer.step() # step current batch back into model parameters


        if batch % 100 == 0: # report every 100 batches.
            loss, current = loss.item(), batch * len(X) # real number is batch*num in batch.
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
#%% Test loop
# parameters fixed in this loop. we will see the effect of it.


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad(): # operation inside this will not track grad, so will not train.
        for X, y in dataloader:
            pred = model(X.to(device)) # prediction value.
            test_loss += loss_fn(pred, y.to(device)).item() # add up loss of predition.
            correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item() # number of correct classification.

    test_loss /= num_batches # loss in train calculate each batch, so here we need to divide num of batches.
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#%% Real training

loss_fn = nn.CrossEntropyLoss() # use cross entropy loss.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10 # times of cycles.
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

#%% Save and Load
# Standard save use torch save.
torch.save(model, 'model.pth')
reloaded_model = torch.load('model.pth')

# or you can just pickle the whole model.
# import OS_Tools_Kit as ot
# ot.Save_Variable(r'D:\ZR\_My_Codes\RNN4_2p\_Projects\230120_Toy_Model','Test_Model',model)



'''
This is a MLP model used for sine wave prediction.

'''


#%% Import 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from tqdm import tqdm
import time

start_time = time.time()
if torch.cuda.is_available(): # On which platform to run.
    device = 'cuda'
else:
    device = 'cpu'

#%% generate sine sequence
sine_series = np.sin(np.linspace(0,2*np.pi,3000))
seq_len = 30 # try 30 to 1 first.
pair_num = len(sine_series)-seq_len-1 # consider the last one of test result.
pairs_list = []
for i in tqdm(range(pair_num)):
    train_set = sine_series[i:i+seq_len]
    test_set = sine_series[i+seq_len+1]
    pairs_list.append((train_set,test_set))
# use 9/10 as train, 1/10 as test.
test_ratio = 0.1
deter_num = int(len(pairs_list)*(1-test_ratio))
train_list = pairs_list[:deter_num]
test_list = pairs_list[deter_num:]

#%% Define data sets.
# This will use tutoral in pytorch to generate datasets.
from torch.utils.data import Dataset
class Manual_Dataset(Dataset): # This class is specified for data format above.
    
    def __init__(self,input_lists,device = 'cpu', transform=None, target_transform=None):
        
        self.input_lists = input_lists
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.input_lists)
        
    def __getitem__(self, idx):
        
        input_series = torch.tensor(self.input_lists[idx][0]).to(torch.float32).to(self.device)
        results = torch.tensor(self.input_lists[idx][1]).to(torch.float32).to(self.device)
        # data need to be float 32 tensors, here we also preload data into gpu.
        
        return input_series, results
        

# %% Define a 2-hidden MLP model.
class MLP(nn.Module):
    name = '2 Hidden Layer Percepton'
    def __init__(self):
        super(MLP, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)# in this model, this is not logits.
        return logits

model = MLP().to(device)
#%% Define hyper parameters
# Hyper parameters are the key to best model function.choose carefully.
learning_rate = 1e-4 # bigger learning rate train faster, but may lose best point.
batch_size = 4 # size of batch, parameter update only batch is over.
epochs = 20 # the number times to iterate over the dataset

#%% Load data into dataloader, have batchsize above.
train_data = Manual_Dataset(train_list,device=device)
test_data = Manual_Dataset(test_list,device=device)
train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle = False)
test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle = False)

#%% methods.
# loss_fn = nn.CrossEntropyLoss() # Cross Entropy is good for classification, but in regression,MSE is better.
loss_fn = nn.MSELoss()
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
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2) # This is grad clipping to avoid grad explosion.
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

    test_loss /= num_batches # loss in train calculate each batch, so here we need to divide num of batches.
    correct /= size
    print(f"Avg loss: {test_loss:>8f} \n")
#%% Do training and testing
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
end_time = time.time()
timecost = end_time-start_time
print(f'Time Cost: {timecost}')

#%% Plot prediciton plots
import matplotlib.pyplot as plt
pred = []
with torch.no_grad():
    for i in range(700):
        input_series = sine_series[2200+i:2230+i]
        c_pred = model(torch.tensor(input_series).to(torch.float32).to(device))
        pred.append(c_pred.to('cpu').numpy())
plt.switch_backend('webAgg') #set graph into a web, so we can interactive.
plt.plot(sine_series)
plt.plot(range(2200,2900),pred)
plt.show()

#%% iterational prediction
# Use current prediction as input, the iterate into 
initial_input = sine_series[2200:2230]
X = initial_input
predictions = []
with torch.no_grad():
    for i in range(500): # cycle 500 points.
        c_y = model(torch.tensor(X).to(torch.float32).to(device))
        predictions.append(np.array(c_y.to('cpu')))
        X = np.append(X[:-1],np.array(c_y.to('cpu')))


'''
This script will generate prediction of all cell inputs. 
Using LSTM network, with all cell in and all cell out.
'''


#%% Import 

from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import OS_Tools_Kit as ot
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import time

start_time = time.time()
if torch.cuda.is_available(): # On which platform to run.
    device = 'cuda'
else:
    device = 'cpu'

# plt.switch_backend('webAgg') 
# plt.plot(loss_seq)
# plt.show()
#%% read in test series, use L76.
test_series = ot.Load_Variable(r'D:\ZR\_Temp_Data\220711_temp\Series76_Run01_4000.pkl')
series_avr = test_series.mean(0)
def EZPlot(series) -> None:
    plt.switch_backend('webAgg') 
    plt.plot(series)
    plt.show()
    
def Single_Batch_Shape(dataloader):# print
    
    for batch, (X, y) in enumerate(dataloader):
        if batch == 0:
            print(f'X shape :{X.shape}')
            print(f'y shape :{y.shape}')
            break
    return X,y

#%% Define training parameters
# Hyper parameters are the key to best model function.choose carefully.
cell_num = test_series.shape[0]
learning_rate = 5e-3 # bigger learning rate train faster, but may lose best point.
batch_size = 2 # size of batch, parameter update only batch is over.
epochs = 50 # the number times to iterate over the dataset
# template_cell = test_series.loc[23,:] Use all cell in this case.
seq_len = 30 # sequence of series.
test_ratio = 0.1 # propotion of test sets.
output_len = 3

#%%
timepoints = test_series.shape[1]
pair_num = timepoints-seq_len-output_len
pairs_list = []
for i in range(pair_num):
    train_set = np.array(test_series.iloc[:,i:i+seq_len])
    test_set = np.array(test_series.iloc[:,i+seq_len:i+seq_len+output_len]).flatten()
    pairs_list.append((train_set,test_set))

deter_num = int(len(pairs_list)*(1-test_ratio))
train_list = pairs_list[:deter_num]
test_list = pairs_list[deter_num:]

# %% Define data set class.
class All_Cell_Dataset(Dataset):
    name = 'All_Cell_Dataset'
    def __init__(self,input_lists,device = 'cpu'):
        self.input_lists = input_lists
        self.device = device
        
    def __len__(self):
        return len(self.input_lists)
    def __getitem__(self,idx):
        
        input_series = torch.tensor(self.input_lists[idx][0]).T.to(torch.float32).to(self.device)
        results = torch.tensor(self.input_lists[idx][1]).to(torch.float32).to(self.device)
        
        return input_series,results
# Load data into dataloader, have batchsize above.
train_data = All_Cell_Dataset(train_list,device=device)
test_data = All_Cell_Dataset(test_list,device=device)
train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle = True)
test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle = True)

#%% Define single unit LSTM model.
class RNN(nn.Module):
    name = 'many to many One Layer RNN,batched.'
    def __init__(self,input_size = 1,hidden_unit=256,layer_num = 1,output_len = 20):
        super(RNN, self).__init__()
        self.hidden_unit = hidden_unit
        self.layer_num = layer_num
        self.output_len = output_len
        self.input_size = input_size
        self.rnn = nn.LSTM(self.input_size,self.hidden_unit,self.layer_num,batch_first = True)
        # Here we define the network structure. 1 input-dim(1D signal),20 hidden neurons, and 1 layer only.
        self.fc1 = nn.Linear(self.hidden_unit,self.output_len*self.input_size) # full connection layer1, only 1 output.
        # this is enough. One Layer RNN and one fc
        
    def forward(self,x):
        # initialize h0 with zero axis
        # while using RNN, you can give h0, but it's not a must if you have no good guess.
        rnn_out,(hn,cn) = self.rnn(x)
        out = self.fc1(rnn_out[:,-1,:])
        return out
    
model = RNN(input_size = cell_num,hidden_unit=1024,layer_num = 1,output_len = output_len).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
#%% Training loop
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2) # This is grad clipping to avoid grad explosion.
        optimizer.step() # step current batch back into model parameters
        if batch % 1000 == 0: # report every 100 batches.
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
    return test_loss
#%%
#%% cycle train.
loss_seq = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    loss_seq.append(test_loop(test_dataloader, model, loss_fn))
print("Done!")
end_time = time.time()
timecost = end_time-start_time
print(f'Time Cost: {timecost}')


#%% Predict next N frame inputs.
import matplotlib.pyplot as plt
pred = np.zeros(shape = (cell_num,1))
with torch.no_grad():
    for i in range(200):
        input_series = np.array(test_series.iloc[:,8500+i*output_len:8500+i*output_len+seq_len]).T
        c_pred = model(torch.tensor(np.array([input_series])).to(torch.float32).to(device)).to('cpu').numpy()
        c_pred = c_pred.flatten().reshape(cell_num,output_len)
        pred = np.concatenate((pred,c_pred),axis = 1)
        
import seaborn as sns
from scipy.stats import pearsonr
plt.switch_backend('webAgg') #set graph into a web, so we can interactive.
sns.heatmap(pred[:,1:])
plt.figure()
sns.heatmap(np.array(test_series.iloc[:,8530:8530+600]))
# plt.plot(range(500+seq_len,500+seq_len+5000),pred)
plt.show()
r,p = pearsonr(pred[:,1:].flatten(),np.array(test_series.iloc[:,8530:8530+600]).flatten())
print(f'Predicion have a total r = {r},with p = {p}')

             
#%%
plt.switch_backend('webAgg') #set graph into a web, so we can interactive.
plt.plot(loss_seq)
plt.show()
# torch.save(model,'cell23_L76_LSTM-5pred.pth')

#%% Cell by Cell Corr
corr_pair = []
for i in range(cell_num):
    c_a = pred[i,1:]
    c_b = np.array(test_series.iloc[i,8530:8530+600])
    c_r,_ = pearsonr(c_a,c_b)
    corr_pair.append(c_r)
    
plt.figure()
plt.hist(corr_pair,bins = 20)
plt.show()


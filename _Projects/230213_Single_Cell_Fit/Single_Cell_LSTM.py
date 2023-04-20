'''
LSTM model used for single cell data processing

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
import random

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
#%% Define training parameters
# Hyper parameters are the key to best model function.choose carefully.
learning_rate = 5e-3 # bigger learning rate train faster, but may lose best point.
batch_size = 4 # size of batch, parameter update only batch is over.
epochs = 300 # the number times to iterate over the dataset
# template_cell = test_series.loc[23,:]
template_cell = np.array(test_series.loc[23,:])[::-1]
# template_cell =  np.array(random.sample(list(template_cell),k = len(template_cell)))
seq_len = 50 # sequence of series.
test_ratio = 0.1 # propotion of test sets.
output_len = 5

#%% process data into lstm fit format.
timepoints = len(template_cell)
pair_num = timepoints-seq_len-output_len
pairs_list = []
for i in range(pair_num):
    train_set = template_cell[i:i+seq_len]
    test_set = template_cell[i+seq_len:i+seq_len+output_len]
    pairs_list.append((train_set,test_set))
# use 0.9 to train,0.1 to test.
deter_num = int(len(pairs_list)*(1-test_ratio))
train_list = pairs_list[:deter_num]
test_list = pairs_list[deter_num:]
# Define data sets.
'''
Transfer pairs_list into pytorch dataset.
'''
class Manual_Data_Set(Dataset):
    
    def __init__(self,input_lists,device = 'cpu', transform=None, target_transform=None):
                
        self.input_lists = input_lists
        self.device = device
        self.transform = transform
        self.target_transform = target_transform

        
    def __len__(self):
        return len(self.input_lists)
    
    def __getitem__(self,idx):
        input_series = torch.tensor(np.array([self.input_lists[idx][0]]).T).to(torch.float32).to(self.device)
        results = torch.tensor(np.array(self.input_lists[idx][1]).T).to(torch.float32).to(self.device)
        # data need to be float 32 tensors, here we also preload data into gpu.
        return input_series, results
    
# Load data into dataloader, have batchsize above.
train_data = Manual_Data_Set(train_list,device=device)
test_data = Manual_Data_Set(test_list,device=device)
train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle = True)
test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle = True)



#%% Define single unit LSTM model.
class RNN(nn.Module):
    name = 'many to many One Layer RNN,batched.'
    def __init__(self,hidden_unit=256,layer_num = 1,output_len = 20):
        super(RNN, self).__init__()
        self.hidden_unit = hidden_unit
        self.layer_num = layer_num
        self.output_len = output_len
        self.rnn = nn.LSTM(1,self.hidden_unit,self.layer_num,batch_first = True)
        # Here we define the network structure. 1 input-dim(1D signal),20 hidden neurons, and 1 layer only.
        self.fc1 = nn.Linear(self.hidden_unit,self.output_len) # full connection layer1, only 1 output.
        # this is enough. One Layer RNN and one fc
        
    def forward(self,x):
        # initialize h0 with zero axis
        # while using RNN, you can give h0, but it's not a must if you have no good guess.
        rnn_out,(hn,cn) = self.rnn(x)
        out = self.fc1(rnn_out[:,-1,:])
        return out
    
model = RNN(hidden_unit=256,output_len = output_len).to(device)
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=8, norm_type=2) # This is grad clipping to avoid grad explosion.
        optimizer.step() # step current batch back into model parameters
        if batch % 500 == 0: # report every 100 batches.
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
pred = []
with torch.no_grad():
    for i in range(100):
        input_series = template_cell[8500+i*output_len:8500+i*output_len+seq_len]
        c_pred = model(torch.tensor(np.array([np.array([input_series]).T])).to(torch.float32).to(device)).to('cpu').numpy()
        pred.extend(list(c_pred[0,:]))
        
plt.switch_backend('webAgg') #set graph into a web, so we can interactive.
# plt.plot(loss_seq)
# plt.show()
plt.plot(np.array(template_cell))
plt.plot(range(8500+seq_len,8500+seq_len+500),pred)
plt.show()
#%%
plt.switch_backend('webAgg') #set graph into a web, so we can interactive.
plt.plot(loss_seq)
plt.show()
# torch.save(model,'cell23_L76_LSTM-5pred.pth')

#%% Calculate pearsonr
from scipy.stats import pearsonr
r,p = pearsonr(np.array(pred),template_cell[8550:9050])
print(f'Pearson r = {r};with p ={p}')
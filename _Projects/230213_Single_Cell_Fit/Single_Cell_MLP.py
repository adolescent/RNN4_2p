
'''

Single cell prediction model of MLP.
Output next 1 time point only, if you want 2 or more, set new nets plz.


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
#%% Define training parameters
# Hyper parameters are the key to best model function.choose carefully.
learning_rate = 1e-4 # bigger learning rate train faster, but may lose best point.
batch_size = 4 # size of batch, parameter update only batch is over.
epochs = 700 # the number times to iterate over the dataset
template_cell = test_series.loc[23,:]
seq_len = 30 # sequence of series.
test_ratio = 0.1 # propotion of test sets.
output_len = 5


timepoints = len(template_cell)
pair_num = timepoints-seq_len
pairs_list = []
for i in range(pair_num):
    train_set = template_cell[i:i+seq_len]
    test_set = template_cell[i+seq_len:i+seq_len+output_len]
    pairs_list.append((train_set,test_set))
# use 0.9 to train,0.1 to test.
deter_num = int(len(pairs_list)*(1-test_ratio))
train_list = pairs_list[:deter_num]
test_list = pairs_list[deter_num:-10]


#%% Define data sets.
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
        input_series = torch.tensor(np.array(self.input_lists[idx][0])).to(torch.float32).to(self.device)
        results = torch.tensor(np.array(self.input_lists[idx][1])).to(torch.float32).to(self.device)
        # data need to be float 32 tensors, here we also preload data into gpu.
        
        return input_series, results
#%% Define MLP model.
class MLP(nn.Module):
    name = r'2 Hidden Layer MLP'
    
    def __init__(self, input_size = 30, hidden_size= 128, output_size=1):
        super(MLP,self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        
    def forward(self,x):
        
        predictions = self.linear_relu_stack(x)
        return predictions

model = MLP(input_size = seq_len,output_size = output_len).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%% Load data into dataloader, have batchsize above.
train_data = Manual_Data_Set(train_list,device=device)
test_data = Manual_Data_Set(test_list,device=device)
train_dataloader = DataLoader(train_data,batch_size=batch_size,shuffle = True)
test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle = True)

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
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2) # This is grad clipping to avoid grad explosion.
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

#%% Do training and testing
loss_list = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    c_loss = test_loop(test_dataloader, model, loss_fn)
    loss_list.append(c_loss)
print("Done!")
end_time = time.time()
timecost = end_time-start_time
print(f'Time Cost: {timecost}')

#%% Predict next N frame.
# import matplotlib.pyplot as plt
# pred = []
# with torch.no_grad():
#     for i in range(100):
#         input_series = template_cell[8500+i:8530+i]
#         c_pred = model(torch.tensor(np.array(input_series)).to(torch.float32).to(device))
#         pred.append(c_pred.to('cpu').numpy())
# plt.switch_backend('webAgg') #set graph into a web, so we can interactive.
# # plt.plot(loss_list)
# plt.plot(np.array(template_cell))
# plt.plot(range(8520,9020),pred)
# plt.show()
# # torch.save(model, 'Cell23_L76_MLP_model.pth')
#%% Predict next N frame inputs.
import matplotlib.pyplot as plt
pred = []
with torch.no_grad():
    for i in range(600):
        input_series = template_cell[8500+i*output_len:8500+i*output_len+seq_len]
        c_pred = model(torch.tensor(np.array(input_series)).to(torch.float32).to(device)).to('cpu').numpy()
        pred.extend(list(c_pred))
        
plt.switch_backend('webAgg') #set graph into a web, so we can interactive.
# plt.plot(loss_list)
# plt.show()
plt.plot(np.array(template_cell))
plt.plot(range(8500+seq_len,8500+seq_len+600),pred)
plt.show()

#%% iteration prediction
initial_input = template_cell[8500:8500+seq_len]
X = np.array(initial_input)
predictions = []
with torch.no_grad():
    for i in range(100):
        c_pred = model(torch.tensor(X).to(torch.float32).to(device)).to('cpu').numpy()
        X = np.append(X[:-output_len],c_pred)
        predictions.extend(list(c_pred))
    

# %% 

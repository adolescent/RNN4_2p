'''
This script will stat all cell prediction front vs backward.
Negative result predicted..

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
from My_Wheels.Analyzer.My_FFT import FFT_Power,FFT_Window_Slide

start_time = time.time()
if torch.cuda.is_available(): # On which platform to run.
    device = 'cuda'
else:
    device = 'cpu'
#%% read in test series, use L76.
test_series = ot.Load_Variable(r'D:\ZR\_Temp_Data\220711_temp\Series76_Run01_4000.pkl')
series_avr = test_series.mean(0)
def EZPlot(series) -> None:
    plt.switch_backend('webAgg') 
    plt.plot(series)
    plt.show()

# get freq power of origin series.
seq_len = 50 # sequence of series.
output_len = 5

#%% Load model
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
    
model = torch.load(r'cell23_L76_LSTM-5pred.pth')

#%% Compare forward and back ward, this calculation is slow.
cell_num,frame_num = test_series.shape
from scipy.stats import pearsonr
for_r = []
bac_r = []
for i in tqdm(range(cell_num)):
    c_series = np.array(test_series.iloc[i,:])
    rev_series = np.array(test_series.iloc[i,:])[::-1]
    pred_f = []
    # forward prediction here.
    with torch.no_grad():
        for i in range(140):
            input_series = c_series[8450+i*output_len:8450+i*output_len+seq_len]
            c_pred = model(torch.tensor(np.array([np.array([input_series]).T])).to(torch.float32).to(device)).to('cpu').numpy()
            pred_f.extend(list(c_pred[0,:]))
    r_f,_ = pearsonr(np.array(pred_f),c_series[8500:9200])
    for_r.append(r_f)
    # reversed prediction here.
    pred_b = []
    with torch.no_grad():
        for i in range(140):
            input_series = rev_series[22+i*output_len:22+i*output_len+seq_len]
            c_pred = model(torch.tensor(np.array([np.array([input_series]).T])).to(torch.float32).to(device)).to('cpu').numpy()
            pred_b.extend(list(c_pred[0,:]))
    r_b,_ = pearsonr(np.array(pred_b),rev_series[22+50:22+50+700])
    bac_r.append(r_b)
    
#%%
r_diff = []
for i in range(len(bac_r)):
    c_diff = for_r[i]-bac_r[i]
    r_diff.append(c_diff)
    
    
#%%
plt.switch_backend('webAgg') 
plt.hist(for_r,bins = 20)
plt.hist(bac_r,bins = 20)
plt.show()
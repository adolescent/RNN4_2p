
'''
This script will shuffle phase of a single series, but remian freq distribution unchanged.
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

#%% get freq power of origin series.
origin_cell = np.array(test_series.loc[23,:])
seq_len = 50 # sequence of series.
output_len = 5
origin_power = FFT_Power(origin_cell,signal_name = 'origin')
freq_power = np.fft.fft(origin_cell)
amp = abs(freq_power)
angle = np.angle(freq_power)
shuffled_angle = np.random.random(len(angle))*2*np.pi-np.pi
# Euler format transformation.
shuffled_freq_power = amp*np.cos(shuffled_angle)+1j*amp*np.sin(shuffled_angle)
restored_series = np.fft.ifft(shuffled_freq_power).real

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


#%% Predict next N frame inputs.
import matplotlib.pyplot as plt
pred = []
with torch.no_grad():
    for i in range(1800):
        input_series = restored_series[0+i*output_len:0+i*output_len+seq_len]
        c_pred = model(torch.tensor(np.array([np.array([input_series]).T])).to(torch.float32).to(device)).to('cpu').numpy()
        pred.extend(list(c_pred[0,:]))
        
plt.switch_backend('webAgg') #set graph into a web, so we can interactive.
# plt.plot(loss_seq)
# plt.show()
plt.plot(np.array(restored_series))
plt.plot(range(0+seq_len,0+seq_len+9000),pred)
plt.show()

#%% Calculate pearsonr
from scipy.stats import pearsonr
r,p = pearsonr(np.array(pred),restored_series[50:9050])
print(f'Pearson r = {r};with p ={p}')

#%% Auto correlation

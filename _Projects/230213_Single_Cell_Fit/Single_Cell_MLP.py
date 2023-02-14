
'''

Single cell prediction model of MLP.
Output next 1 time point only, if you want 2 or more, set new nets plz.


'''
#%% Import 

from Series_Analyzer.Preprocessor_Cai import Pre_Processor_Cai
import OS_Tools_Kit as ot
import matplotlib.pyplot as plt


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
#%% # cut series into series we need.
template_cell = test_series.loc[23,:]
seq_len = 30 # sequence of series.
test_prop = 0.1 # propotion of test sets.


timepoints = len(template_cell)
pair_num = timepoints-seq_len



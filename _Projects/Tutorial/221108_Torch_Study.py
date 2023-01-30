


#%%

import torch
import numpy as np
x = torch.rand(5, 3)
print(x)
print(f'Cuda aviable: {torch.cuda.is_available()}')

print('This script runs well.')

# %% Tensor can be defined directly from data or array like.
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
# %%
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
#%%
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n{x_ones} \n")
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
#%%
x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
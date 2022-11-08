


#%%

import torch
import numpy as np
x = torch.rand(5, 3)
print(x)
torch.cuda.is_available()

print('This script runs well.')

# %% Tensor can be defined directly from data or array like.
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
# %%

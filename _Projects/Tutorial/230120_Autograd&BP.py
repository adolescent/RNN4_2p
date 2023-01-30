'''
This part will explain Autogradient function and back propogation.
Such function will established insed pytorch, and is the basic function of model training.

'''

#%% Import
import torch


#%% define different random variable and loss function.
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output

# w and b are model parameter.
w = torch.randn(5, 3, requires_grad=True)# step1, multiple matrix
b = torch.randn(3, requires_grad=True)# step2, add to matrix.
z = torch.matmul(x, w)+b# predicted output of model. X is input and W is matrix.
#Input is always on the first place of matmul.

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# loss function, calculate difference between predict and real output.
# Use binary cross entropy method to calculate loss funnction.
#%% the gradient command is a function.
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")
#%% Use backward to prop result back.
loss.backward() # this will prop loss value into sequence before,
print(w.grad)
print(b.grad)
# Remember, only values set 'requires_grad = True' have grad property, which indicates this variable will bp.

# use torch.no_grad can stop grad tracking for some reason.
z = torch.matmul(x, w)+b
print(z.requires_grad)
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
# use tensor.detach() can do the same job.
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
#%%
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)# retain_graph = True allow second attempt of backward.
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")
#retain_graph = True will allow second gradient
# tensor.backward() is tensor.backward(tensor([1.0]))
# and you need to set tensor.grad.zero_() to reset gradient.
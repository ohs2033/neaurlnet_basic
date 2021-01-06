import torch

x = torch.ones(2,2 , requires_grad=True)
print(x)


y = x+2
print(y, y.grad_fn)

z = y*y*3
print(z, z.grad_fn)

z = torch.sum(z)
print(z)

z.backward()

print(x.grad) # must be 8

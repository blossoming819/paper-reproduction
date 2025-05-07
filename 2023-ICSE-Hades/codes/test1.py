import torch
a = torch.arange(60, dtype=float).reshape((3, 4, 5, 1))
print(a.shape)
a = torch.mean(a, dim=-1)
print(a.shape)
a = a.squeeze()
print(a.shape)
a = a.permute(1, 2, 0)
print(a.shape)
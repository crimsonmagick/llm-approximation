import torch

original = torch.arange(24)
print('original:')
print(original)
print(f'original shape: {original.shape}')

shaped = original.view(2, 3, 4)
print('shaped:')
print(shaped)
print(f'shaped shape: {shaped.shape}')

expanded = shaped[:,:, None, :]
print(expanded)
expanded = expanded.expand(2, 3, 2, 4)
print(expanded)
expanded = expanded.reshape(2, 6, 4)
print(expanded)


#
# reshaped = original.view(4, 6)
# print('reshaped:')
# print(reshaped)
# print(f'reshaped shape: {reshaped.shape}')
#
# reshaped_2 = reshaped.view(-1, 8)
# print('reshaped_2:')
# print(reshaped_2)
# print(f'reshaped_2 shape: {reshaped_2.shape}')
#
# print('hi')
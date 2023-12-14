import numpy as np
import torch
# import paddle
import sys


input_fp = "init_param/init_burgers.pth"
output_fp = "../PINO_paddle/init_param/init_burgers.pdparams"
torch_dict = torch.load(input_fp, map_location=torch.device('cpu'))
paddle_dict = {}
for key in torch_dict:
    weight = torch_dict[key].cpu().detach().numpy()
    check = 'fc'
    if check in key:
        # print("weight {} need to be trans".format(key))
        weight = weight.transpose()
    if weight.dtype == 'complex64':
        weight_real = weight.real
        weight_imag = weight.imag
        check = 'weights1'
        if check in key:
            key_real = key.replace('weights1','weight.0.real')
            key_imag = key.replace('weights1','weight.0.imag')
        check = 'weights2'
        if check in key:
            key_real = key.replace('weights2','weight.1.real')
            key_imag = key.replace('weights2','weight.1.imag')
        paddle_dict[key_real] = weight_real
        paddle_dict[key_imag] = weight_imag
        print(key_real)
    else:
        key = key.replace('lin', 'linear')
        print(key)
        paddle_dict[key] = weight
# paddle.save(paddle_dict, output_fp)

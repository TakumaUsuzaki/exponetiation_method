from exponentiation_deeplearning import exponentiation_dl
import torch
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

"""
exponent = 5.0
epoch = 100
"../tests/train1"
"../tests/train2"
"../tests/test1"
"../tests/test2"
"../test"
"""

if __name__ == "__main__":
    exponent = input('Please Enter an exponent value: ')
    epoch = input('Please Enter the epoch: ')
    batch = input('Please Enter the batch size: ')
    print("Now {0} power is calculating".format(exponent))
    exponentiation_dl(float(exponent), int(epoch), int(batch))

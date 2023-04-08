import numpy as np
import torch
import torch.nn as nn


def conv_block(input_size, kernel_size):
    """Mendefinisikan Convolutional NN block untuk model Sequential CNN. """
    
    block = Sequential(
        nn.Conv2d(input_size, input_size+96, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(input_size+96, input_size+64, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        # BatchNorm2d(),
        nn.MaxPool2d(2, 2)
    )
    
    return block
    
def dense_block(input_size, output_unit, dropout_rate):
    """Mendefinisikan Dense NN block untuk model Sequential CNN. """
    
    block = Sequential(
        nn.Linear(input_size, output_unit),
        nn.ReLU(),
        # BatchNorm2d(),
        nn.Dropout(dropout_rate)
    )
    
    return block
    

def net(num_classes, input_size, kernel_size, dropout_rate):
    
    model = nn.Sequential(
        nn.Conv2d(input_size, 128 , kernel_size=kernel_size, padding='same'),
        nn.ReLU(),
        nn.Conv2D(128, 64, kernel_size=kernel_size, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
        conv_block(input_size=64, kernel_size=kernel_size),
        conv_block(input_size=128, kernel_size=kernel_size),
        conv_block(input_size=192, kernel_size=kernel_size),
        nn.Dropout(dropout_rate),
        conv_block(input_size=256, kernel_size=kernel_size),
        nn.Dropout(dropout_rate),
        nn.Flatten(),
        dense_block(input_size=256*(kernel_size+1)*(kernel_size+1), output_unit=512, 0.7),
        dense_block(input_size=512, output_unit=128, 0.5),
        dense_block(input_size=128, output_unit=64, 0.3),
        nn.Linear(64, num_classes)
    )
    
    return model
    
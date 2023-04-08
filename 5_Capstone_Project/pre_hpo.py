import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True #disable image truncated error

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

num_classes = 4

def test(model, test_loader, criterion, device):
    '''
    Function that tests the model
    
    Args:
        - model:
        - test_loader: data_loader of test dataset
        - criterion: loss function
        - device: CPU/ GPU
        
    '''
    
    model.eval() 
    
    running_loss = 0.0
    running_corrects = 0
    running_samples = 0
    total_samples = len(test_loader.dataset)
    

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, pred = torch.max(outputs, 1)

        running_loss += loss.item()*inputs.size(0)  # sum up batch loss
        running_corrects += torch.sum(pred == labels.data).item()
        running_samples += len(inputs)
        
        if running_samples>(1*total_samples):
            break

    total_loss = running_loss/total_samples
    total_acc = running_corrects/total_samples

    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")

    
def validate(model, valid_loader, criterion, device):
    '''
    Function that validates the model
    
    Args:
        - model:
        - valid_loader: data_loader of valid dataset
        - criterion: loss function
        - device: CPU/ GPU
    
    Returns:
        - total_loss: validation loss
        - total_acc: validation accuracy
    '''
    
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    running_samples = 0
    total_samples = len(valid_loader.dataset)
    
    for inputs, labels in valid_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
    
        _, pred = torch.max(outputs, 1)
        
        running_loss += loss.item()*inputs.size(0)  # sum up batch loss
        running_corrects += torch.sum(pred == labels.data).item()
        running_samples += len(inputs)
        
        if running_samples>(1*total_samples):
            break

    total_loss = running_loss/running_samples
    total_acc = running_corrects/running_samples
    
    return total_loss, total_acc


def train(model, train_loader, criterion, optimizer, device):
    '''
    Function that trains the model
    
    Args:
        - model:
        - train_loader: data_loader of train dataset
        - criterion: loss function
        - optimizer: model optimizer
        - device: CPU/ GPU
    
    Returns:
        - epoch_loss: training epoch loss
        - epoch_acc: training epoch accuracy
        
    '''  
    model = model.to(device)
    model.train()

    running_loss = 0.0
    running_corrects = 0
    running_samples = 0
    total_samples = len(train_loader.dataset)

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(outputs, 1)
        running_loss += loss.item()*inputs.size(0)
        running_corrects += torch.sum(pred == labels.data).item()
        running_samples += len(inputs)
        
        if running_samples>(1*total_samples):
            break
    
    total_loss = running_loss/running_samples
    total_acc = running_corrects/running_samples
    
    return total_loss, total_acc


def model_train_validate(model, data_loaders, epochs, criterion, optimizer, device):
    '''
    Train and validate the model by calling `train` and `validate` functions
    
    Args:
        - model:
        - data_loaders, dict: dictionary of data_loaders of train, test and valid
        - epochs: number of epoch to train the model
        - criterion: loss function
        - optimizer: model optimizer
        - device: CPU/ GPU
        
    '''
           
    best_loss=1e6    
    
    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(model, data_loaders['train'], criterion, optimizer, device)
        valid_loss, valid_acc = validate(model, data_loaders['valid'], criterion, device)
        
        print("Epoch {}/{} . . . ".format(epoch, epochs+1))
        logger.info('Train loss: {:.4f}, acc: {:.4f}'.format( train_loss, train_acc ))
        logger.info('Valid loss: {:.4f}, acc: {:.4f}'.format( valid_loss, valid_acc ))
        
        
        if valid_loss<best_loss:
            best_loss=valid_loss
        else:
            print("Validation loss is increasing")
            break

            
def model_fn(model_dir):
    ''' Load the model '''
    
    model = net(num_classes)
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    print("model loaded")
    return model


def save_model(model, model_dir):
    ''' Save the model '''
    
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), path)
    print(f"Model saved to path: {path}")
    

def net(num_classes):
    '''Initializes the pre-trained model '''
    
    model = models.vgg16(pretrained=True)

    for param in model.parameters(): 
        param.requires_grad=False
    
    # num_features = model.fc.in_features
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 2048),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(2048, 516),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(516, num_classes)
    )
    
    return model


def create_data_loaders(data_dir, batch_size, data_type):
    '''
    Loads and transforms the data and creates `DataLoader` object
    
    Args:
        - data: dataset
        - batch_size: batch size
        
    Return:
        - data_loader: DataLoader object 
        
    '''
    transform_train = transforms.Compose([
        transforms.Resize(200),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.497,0.402,0.425), (0.308, 0.325, 0.301))
    ])
    
    transform_valid = transforms.Compose([
        transforms.Resize(200), 
        transforms.ToTensor(),
        transforms.Normalize((0.497,0.402,0.425), (0.308, 0.325, 0.301))
    ])
    
    if data_type=='train':
        data_image = ImageFolder(root=data_dir, transform=transform_train)
        data_loader = torch.utils.data.DataLoader(data_image, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        data_image = ImageFolder(root=data_dir, transform=transform_valid)
        data_loader = torch.utils.data.DataLoader(data_image, batch_size=batch_size, shuffle=True, num_workers=2)
        
    return data_loader
    

def main(args):
    # initialized the model
    model=net(num_classes)
    
    # define the loss function and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
         
    # setup the device type
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # load the data
    batch_size=args.batch_size
    data_loaders = {'train':create_data_loaders(args.train, batch_size, 'train'),
                    'test':create_data_loaders(args.test, batch_size,'test'),
                    'valid':create_data_loaders(args.valid, batch_size,'valid')
                   }

    # train and test the model
    model_train_validate(model, data_loaders, args.epochs, loss_criterion, optimizer, device)
    test(model, data_loaders['test'], loss_criterion, device)

    # save the model
    save_model(model, args.model_dir)
    
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        metavar="N",
        help="number of epochs to train (default: 14)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default:1.0)"
    )
    
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--valid', type=str, default=os.environ['SM_CHANNEL_VALID'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    
    args=parser.parse_args()
    
    main(args)
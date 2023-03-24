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
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True #disable image truncated error


def test(model, test_loader, criterion, device):
    
    '''
    Tests the model on testing data
    
    Args:
        - model: Pytorch model
        - test_loader: DataLoader object of the test data
        - criterion: Loss function
        - device: cpu/gpu
    '''
    
    model.eval()
    
    running_loss = 0
    running_corrects = 0
    

    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")

def train(model, train_loader, valid_loader, criterion, optimizer, device):

    '''
    Trains the pre-trained model on training data and validates on validation data over given number of epochs. 
    
    Args:
        - model: Pytorch model
        - train_loader: DataLoader object of the train data
        - valid_loader: DataLoader object of the valid data
        - criterion: Loss function
        - optimizer: Optimizer of the model
        - device: cpu/gpu
    '''
        
    image_dataset = {'train':train_loader, 'valid':valid_loader}
    loss_counter = 0
    best_loss=1e6
    
    for epoch in range(1, args.epochs+1):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for batch_idx, (inputs, labels) in enumerate(image_dataset[phase]):
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples += len(inputs)
                
                if batch_idx %100 == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                if running_samples>(0.2*len(image_dataset[phase].dataset)):
                    break
                
            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples

            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
    return model
      
    
def net():
    '''
    Initializes the pre-trained model
    
    Args:
    
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 252),
        nn.Linear(252, 133)
    )
    
    print("Model loaded")
    
    return model

def create_data_loaders(data, batch_size):
    '''
    Loads and transforms the data and creates `DataLoader` object
    
    Args:
        - data: dataset
        - batch_size: batch size
        
    Return:
        - data_loader: DataLoader object 
        
    '''
    
    transform = transforms.Compose([
        transforms.Resize(size=(300,400)), 
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,)
        )
    ])
    
    data_image = ImageFolder(root=data, transform=transform)
    data_loader = torch.utils.data.DataLoader(data_image, batch_size=batch_size, shuffle=True)
    
    print("Loaded", data)
    
    return data_loader
    

def main(args):
    
    # initialized the model
    model=net()
    
    # define the loss function and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
         
    # setup the device type
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # get the batch_size from the args
    batch_size=args.batch_size
    
    # load the data
    train_loader=create_data_loaders(args.train, batch_size)
    valid_loader=create_data_loaders(args.valid, batch_size)
    test_loader=create_data_loaders(args.test, batch_size)  

    # train and test the model
    train(model, train_loader, valid_loader, loss_criterion, optimizer, device)
    test(model, test_loader, loss_criterion, device)
    
    # save the model
    with open(os.path.join(args.model_dir, 'dog_bread_model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)

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
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 64)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
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

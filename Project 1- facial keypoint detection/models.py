## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 
        # 5x5 square convolution kernel
        ## output size after pooling layer = 224/2=112
        # the output Tensor for one image, will have the dimensions: (32, 112, 112)


        self.conv1 = nn.Conv2d(1, 32, 5, padding=(2,2))
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.1)
        
        
        # 32 input image channel (grayscale), 64 output channels/feature maps, 
        # 5x5 square convolution kernel
        ## output size after pooling layer = 112/2=56
        # the output Tensor for one image, will have the dimensions: (64, 56, 56)
        
        self.conv2 = nn.Conv2d(32, 64, 5, padding=(2,2))
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p=0.1)
        
        # 64 input image channel (grayscale), 128 output channels/feature maps, 
        # 5x5 square convolution kernel
        ## output size after pooling layer = 56/2=28
        # the output Tensor for one image, will have the dimensions: (128, 28, 28)
        
        self.conv3 = nn.Conv2d(64, 128, 5, padding=(2,2))
        self.norm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(p=0.1)
        
        
        # 128 input image channel (grayscale), 256 output channels/feature maps, 
        # 5x5 square convolution kernel
        ## output size after pooling layer = 28/2=14
        # the output Tensor for one image, will have the dimensions: (256, 14, 14)
        
        self.conv4 = nn.Conv2d(128, 256, 5, padding=(2,2))
        self.norm4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(p=0.1)
        
        
        # 256 input image channel (grayscale), 512 output channels/feature maps, 
        # 5x5 square convolution kernel
        ## output size after pooling layer = 14/2=7
        # the output Tensor for one image, will have the dimensions: (512, 7, 7)
        
        self.conv5 = nn.Conv2d(256, 512, 5, padding=(2,2))
        self.norm5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.drop5 = nn.Dropout(p=0.1)
        
        # 512 outputs * the 7*7 filtered/pooled map size

        self.fc1 = nn.Linear(512*7*7, 2048)
        self.fc1_drop = nn.Dropout(p=0.1)
        self.fc1_norm = nn.BatchNorm1d(2048)
      
        
        # finally, create 136 output channels (for the 136 classes)
        self.fc2 = nn.Linear(2048, 136)
        
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        # a modified x, having gone through all the layers of your model, should be returned
        
        x = self.pool1(self.norm1(F.relu(self.conv1(x))))
        x = self.drop1(x)

        x = self.pool2(self.norm2(F.relu(self.conv2(x))))
        x = self.drop2(x)

        x = self.pool3(self.norm3(F.relu(self.conv3(x))))
        x = self.drop3(x)


        x = self.pool4(self.norm4(F.relu(self.conv4(x))))
        x = self.drop4(x)
        
        x = self.pool5(self.norm5(F.relu(self.conv5(x))))
        x = self.drop5(x)

        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1_norm(self.fc1(x)))
        x = self.fc1_drop(x)
    
        x = self.fc2(x)

        
        # final output
        return x
        
       

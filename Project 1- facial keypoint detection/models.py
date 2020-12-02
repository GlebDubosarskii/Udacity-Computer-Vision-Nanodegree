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
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
        # 1 input image channel (grayscale), 64 output channels/feature maps, 5x5 conv
        # 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (64, 220, 220)
        # after one pool layer, this becomes (64, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.2)
        
        # second conv layer: 64 inputs, 64 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (64, 106, 106)
        # after another pool layer this becomes (64, 53, 53) 
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p=0.2)
        
        # third conv layer: 64 inputs, 64 outputs, 4x4 conv
        ## output size = (W-F)/S +1 = (53-4)/1 +1 = 50
        # the output tensor will have dimensions: (64, 50, 50)
        # after another pool layer this becomes (64, 25, 25) 
        self.conv3 = nn.Conv2d(64, 128, 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout(p=0.2)
        
        # forth conv layer: 64 inputs, 64 outputs, 4x4 conv
        ## output size = (W-F)/S +1 = (25-4)/1 +1 = 22
        # the output tensor will have dimensions: (64, 22, 22)
        # after another pool layer this becomes (64, 11, 11) 
        self.conv4 = nn.Conv2d(128, 128, 4)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout(p=0.2)
        
        # fifth conv layer: 64 inputs, 64 outputs, 4x4 conv
        # output size = (W-F)/S +1 = (11-4)/1 +1 = 8
        # the output tensor will have dimensions: (64, 8, 8)
        # after another pool layer this becomes (64, 4, 4) 
        self.conv5 = nn.Conv2d(128, 128, 4)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.drop5 = nn.Dropout(p=0.2)
        
        # 64 outputs * the 4*4 filtered/pooled map size
        self.fc1 = nn.Linear(128*4*4, 512)
        #self.fc1 = nn.Linear(64*4*4, 128)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.2)
        
        # finally, create 136 output channels (for the 136 classes)
        self.fc2 = nn.Linear(512, 136)
        
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        
        # two conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.drop4(x)
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.drop5(x)

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        x = nn.functional.tanh(x)
        
        # final output
        return x.type(torch.FloatTensor)
        
        
        
        return x

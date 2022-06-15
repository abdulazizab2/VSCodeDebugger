"""
    This tutorial will go over intermediate-level debugging by using a debugger configuration file
    1- Adding a debugging configuration
    2- Going inside the configurations
    3- Go over behind the scenes in PyTorch
"""

# step1: Execute the code. Should receive a runtime error to move to next step
# step2: Start debugging. What can you notice?



import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_size', required=True, type=int)
parser.add_argument('--hidden_size', required=True, type=int)


class LinearRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=hidden_size ,out_features=1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    args = parser.parse_args()
    test_input = torch.randn(1, args.input_size)
    net = LinearRegression(input_size=100, hidden_size=args.hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    
    x = torch.randn(8, args.input_size) # samples = 8, 100 dimension input
    ground_truth = torch.randn(8) # 8 labels (regression value)
    batch_size = 4
    train_steps_per_epoch = x.size(0) // batch_size
    
    for epoch in range(100):
        running_loss = 0
        for i in range(train_steps_per_epoch):
            inputs = x[i*batch_size : (i+1)*batch_size]

            labels = ground_truth[i*batch_size : (i+1)*batch_size]
            labels = labels.sum()# Shape mismatch
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Loss @epoch{epoch}: {loss:.4f}')
    print('Finished Training')
    with torch.no_grad():
        print(net(test_input)) 
            
    
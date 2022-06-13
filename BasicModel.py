"""
    This tutorial will cover a simple model deubgging:
    1- To check what is inside a debugging environment
"""

# step1: Skim through the code and imports
# step2: Execute the code successfully
# step3: Start debugging: F5 or Run -> Debug
# Question: When and where does a debugger stop? ()
# step4: Explore the debugging toolbar
 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(in_features=input_size, out_features=5)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=5, out_features=1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    test_input = torch.randn(1, 100)
    net = LinearRegression(input_size=100)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    
    x = torch.randn(8, 100) # samples = 8, 100 dimension input
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
            
        # print(f'Loss @epoch{epoch}: {loss:.4f}')
    print('Finished Training')
    with torch.no_grad():
        print(net(test_input)) 
            
    
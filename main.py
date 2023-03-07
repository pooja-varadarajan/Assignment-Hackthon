import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)
def c2f(x):
    return 1.8*x + 32

# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        return torch.tanh(self.fc1(x))

# Initialize the model and optimizer
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

# Define the loss function
criterion = nn.MSELoss()

# Generate some random data
inputs = torch.randn(100, 1)
labels = torch.from_numpy(np.vectorize(c2f)(inputs))
# Train the model
for i in range(500):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print("Iteration {}, Loss: {}".format(i, loss.item()))

# Evaluate the model
with torch.no_grad():
    # Generate some random data
    test_inputs = torch.randn(100, 1)
    test_labels = torch.from_numpy(np.vectorize(c2f)(test_inputs))
    test_outputs = net(test_inputs)
    test_loss = criterion(test_outputs, test_labels)
    print("Test Loss: {}".format(test_loss.item()))

assert c2f(0) == net(torch.tensor(0, dtype=torch.float32)).item(), Exception("Model didn't learn the parameters")

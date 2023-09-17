import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

from DegreeDropout import DegreeDropout

# Make PyTorch use MPS
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    # torchcudo.set_device(mps_device)
else:
    print ("MPS device not found.")



class FullyConnectedMNIST(nn.Module):
    def __init__(self, dropout_probability):
        super(FullyConnectedMNIST, self).__init__()
        overfitting_multiplier = 3
        self.fc1 = nn.Linear(28 * 28, 1024 * overfitting_multiplier)
        self.fc2 = nn.Linear(1024 * overfitting_multiplier, 1024 * overfitting_multiplier)
        self.fc3 = nn.Linear(1024 * overfitting_multiplier, 1024* overfitting_multiplier)
        self.fc4 = nn.Linear(1024 * overfitting_multiplier, 1024 * overfitting_multiplier)
        self.fc5 = nn.Linear(1024 * overfitting_multiplier, 10)
        self.dropout = DegreeDropout(p=dropout_probability)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x, self.fc2.weight)
        x = F.relu(self.fc2(x))
        x = self.dropout(x, self.fc3.weight)
        x = F.relu(self.fc3(x))
        x = self.dropout(x, self.fc4.weight)
        x = F.relu(self.fc4(x))
        x = self.dropout(x, self.fc5.weight)
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


# Training parameters
batch_size = 64
epochs = 10
lr = 0.01

# Load the dataset
indices_train = list(range(1024))
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    ),
    batch_size=batch_size,
    # shuffle=True,
    sampler=SubsetRandomSampler(indices_train)
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
)

# Train the model
def train(epoch, model, optimizer):
    model.train()
    model.parameters()
    tqdm_data = tqdm(train_loader)
    loss_sum = 0
    n_iterations = 0
    for batch_idx, (data, target) in enumerate(tqdm_data):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction="mean")
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            tqdm_data.set_postfix({f"Loss": loss.item()}, refresh=True)
        n_iterations += 1

    return loss_sum / n_iterations

def validate(model):
    model.eval()
    validation_accuracy = 0
    correct = 0
    with torch.no_grad():
        n_iterations = 0
        for data, target in test_loader:
            output = model(data)
            validation_accuracy += F.nll_loss(output, target, reduction="mean").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            n_iterations += 1

    validation_accuracy /= n_iterations

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            validation_accuracy,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    return validation_accuracy


def train_and_validate(n_train, dropout_probability):
    model = FullyConnectedMNIST(dropout_probability)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train_losses = []
    validation_losses = []
    for epoch in range(1, n_train + 1):
        train_loss = train(epoch, model, optimizer)
        validation_loss = validate(model)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        if epoch % 10 == 0:
            plt.plot(train_losses, label="Train loss")
            plt.plot(validation_losses, label="Validation loss")
            plt.legend()
            plt.show()
    return validation_loss


train_and_validate(n_train=100, dropout_probability=0)

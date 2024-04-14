""" Annotated torch code for the MNIST portion of cs637hw3 """
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pickle as pkl

from models import Net, SaveOutput

def train(args, model, device, train_loader, optimizer, epoch,
          save_per_epoch=False):
    """
    1. Unpack the next batch's index and the feature/label values
    2. Buffer the batch data on the requested device
    3. Reset the gradients from the previous batch
    4. Calculate negative log likelihood loss
    5. Back-propagate the gradients
    6. Call the optimizer
    7. Log progress at a user-defined interval
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        ## negative log-likelihood loss
        loss = F.nll_loss(output, target)
        ## backward pass
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    """
    1. Set evaluation mode to disable dropout, and batchnorm moving average
    2. In a no-gradient context load the test data to the device
    3. Accumulate test loss over all of the batches, and number correct
    4. Print evaluation metrics
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Avg loss: {:.4f}, Acc: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    """
    1. parse arguments
    2. establish a random seed
    3. determine GPU/CPU device
    4. make a transform to normalize data
    5. Load training and testing MNIST dataset with transform and batch size
    6. Initialize the above CNN on the selected device
    7. Use Adadelta to optimize
    """
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument(
            '--batch-size', type=int, default=64, metavar='N',
            help='input batch size for training (default: 64)')
    parser.add_argument(
            '--test-batch-size', type=int, default=1000, metavar='N',
            help='input batch size for testing (default: 1000)')
    parser.add_argument(
            '--epochs', type=int, default=14, metavar='N',
            help='number of epochs to train (default: 14)')
    parser.add_argument(
            '--lr', type=float, default=1.0, metavar='LR',
            help='learning rate (default: 1.0)')
    parser.add_argument(
            '--gamma', type=float, default=0.7, metavar='M',
            help='Learning rate step gamma (default: 0.7)')
    parser.add_argument(
            '--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument(
            '--no-mps', action='store_true', default=False,
            help='disables macOS GPU training')
    parser.add_argument(
            '--dry-run', action='store_true', default=False,
            help='quickly check a single pass')
    parser.add_argument(
            '--seed', type=int, default=20000722, metavar='S',
            help='random seed (default: 20000722)')
    parser.add_argument(
            '--log-interval', type=int, default=10, metavar='N',
            help='batches to wait between logging entries')
    parser.add_argument(
            '--save-model', action='store_true', default=False,
            help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net()
    #out_hook = SaveOutput() ## class for saving output states
    #model.conv1.register_forward_hook(out_hook)
    model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        torch.save(model.state_dict(),
                   f'data/models/mnist/model_mnist_{epoch:02}.pt')

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()

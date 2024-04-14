import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imageio
from torchvision import datasets
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
from matplotlib.colors import hsv_to_rgb

from models import CNNNet,Net

cifar_labels = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')

def generate_raw_image(RGB:np.ndarray, image_path:Path, gif:bool=False,
                       fps:int=5):
    """
    Use imageio to write a raw full-resolution image to image_path
    :param RGB: (H, W, 3) array of RGB values to write as an image, or
            (H, W, T, 3) array of RGB values to write as a gif, if the gif
            attributeis True.
    :param image_path: Path of image file to write
    :param gif: if True, attempts to write as a full-resolution gif
    """
    if not gif:
        imageio.imwrite(image_path.as_posix(), RGB)
        return image_path
    RGB = np.moveaxis(RGB, 2, 0)
    imageio.mimwrite(uri=image_path, ims=RGB, format=".gif", fps=fps)
    return image_path

def scal_to_rgb(X:np.ndarray, hue_range:tuple=(0,.6), sat_range:tuple=(1,1),
                val_range:tuple=(1,1), normalize=True):
    """
    Convert a 2d array of data values to a [0,1]-normalized RGB using an hsv
    reference system. For a basic color scale, just change the hue parameter.

    Data values must be binned to 256 in order to convert,
    so data integrity is compromised by this method.
    """
    assert len(X.shape)==2
    for r in (hue_range, sat_range, val_range):
        i,f = r
        if not 0<=i<=1 and 0<=f<=1:
            raise ValueError(f"All bounds must be between 0 and 1 ({i},{f})")
    if normalize:
        X  = (X-np.amin(X))/np.ptp(X)
    to_interval = lambda X, interval: X*(interval[1]-interval[0])+interval[0]
    hsv = np.dstack([to_interval(X,intv) for intv in
                     (hue_range, sat_range, val_range)])
    return np.asarray(hsv_to_rgb(hsv)*255).astype(np.uint8)

def eval_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    loader = torch.utils.data.DataLoader(datasets.MNIST(
        './data', train=True, download=True, transform=transform))
    next_8 = lambda: next(x for x,y in loader if y==8)
    next_8() ## skip the first one
    sample = next_8()
    generate_raw_image(scal_to_rgb(np.squeeze(sample.detach().numpy())),
                       Path(f"figures/activations/mnist_sample_8.png"))
    model = Net()
    state_dicts = list(map(
        torch.load, sorted(Path("data/models/mnist").iterdir())
        ))
    for i,sd in enumerate(state_dicts):
        model.load_state_dict(sd)
        model.eval()
        model(sample)
        if i==0:
            print(f"MNIST L1: {model.conv1_out[0,0].shape}")
            print(f"MNIST L2: {model.conv2_out[0,0].shape}")
        for j in range(model.conv1_out.shape[1]):
            generate_raw_image(
                    scal_to_rgb(
                        model.conv1_out[0,j].detach().numpy(),
                        sat_range=(.7,1.),
                        val_range=(1.,.7),
                        ),
                    Path(f"figures/activations/mnist_L1-E{i:02}-K{j:02}.png")
                    )
        for j in range(model.conv2_out.shape[1]):
            generate_raw_image(
                    scal_to_rgb(
                        model.conv2_out[0,j].detach().numpy(),
                        sat_range=(.7,1.),
                        val_range=(1.,.7),
                        ),
                    Path(f"figures/activations/mnist_L2-E{i:02}-K{j:02}.png")
                    )

def eval_cifar10():
    ## Define a training data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    loader = torch.utils.data.DataLoader(datasets.CIFAR10(
        './data', train=True, download=True, transform=transform))

    ## Function to find the next plane image
    next_plane = lambda: next(
            x for x,y in loader
            if y==cifar_labels.index("plane")
            )
    next_plane() ## skip the first one
    sample = next_plane()

    ## Generate an RGB of the input data
    rgb = np.squeeze(sample.detach().numpy()*255).transpose((1,2,0))
    generate_raw_image(
            rgb.astype(np.uint8),
            Path(f"figures/activations/mnist_sample_plane.png")
            )


    ## Initialize a model and load the weights from each epoch
    model = CNNNet()
    state_dicts = list(map(
        torch.load, sorted(Path("data/models/cifar10").iterdir())))
    for i,sd in enumerate(state_dicts):
        model.load_state_dict(sd)
        model.eval()
        model(sample) ##
        if i==0:
            print(f"CIFAR10 L1: {model.conv1_out[0,0].shape}")
            print(f"CIFAR10 L2: {model.conv2_out[0,0].shape}")
            print(f"CIFAR10 L2: {model.conv3_out[0,0].shape}")
        for j in range(model.conv1_out.shape[1]):
            generate_raw_image(
                    scal_to_rgb(
                        model.conv1_out[0,j].detach().numpy(),
                        sat_range=(.7,1.)),
                    Path(f"figures/activations/cifar10_L1-E{i:02}-K{j:02}.png")
                    )
        for j in range(model.conv2_out.shape[1]):
            generate_raw_image(
                    scal_to_rgb(
                        model.conv2_out[0,j].detach().numpy(),
                        sat_range=(.7,1.)),
                    Path(f"figures/activations/cifar10_L2-E{i:02}-K{j:02}.png")
                    )
        for j in range(model.conv3_out.shape[1]):
            generate_raw_image(
                    scal_to_rgb(
                        model.conv3_out[0,j].detach().numpy(),
                        sat_range=(.7,1.)),
                    Path(f"figures/activations/cifar10_L3-E{i:02}-K{j:02}.png")
                    )

if __name__=="__main__":
    eval_cifar10()
    eval_mnist()

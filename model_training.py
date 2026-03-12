from torchvision import datasets
from torchvision.transforms import ToTensor, PILToTensor, Normalize, Compose, ConvertImageDtype

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch import nn
import torch

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def get_data(transform, val_split = 0.0):
    """
    Load train and test datasets with the specified transform applied. ts docstring generated with claude. If `val_split` is provided, it takes that split out of the test set and returns a validation set.
    Args:
        transform (str): The preprocessing transform to apply to the data.
            - ``"none"``: Raw integer pixel intensities in [0, 255].
            - ``"normalize"``: Pixel values scaled to [0, 1].
            - ``"standardize"``: Values scaled to mean 0 and std 1 per channel,
              computed from the training set.
        val_split (float, optional): Fraction of training data to reserve as a
            validation set. Must be in [0, 1). If 0, no validation set is
            created. Defaults to 0.0.
    Returns:
        tuple[Dataset, ...]: A tuple of datasets. Returns
            ``(train_data, val_data, test_data)`` if ``val_split > 0``,
            otherwise ``(train_data, test_data)``.
    """
    if transform == "none":
        print("No transform selected. Data as integers in range [0, 255]\n")
        transform_stack = Compose([PILToTensor(), ConvertImageDtype(torch.float32)])
    
    elif transform == "normalize":
        print("Data normalized to range [0, 1]\n")
        transform_stack = ToTensor()
        
    elif transform == "standardize":
        means = torch.tensor([0.4914, 0.4822, 0.4465])
        stds = torch.tensor([0.2470, 0.2435, 0.2616]) # both are precomputed

        transform_stack = Compose([ToTensor(), Normalize(means, stds)])
        print(f"Data standardized with means {means} and stds {stds}\n")

    else:
        raise ValueError("Invalid Preprocessing type")

    train_data = datasets.CIFAR10("data", train=True, download=True, transform=transform_stack)
    test_data = datasets.CIFAR10("data", train=False, download=True, transform=transform_stack)
    
    if val_split > 0:
        test_data, val_data = random_split(test_data, [1-val_split, val_split])
        return train_data, val_data, test_data
    else:
        return train_data, test_data

# eval function
def model_eval(model, device, val_loader):
    model.eval()
    
    losses = []

    for batch in val_loader:
        samples, targets = batch
        samples = samples.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            _, loss = model(samples, targets)

        losses.append(loss.item())

    model.train()

    return losses

def training_loop(model, device, train_loader, num_epochs, lr, eval_interval = 10, /, val_loader=None, optim="SGD"):
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optim == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    eval_points = []

    avg = lambda x : sum(x)/len(x)

    model = model.to(device)

    # main training loop
    for epoch in range(num_epochs):
        epoch_train_losses = []
        epoch_test_losses = []

        running_train_losses = []
        running_test_losses = []

        for i, batch in enumerate(train_loader):
            samples, targets = batch
            samples = samples.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            _, loss = model(samples, targets)

            loss.backward()
            optimizer.step()

            running_train_losses.append(loss.item())

            if (i+1) % eval_interval == 0 or (i+1) == len(train_loader): #eval point
                avg_train_loss = avg(running_train_losses)
                epoch_train_losses.append(avg_train_loss)
                eval_points.append((epoch*len(train_loader)) + (i+1))
                
                if val_loader is not None: 
                    running_test_losses = model_eval(model, device, val_loader)
                    avg_test_loss = avg(running_test_losses)
                    epoch_test_losses.append(avg_test_loss)
                    print(f"Batch {i+1}/{len(train_loader)}: train loss: {avg_train_loss}, test_loss: {avg_test_loss}")
                else:
                    print(f"Batch {i+1}/{len(train_loader)}: train loss: {avg_train_loss}")

                running_train_losses = []
                running_test_losses = []
        
        print("\n"+"-"*100)
        print(f"Epoch {epoch+1} complete.")
        print(f"Avg train loss: {avg(epoch_train_losses)}")
        
        if val_loader is not None:
            print(f"Avg test loss: {avg(epoch_test_losses)}")
            print(f"Best test loss: {min(epoch_test_losses)}")

        print("-"*100+"\n")

        train_losses.extend(epoch_train_losses)
        
        if val_loader is not None:
            test_losses.extend(epoch_test_losses)

    return eval_points, train_losses, test_losses

# plot losses on a graph
def plot_losses(eval_points, train_losses, test_losses, num_batches, epochs, save_loss_graph, filename):
    fig, axes = plt.subplots()

    axes.plot(eval_points, train_losses, label="Train Loss")
    if len(test_losses) > 0:
        axes.plot(eval_points, test_losses, label="Test Loss")

    axes.legend()

    axes.set_ylabel("Loss (Cross Entropy)")

    axes.set_xlabel("Epoch")
    axes.set_xticks(range(0, (num_batches*epochs)+1, num_batches))
    axes.set_xticklabels([f"{i}" for i in range(epochs+1)])

    axes.set_title("Loss over time")

    plt.show()

    if save_loss_graph:
        filedir = Path(".") / "losses"
        if not filedir.exists(): filedir.mkdir()
        
        file_path = filedir / f"{filename}.png"

        fig.savefig(file_path)
        print(f"Saved loss graph to {str(file_path)}")

def train(model, train_data, batch_size, num_epochs, lr, /, val_data=None, eval_interval=10, plot_loss=True, save_loss_graph=True, save_model = True, optimizer="SGD"):
    """
    Build data loaders, trains the model, and plots results.

    Args:
        model (nn.Module): The model to train. Must accept
            ``(samples, targets)`` and return ``(output, loss)``.
        train_data (Dataset): Training dataset.
        batch_size (int): Mini-batch size for both train and validation
            loaders.
        num_epochs (int): Number of full passes over the training set.
        lr (float): Learning rate passed to the optimizer.
        val_data (Dataset | None, optional): Validation dataset. If ``None``,
            no validation evaluation is performed. Defaults to ``None``.
        eval_interval (int, optional): Evaluate every this many batches.
            Defaults to 10.
        plot_loss (bool, optional): If ``True``, display and optionally save a
            loss curve after training. Defaults to ``True``.
        save_loss_graph (bool, optional): If ``True`` (and *plot_loss* is also
            ``True``), persist the loss plot as a PNG. Defaults to ``True``.
        save_model (bool, optional): If ``True``, saves the model in `weights` subdirectory. Defaults to ``True``.
        optimizer (str, optional): Optimizer name — ``"SGD"`` or ``"AdamW"``.
            Defaults to ``"SGD"``.
    """
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if val_data is None:
        val_loader = None
    else:
        val_loader = DataLoader(val_data, batch_size=batch_size)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Starting training on {device}")
    print()

    if val_loader is not None:
        eval_points, train_losses, test_losses = training_loop(model, device, train_loader, num_epochs, lr, eval_interval, val_loader=val_loader, optim=optimizer)
    else:
        eval_points, train_losses, test_losses = training_loop(model, device, train_loader, num_epochs, lr, eval_interval, optim=optimizer)

    model_name = type(model).__name__
    
    filename = f"{model_name}_{datetime.now().strftime("%y-%m-%d_%H-%M-%S")}"

    if plot_loss:
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)
        eval_points = np.array(eval_points)

        plot_losses(eval_points, train_losses, test_losses, len(train_loader), num_epochs, save_loss_graph, filename)

    if save_model:
        weights_dir = Path(".") / "weights"
        if not weights_dir.exists(): weights_dir.mkdir()

        weights_file = weights_dir / f"{filename}.pth"

        torch.save(model.state_dict(), weights_file)

        print(f"Saved model weights to {weights_file}")

    
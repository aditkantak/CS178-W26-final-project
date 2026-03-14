from torchvision import datasets
from torchvision.transforms import ToTensor, PILToTensor, Normalize, Compose, ConvertImageDtype, RandomCrop

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch import nn
import torch

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import redirect_stdout

def get_data(transform, random_crop=False, val_split = 0.0):
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
    transforms = []

    if random_crop:
        print(f"Random Crop will be used with padding=4")
        transforms.append(RandomCrop(32, padding=4))

    if transform == "none":
        print("No transform selected. Data as integers in range [0, 255]\n")
        transforms.extend([PILToTensor(), ConvertImageDtype(torch.float32)])
    
    elif transform == "normalize":
        print("Data normalized to range [0, 1]\n")
        transforms.extend([ToTensor()])
        
    elif transform == "standardize":
        means = torch.tensor([0.4914, 0.4822, 0.4465])
        stds = torch.tensor([0.2470, 0.2435, 0.2616]) # both are precomputed

        transforms.extend([ToTensor(), Normalize(means, stds)])
        print(f"Data standardized with means {means} and stds {stds}\n")

    else:
        raise ValueError("Invalid Preprocessing type")

    train_transform_stack = Compose(transforms)
    if type(transforms[0]) == RandomCrop:
        print("No random crop in test set")
        test_transform_stack = Compose(transforms[1:])
    else:
        test_transform_stack = train_transform_stack

    train_data = datasets.CIFAR10("data", train=True, download=True, transform=train_transform_stack)
    test_data = datasets.CIFAR10("data", train=False, download=True, transform=test_transform_stack)
    
    if val_split > 0:
        test_data, val_data = random_split(test_data, [1-val_split, val_split])
        return train_data, val_data, test_data
    else:
        return train_data, test_data

# eval function
def model_eval(model, device, val_loader):
    model.eval()
    
    losses = []

    with torch.no_grad():
        for batch in val_loader:
            samples, targets = batch
            samples = samples.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                _, loss = model(samples, targets)

            losses.append(loss.item())

    model.train()

    return losses

def training_loop(model, device, train_loader, num_epochs, optimizer, lr_scheduler, eval_interval = 10, /, val_loader=None):
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
        print(f"LR: {lr_scheduler.get_last_lr()[0]}")
        print(f"Avg train loss: {avg(epoch_train_losses)}")
        
        if val_loader is not None:
            print(f"Avg test loss: {avg(epoch_test_losses)}")
            print(f"Best test loss: {min(epoch_test_losses)}")

        print("-"*100+"\n")

        train_losses.extend(epoch_train_losses)
        
        if val_loader is not None:
            test_losses.extend(epoch_test_losses)

        lr_scheduler.step()

    return eval_points, train_losses, test_losses

# plot losses on a graph
def plot_losses(eval_points, train_losses, test_losses, num_batches, epochs):
    fig, axes = plt.subplots()

    axes.plot(eval_points, train_losses, label="Train Loss")
    if len(test_losses) > 0:
        axes.plot(eval_points, test_losses, label="Test Loss")

    axes.legend()

    axes.set_ylabel("Loss (Cross Entropy)")
    y_max = max(np.percentile(train_losses, 99), np.percentile(test_losses, 99))*1.1
    axes.set_ylim(0, y_max)

    axes.set_xlabel("Epoch")

    # tick_interval = 5 if epochs > 20 else 1
    axes.set_xticks(range(0, (num_batches*epochs)+1, num_batches))#*tick_interval))
    axes.set_xticklabels([f"{i}" for i in range(0, epochs+1)])#, tick_interval)])

    axes.set_title("Loss over time")

    return fig, axes

def train(model, train_data, batch_size, num_epochs, lr, /, val_data=None, eval_interval=10, optim="SGD", lr_decay_factor=1, l2_reg_strength=0.1, plot_loss=True, save_loss_graph=True, save_model = True):
    print()
    print(f"Training {num_epochs} epochs, starting with learning rate {lr}, with {optim} optimizer, with gamma {lr_decay_factor}")
    
    #configuring train loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    #configuring val loader
    if val_data is None:
        val_loader = None
    else:
        val_loader = DataLoader(val_data, batch_size=batch_size)

    # configuring optimizer
    if optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg_strength)
    elif optim == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg_strength)

    # configuring learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay_factor)

    # configuring device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get start time
    start_dt = datetime.now(ZoneInfo("America/Los_Angeles"))
    start_timestamp = start_dt.strftime("%m/%d/%Y %I:%M:%S %p")

    model_name = type(model).__name__
    filedir_name = f"{model_name}_{start_dt.strftime("%y-%m-%d_%H-%M-%S")}"
    filedir = Path(".") / "runs" /filedir_name
    if not filedir.exists(): filedir.mkdir()

    log_file = filedir / "training_log.txt"

    with open(log_file, "w", encoding="utf-8", buffering=1) as file:
        with redirect_stdout(file):
            print(f"Model type: {model_name}")
            print(f"Epochs: {num_epochs}")
            print(f"Starting LR: {lr}")
            if lr_decay_factor < 1: print(f"Gamma: {lr_decay_factor}")
            print(f"Optimizer: {optim}")
            print()
            print(f"Starting training on {device} at {start_timestamp}.")
            print()

            # train model and get points for graph
            eval_points, train_losses, test_losses = training_loop(model, device, train_loader, num_epochs, optimizer, lr_scheduler, eval_interval, val_loader=val_loader)

            # calculate elapsed time
            end_dt = datetime.now(ZoneInfo("America/Los_Angeles"))
            end_timestamp = end_dt.strftime("%m/%d/%Y %I:%M:%S %p")
            print(f"Training ended at {end_timestamp}.")
            print(f"Training elapsed time (hh:mm:ss): {str(end_dt-start_dt)}\n")

    print()
    print(f"Saved training log saved to {str(log_file)}")

    # loss curve graph
    if plot_loss:
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)
        eval_points = np.array(eval_points)

        fig, axes = plot_losses(eval_points, train_losses, test_losses, len(train_loader), num_epochs)

        axes.set_title(f"{model_name}, lr={lr}, gamma={lr_decay_factor}, {num_epochs} epochs")

        plt.show()

        if save_loss_graph:
            filedir = Path(".") / filedir_name
            if not filedir.exists(): filedir.mkdir()
            
            file_path = filedir / f"loss_curve.png"

            fig.savefig(file_path)
            print(f"Saved loss graph to {str(file_path)}")

    # saves model weights
    if save_model:
        filedir = Path(".") / filedir_name
        if not filedir.exists(): filedir.mkdir()

        weights_file = filedir / f"weights.pth"

        torch.save(model.state_dict(), weights_file)

        print(f"Saved model weights to {weights_file}")

def get_predictions(model, dataset, k=0):
    test_dl = DataLoader(dataset, 1)

    all_targets = []
    all_preds = []

    model.eval()

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in test_dl:
            sample, target = batch

            sample = sample.to(device)
            target = target.to(device)

            pred = model.predict(sample, k)
            
            all_targets.append(target.item())
            all_preds.append(pred.item())

    return all_preds, all_targets

    
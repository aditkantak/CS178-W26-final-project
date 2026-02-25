from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data.dataloader import DataLoader
from torch import nn
import torch

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def get_data():
    train_data = datasets.CIFAR10("data", train=True, download=True, transform=ToTensor())
    test_data = datasets.CIFAR10("data", train=False, download=True, transform=ToTensor())

    return train_data, test_data

def get_loaders(train_data, test_data, batch_size):
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size)

    return train_loader, test_loader

# eval function
def model_eval(model, device, test_loader):
    model.eval()
    
    losses = []

    for batch in test_loader:
        samples, targets = batch
        samples = samples.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            _, loss = model(samples, targets)

        losses.append(loss.item())

    model.train()

    return losses

def training_loop(model, device, train_loader, test_loader, num_epochs, lr, eval_interval = 10, optim="SGD"):
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
                running_test_losses = model_eval(model, device, test_loader)
                
                avg_train_loss = avg(running_train_losses)
                avg_test_loss = avg(running_test_losses)
                
                epoch_train_losses.append(avg_train_loss)
                epoch_test_losses.append(avg_test_loss)
                eval_points.append((epoch*len(train_loader)) + (i+1))

                print(f"Batch {i+1}/{len(train_loader)}: train loss: {avg_train_loss}, test_loss: {avg_test_loss}")

                running_train_losses = []
                running_test_losses = []
        
        print("\n"+"-"*100)
        print(f"Epoch {epoch+1} complete.")
        print(f"Avg train loss: {avg(epoch_train_losses)}")
        print(f"Avg test loss: {avg(epoch_test_losses)}")
        print(f"Best test loss: {min(epoch_test_losses)}")
        print("-"*100+"\n")

        train_losses.extend(epoch_train_losses)
        test_losses.extend(epoch_test_losses)

    return eval_points, train_losses, test_losses

# plot losses on a graph
def plot_losses(eval_points, train_losses, test_losses, num_batches, epochs, eval_interval, save_loss_graph, model_name):
    fig, axes = plt.subplots()

    axes.plot(eval_points, train_losses, label="Train Loss")
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
        
        file_path = filedir / f"{model_name}_{datetime.now().strftime("%y-%m-%d_%H-%M-%S")}.png"

        fig.savefig(file_path)
        print(f"Saved loss graph to {str(file_path)}")

def train(model, batch_size, num_epochs, lr, eval_interval=10, plot_loss=True, save_loss_graph=True, optimizer="SGD"):
    train_data, test_data = get_data()
    train_loader, test_loader = get_loaders(train_data, test_data, batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_points, train_losses, test_losses = training_loop(model, device, train_loader, test_loader, num_epochs, lr, eval_interval, optimizer)

    if plot_loss:
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)
        eval_points = np.array(eval_points)

        model_name = type(model).__name__

        plot_losses(eval_points, train_losses, test_losses, len(train_loader), num_epochs, eval_interval, save_loss_graph, model_name)
import copy
import random
from itertools import product
from pathlib import Path
import csv

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


SEED = 42
DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BATCH_SIZE = 128
DEFAULT_OPTIMIZER = "SGD"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_activation(name: str) -> nn.Module:
    activation_name = name.lower()

    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "tanh":
        return nn.Tanh()
    if activation_name == "sigmoid":
        return nn.Sigmoid()

    raise ValueError(f"Unsupported activation function: {name}")


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        activation: str,
        output_size: int = 10,
    ):
        super().__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(get_activation(activation))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        x = torch.flatten(x, start_dim=1)
        logits = self.network(x)

        if y is None:
            return logits, None

        loss = self.loss_fn(logits, y)
        return logits, loss

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self(x)
        return torch.argmax(logits, dim=1)


def get_datasets(seed: int = SEED):
    full_train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=ToTensor())
    test_data = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=ToTensor())

    train_size = 40000
    val_size = 10000

    generator = torch.Generator().manual_seed(seed)
    train_data, val_data = random_split(full_train, [train_size, val_size], generator=generator)
    return train_data, val_data, test_data


def get_loaders(train_data, val_data, test_data, batch_size: int = DEFAULT_BATCH_SIZE):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, y)
        preds = torch.argmax(logits, dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == y).sum().item()
        total_examples += batch_size

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    model.train()
    return avg_loss, accuracy


def build_optimizer(model: nn.Module, lr: float):
    if DEFAULT_OPTIMIZER == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if DEFAULT_OPTIMIZER == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    raise ValueError(f"Unsupported optimizer: {DEFAULT_OPTIMIZER}")



def train_one_configuration(
    config: dict,
    train_data,
    val_data,
    test_data,
    num_epochs: int = 10,
    seed: int = SEED,
):
    set_seed(seed)

    train_loader, val_loader, test_loader = get_loaders(
        train_data,
        val_data,
        test_data,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    model = FeedForwardNN(
        input_size=32 * 32 * 3,
        hidden_sizes=config["hidden_sizes"],
        activation=config["activation"],
        output_size=10,
    ).to(DEVICE)

    optimizer = build_optimizer(model, config["lr"])

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        model.train()

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = evaluate(model, train_loader, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, DEVICE)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | "
            f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
        )

        if (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, DEVICE)

    result = {
        "config": config,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history,
        "state_dict": best_state,
    }
    return result


def tune_model(search_space: dict, num_epochs: int = 10, seed: int = SEED):
    set_seed(seed)
    train_data, val_data, test_data = get_datasets(seed)

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    results = []
    best_result = None

    for i, combo in enumerate(product(*values), start=1):
        config = dict(zip(keys, combo))
        print("\n" + "=" * 100)
        print(f"Configuration {i}")
        print(config)
        print("=" * 100)

        result = train_one_configuration(
            config=config,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            num_epochs=num_epochs,
            seed=seed,
        )
        results.append(result)

        print(
            f"Best epoch: {result['best_epoch']} | "
            f"best val acc: {result['best_val_acc']:.4f} | "
            f"best val loss: {result['best_val_loss']:.4f} | "
            f"test acc: {result['test_acc']:.4f}"
        )

        if best_result is None:
            best_result = result
        else:
            better_val_acc = result["best_val_acc"] > best_result["best_val_acc"]
            tie_broken_by_loss = (
                result["best_val_acc"] == best_result["best_val_acc"]
                and result["best_val_loss"] < best_result["best_val_loss"]
            )
            if better_val_acc or tie_broken_by_loss:
                best_result = result

    return best_result, results


def save_results(best_result: dict, all_results: list[dict], output_dir: str = "ffnn_results"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(best_result["state_dict"], output_path / "best_ffnn_model.pt")

    ranked_results = sorted(
        all_results,
        key=lambda result: (-result["best_val_acc"], result["best_val_loss"]),
    )
    top_results = ranked_results[:5]

    summary_file = output_path / "tuning_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Feedforward Neural Network Hyperparameter Tuning Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Fixed batch size: {DEFAULT_BATCH_SIZE}\n")
        f.write(f"Fixed optimizer: {DEFAULT_OPTIMIZER}\n\n")

        for i, result in enumerate(all_results, start=1):
            f.write(f"Configuration {i}:\n")
            f.write(f"  config: {result['config']}\n")
            f.write(f"  best epoch: {result['best_epoch']}\n")
            f.write(f"  best validation accuracy: {result['best_val_acc']:.4f}\n")
            f.write(f"  best validation loss: {result['best_val_loss']:.4f}\n")
            f.write(f"  test accuracy: {result['test_acc']:.4f}\n")
            f.write(f"  test loss: {result['test_loss']:.4f}\n\n")

        f.write("Top 5 Configurations\n")
        f.write("-" * 70 + "\n")
        for rank, result in enumerate(top_results, start=1):
            f.write(f"Rank {rank}:\n")
            f.write(f"  config: {result['config']}\n")
            f.write(f"  best epoch: {result['best_epoch']}\n")
            f.write(f"  best validation accuracy: {result['best_val_acc']:.4f}\n")
            f.write(f"  best validation loss: {result['best_val_loss']:.4f}\n")
            f.write(f"  test accuracy: {result['test_acc']:.4f}\n")
            f.write(f"  test loss: {result['test_loss']:.4f}\n\n")

        f.write("Best Configuration\n")
        f.write("-" * 70 + "\n")
        f.write(f"config: {best_result['config']}\n")
        f.write(f"best epoch: {best_result['best_epoch']}\n")
        f.write(f"best validation accuracy: {best_result['best_val_acc']:.4f}\n")
        f.write(f"best validation loss: {best_result['best_val_loss']:.4f}\n")
        f.write(f"final test accuracy: {best_result['test_acc']:.4f}\n")
        f.write(f"final test loss: {best_result['test_loss']:.4f}\n")

    top5_table_file = output_path / "top5_results.csv"
    with open(top5_table_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "rank",
            "hidden_sizes",
            "learning_rate",
            "activation",
            "best_epoch",
            "best_val_acc",
            "best_val_loss",
            "test_acc",
            "test_loss",
        ])

        for rank, result in enumerate(top_results, start=1):
            config = result["config"]
            writer.writerow([
                rank,
                str(config["hidden_sizes"]),
                config["lr"],
                config["activation"],
                result["best_epoch"],
                f"{result['best_val_acc']:.4f}",
                f"{result['best_val_loss']:.4f}",
                f"{result['test_acc']:.4f}",
                f"{result['test_loss']:.4f}",
            ])

    print(f"\nSaved best model to {output_path / 'best_ffnn_model.pt'}")
    print(f"Saved tuning summary to {summary_file}")
    print(f"Saved top-5 results table to {top5_table_file}")


if __name__ == "__main__":
    search_space = {
        "hidden_sizes": [
            [128],
            [256],
            [128, 64],
            [256, 128],
            [256, 128, 64],
        ],
        "lr": [0.01, 0.005, 0.001],
        "activation": ["ReLU", "Tanh"],
    }

    NUM_EPOCHS = 30

    best_result, all_results = tune_model(search_space, num_epochs=NUM_EPOCHS, seed=SEED)

    print("\n" + "#" * 100)
    print("Best overall configuration")
    print(best_result["config"])
    print(f"Best epoch: {best_result['best_epoch']}")
    print(f"Validation accuracy: {best_result['best_val_acc']:.4f}")
    print(f"Validation loss: {best_result['best_val_loss']:.4f}")
    print(f"Test accuracy: {best_result['test_acc']:.4f}")
    print(f"Test loss: {best_result['test_loss']:.4f}")
    print("#" * 100)

    save_results(best_result, all_results)

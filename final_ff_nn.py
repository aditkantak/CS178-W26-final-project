import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


SEED = 42
DATA_DIR = "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128

BEST_HIDDEN_SIZES = [256, 128]
BEST_LR = 0.001
BEST_ACTIVATION = "ReLU"
NUM_EPOCHS = 75


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
    def __init__(self, input_size: int, hidden_sizes: list[int], activation: str, output_size: int = 10):
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


def get_loaders(train_data, val_data, test_data, batch_size: int = BATCH_SIZE):
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


def train_model(hidden_sizes, lr, activation, num_epochs=NUM_EPOCHS, seed: int = SEED):
    set_seed(seed)

    train_data, val_data, test_data = get_datasets(seed)
    train_loader, val_loader, test_loader = get_loaders(train_data, val_data, test_data)

    model = FeedForwardNN(
        input_size=32 * 32 * 3,
        hidden_sizes=hidden_sizes,
        activation=activation,
        output_size=10,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_state = None
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(DEVICE)
    test_loss, test_acc = evaluate(model, test_loader, DEVICE)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "state_dict": best_state,
    }


def save_run(results: dict, output_dir: str = "best_ffnn_run"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "best_ffnn_model.pt"
    torch.save(results["state_dict"], model_path)

    summary_path = output_path / "run_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Single FFNN Training Run\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"hidden_sizes: {BEST_HIDDEN_SIZES}\n")
        f.write(f"learning_rate: {BEST_LR}\n")
        f.write(f"activation: {BEST_ACTIVATION}\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
        f.write(f"optimizer: AdamW\n")
        f.write(f"epochs: {NUM_EPOCHS}\n\n")
        f.write(f"best_epoch: {results['best_epoch']}\n")
        f.write(f"best_validation_accuracy: {results['best_val_acc']:.4f}\n")
        f.write(f"best_validation_loss: {results['best_val_loss']:.4f}\n")
        f.write(f"test_accuracy: {results['test_acc']:.4f}\n")
        f.write(f"test_loss: {results['test_loss']:.4f}\n")

    print(f"\nSaved model to {model_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    results = train_model(
        hidden_sizes=BEST_HIDDEN_SIZES,
        lr=BEST_LR,
        activation=BEST_ACTIVATION,
        num_epochs=NUM_EPOCHS,
        seed=SEED,
    )

    print("\n" + "#" * 80)
    print("Final Results")
    print(f"hidden_sizes: {BEST_HIDDEN_SIZES}")
    print(f"learning_rate: {BEST_LR}")
    print(f"activation: {BEST_ACTIVATION}")
    print(f"epochs: {NUM_EPOCHS}")
    print(f"best epoch: {results['best_epoch']}")
    print(f"best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"best validation loss: {results['best_val_loss']:.4f}")
    print(f"test accuracy: {results['test_acc']:.4f}")
    print(f"test loss: {results['test_loss']:.4f}")
    print("#" * 80)

    save_run(results)

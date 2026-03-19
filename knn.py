import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from model_training import get_data, get_dataset_as_numpy

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Consistent colour palette
PALETTE = {
    "primary":   "#1E53C7",   # blue  – train / val
    "secondary": "#16A34A",   # green – test
    "accent":    "#C72222",   # red   – best-k marker / above-mean bars
    "mean_line": "#2563EB",   # blue  – mean reference line
}


def load_flat_data(val_split: float = 0.1):
    print("Loading CIFAR-10 via model_training.get_data() …")
    train_ds, val_ds, test_ds = get_data(
        transform="standardize",
        random_crop=False,       # no augmentation for KNN
        horizontal_flip=False,   # no augmentation for KNN
        val_split=val_split,
    )

    print(f"Dataset splits — train: {len(train_ds)}, "
          f"val: {len(val_ds)}, test: {len(test_ds)}")

    print("Converting datasets to flat NumPy arrays …")
    X_train, y_train = _to_flat_numpy(train_ds)
    X_val,   y_val   = _to_flat_numpy(val_ds)
    X_test,  y_test  = _to_flat_numpy(test_ds)

    return X_train, y_train, X_val, y_val, X_test, y_test


def _to_flat_numpy(dataset):
    """
    Use model_training.get_dataset_as_numpy() where possible; fall back to a
    manual DataLoader pass for Subset objects (which don't report len() the
    same way as full datasets).
    """
    try:
        X, y = get_dataset_as_numpy(dataset)
    except Exception:
        loader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
        X, y = next(iter(loader))
        X, y = X.numpy(), y.numpy()

    n = X.shape[0]
    return X.reshape(n, -1), y  # (N, 3, 32, 32) → (N, 3072)


def select_k(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    k_values=None,
    metric: str = "euclidean",
    weights: str = "uniform",
    algorithm: str = "auto",
    p: int = 2,
):
    if k_values is None:
        k_values = [1, 3, 5, 7, 9, 11, 14, 17, 20]

    val_results  = {}
    test_results = {}

    print(f"\nK selection (metric={metric}, weights={weights}, p={p}) …")
    print(f"{'k':>4}  {'Val Err':>8}  {'Test Err':>9}")
    print("─" * 26)

    for k in k_values:
        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric,
            weights=weights,
            algorithm=algorithm,
            p=p,
            n_jobs=-1,
        )
        knn.fit(X_train, y_train)
        val_err  = 1.0 - accuracy_score(y_val,  knn.predict(X_val))
        test_err = 1.0 - accuracy_score(y_test, knn.predict(X_test))
        val_results[k]  = val_err
        test_results[k] = test_err
        print(f"{k:>4}  {val_err:>8.4f}  {test_err:>9.4f}")

    best_k = min(val_results, key=val_results.get)   # minimise error
    print(f"\nBest k = {best_k}  (val error = {val_results[best_k]:.4f})")
    return val_results, test_results, best_k


def evaluate(model: KNeighborsClassifier, X_test, y_test):
    """
    Evaluate the final KNN on the test set.

    Mirrors model_training.model_eval() in spirit:
    both print error rate + a per-class breakdown for the paper.
    """
    y_pred    = model.predict(X_test)
    acc       = accuracy_score(y_test, y_pred)
    test_err  = 1.0 - acc

    print(f"\nTest Error:    {test_err:.4f}  ({test_err * 100:.2f}%)")
    print(f"Test Accuracy: {acc:.4f}  ({acc * 100:.2f}%)\n")
    print(classification_report(y_test, y_pred, target_names=CIFAR10_CLASSES))
    return y_pred, test_err


def plot_k_selection(val_results: dict, test_results: dict, best_k: int,
                     save: bool = True):
    """Validation vs. test error rate across k values."""
    ks       = list(val_results.keys())
    val_err  = list(val_results.values())
    test_err = list(test_results.values())

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(ks, val_err,  marker="o", linewidth=2,
            color=PALETTE["primary"],   label="Val Error")
    ax.plot(ks, test_err, marker="s", linewidth=2, linestyle="--",
            color=PALETTE["secondary"], label="Test Error")
    ax.axvline(best_k, color=PALETTE["accent"], linestyle=":", linewidth=1.5,
               label=f"Best k = {best_k}")
    ax.scatter([best_k], [val_results[best_k]],
               color=PALETTE["primary"],   zorder=5)
    ax.scatter([best_k], [test_results[best_k]],
               color=PALETTE["secondary"], zorder=5)

    ax.set_xlabel("k (number of neighbours)", fontsize=12)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.set_title("KNN – Validation vs. Test Error across k",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(range(0, max(ks) + 1, 2))

    plt.tight_layout()
    if save:
        plt.savefig("knn_k_selection.png", dpi=150, bbox_inches="tight")
        print("Saved → knn_k_selection.png")
    plt.show()


def plot_confusion_matrix(y_test, y_pred, save: bool = True):
    """Confusion matrix on the test set."""
    cm  = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=CIFAR10_CLASSES).plot(
        ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45
    )
    ax.set_title(f"KNN Predictions | Accuracy = {acc * 100:.2f}%",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig("knn_confusion_matrix.png", dpi=150, bbox_inches="tight")
        print("Saved → knn_confusion_matrix.png")
    plt.show()


def plot_per_class_error(y_test, y_pred, save: bool = True):
    """
    Bar chart of per-class accuracy.
    Green bars = at or above the mean accuracy; red bars = below the mean.
    Mirrors the colour convention used in the CNN per-class charts.
    """
    per_class = {
        name: accuracy_score(y_test[y_test == idx], y_pred[y_test == idx])
        for idx, name in enumerate(CIFAR10_CLASSES)
    }

    names  = list(per_class.keys())
    accs   = list(per_class.values())
    mean   = float(np.mean(accs))
    colors = [
        PALETTE["secondary"] if a >= mean else PALETTE["accent"]
        for a in accs
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(mean, color=PALETTE["mean_line"], linestyle="--",
               linewidth=1.5, label=f"Mean = {mean:.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("KNN – Per-class Accuracy (Test Set)",
                 fontsize=14, fontweight="bold")
    ax.legend()

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save:
        plt.savefig("knn_per_class_accuracy.png", dpi=150, bbox_inches="tight")
        print("Saved → knn_per_class_accuracy.png")
    plt.show()


if __name__ == "__main__":
    VAL_SPLIT    = 0.1  
    K_CANDIDATES = [1, 3, 5, 7, 9, 11, 14, 17, 20]
    DISTANCE_METRIC = "euclidean"
    # Weighting scheme: 'uniform' (all neighbours vote equally) or
    # 'distance' (closer neighbours cast a stronger vote). 'distance'
    # tends to reduce error near decision boundaries.
    WEIGHTS = "distance"
    ALGORITHM = "auto"
    P = 2

    X_train, y_train, X_val, y_val, X_test, y_test = load_flat_data(VAL_SPLIT)

    k_val_results, k_test_results, best_k = select_k(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        k_values=K_CANDIDATES,
        metric=DISTANCE_METRIC,
        weights=WEIGHTS,
        algorithm=ALGORITHM,
        p=P,
    )
    plot_k_selection(k_val_results, k_test_results, best_k)

    print(f"\nTraining final KNN (k={best_k}) on train + val …")
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    final_knn = KNeighborsClassifier(
        n_neighbors=best_k,
        metric=DISTANCE_METRIC,
        weights=WEIGHTS,
        algorithm=ALGORITHM,
        p=P,
        n_jobs=-1,
    )
    final_knn.fit(X_trainval, y_trainval)

    y_pred, test_err = evaluate(final_knn, X_test, y_test)

    plot_confusion_matrix(y_test, y_pred)
    plot_per_class_error(y_test, y_pred)

    print(f"\nDone. Final test error: {test_err * 100:.2f}%")

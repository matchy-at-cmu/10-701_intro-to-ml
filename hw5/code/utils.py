### DO NOT CHANGE ###
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(arr, n=10):
    """
    Smooth a 1D sequence with a trailing window of size n.
    """
    arr = np.asarray(arr, dtype=float)
    n = min(n, len(arr))
    average = np.cumsum(arr)
    average[n:] = (average[n:] - average[:-n]) / n
    average[:n] = average[:n] / (1 + np.arange(n))
    return average


def plot_returns(train_returns, eval_returns, eval_every,
                 filename="training_returns.png", window=10):
    """
    Plot training and evaluation returns against the number of policy updates
    and save the figure. Include this figure in your writeup.
    """
    train_returns = np.asarray(train_returns, dtype=float)
    updates = np.arange(1, len(train_returns) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(updates, train_returns, alpha=0.35, label="Training return (per update)")
    plt.plot(updates, moving_average(train_returns, n=window),
             label=f"Training return (rolling mean, window = {window})")

    if len(eval_returns) > 0:
        eval_returns = np.asarray(eval_returns, dtype=float)
        eval_updates = np.arange(1, len(eval_returns) + 1) * eval_every
        plt.plot(eval_updates, eval_returns, marker="o", markersize=3,
                 label="Evaluation return")

    plt.xlabel("Policy updates")
    plt.ylabel("Undiscounted return")
    plt.title("REINFORCE on CartPole")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return

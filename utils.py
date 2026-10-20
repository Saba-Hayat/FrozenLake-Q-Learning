@'
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window=500):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def save_training_plot(rewards, output_path, window=500):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(rewards, alpha=0.22, label="Episode reward")
    smooth = moving_average(rewards, window)
    if len(smooth):
        x = np.arange(len(smooth)) + (window - 1 if len(rewards) >= window else 0)
        plt.plot(x, smooth, linewidth=2, label="Moving average")
    plt.xlabel("Episode")
    plt.ylabel("Reward (0 or 1)")
    plt.title("FrozenLake Q-Learning Training Progress")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_comparison_plot(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(results.keys())
    rates = [results[name]["success_rate"] for name in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, rates)
    plt.ylabel("Evaluation success rate")
    plt.xlabel("Hyperparameter setting")
    plt.title("Hyperparameter Sensitivity")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.1%}",
            ha="center",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_gif(frames, output_path, duration=0.35):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames were recorded; GIF cannot be created.")
    imageio.mimsave(output_path, frames, duration=duration, loop=0)
'@ | Set-Content -Encoding UTF8 utils.py
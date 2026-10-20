
from pathlib import Path
import json
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from agent import QLearningAgent

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
NOTEBOOKS = ROOT / "notebooks"

for folder in (MODELS, OUTPUTS, NOTEBOOKS):
    folder.mkdir(parents=True, exist_ok=True)

Q_TABLE_PATH = MODELS / "q_table.npy"
TRAINING_PLOT = OUTPUTS / "training_rewards.png"
EVAL_PATH = OUTPUTS / "evaluation_results.json"
GIF_PATH = OUTPUTS / "trained_agent.gif"
COMPARISON_PLOT = OUTPUTS / "hyperparameter_comparison.png"
COMPARISON_JSON = OUTPUTS / "hyperparameter_results.json"
PDF_PATH = ROOT / "report.pdf"
NOTEBOOK_PATH = NOTEBOOKS / "FrozenLake_Q_Learning.ipynb"


def evaluate(agent, episodes=300):
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
    )
    successes = 0
    total_steps = 0

    for episode in range(episodes):
        state, _ = env.reset(seed=300000 + episode)

        for step in range(1, 101):
            action = agent.choose_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                successes += int(reward > 0)
                total_steps += step
                break

    env.close()
    return {
        "success_rate": successes / episodes,
        "average_steps": total_steps / episodes,
        "episodes": episodes,
    }


def run_comparison():
    settings = [
        {"name": "Low LR (0.05)", "lr": 0.05},
        {"name": "Default LR (0.10)", "lr": 0.10},
        {"name": "High LR (0.30)", "lr": 0.30},
    ]
    results = {}

    for setting in settings:
        print("Comparing:", setting["name"])
        env = gym.make(
            "FrozenLake-v1",
            map_name="4x4",
            is_slippery=False,
        )
        agent = QLearningAgent(
            state_size=env.observation_space.n,
            action_size=env.action_space.n,
            learning_rate=setting["lr"],
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.02,
            epsilon_decay=0.999,
            seed=42,
        )

        for episode in range(4000):
            state, _ = env.reset(seed=50000 + episode)
            for _ in range(100):
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.learn(state, action, reward, next_state, terminated)
                state = next_state
                if terminated or truncated:
                    break
            agent.decay_exploration()

        env.close()
        results[setting["name"]] = evaluate(agent, episodes=300)

    with COMPARISON_JSON.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    names = list(results)
    rates = [results[name]["success_rate"] for name in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, rates)
    plt.ylim(0, 1.12)
    plt.ylabel("Evaluation success rate")
    plt.title("Q-Learning Learning-Rate Comparison")
    plt.grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{rate:.1%}",
            ha="center",
        )
    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT, dpi=160)
    plt.close()
    return results


def make_pdf(evaluation, comparison):
    with PdfPages(PDF_PATH) as pdf:
        # Page 1: Problem and environment
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.94, "FrozenLake-v1: Q-Learning Agent",
                 ha="center", fontsize=19, weight="bold")
        fig.text(0.5, 0.905, "Reinforcement Learning Assignment Report",
                 ha="center", fontsize=12)
        page1 = [
            ("1. Objective",
             "Implement a tabular reinforcement-learning agent from scratch "
             "and train it to navigate the FrozenLake grid from the start tile "
             "to the goal while avoiding holes."),
            ("2. Environment",
             "Environment: FrozenLake-v1, 4x4 map, is_slippery=False. "
             "The deterministic setting makes actions predictable and supports "
             "clear evaluation of the learned policy."),
            ("3. States, actions and rewards",
             "There are 16 discrete states and 4 actions: left, down, right "
             "and up. Reaching the goal gives reward 1; other transitions "
             "give reward 0. Episodes end at the goal, a hole or the step limit."),
            ("4. State representation",
             "Each grid position is represented by an integer from 0 to 15. "
             "The agent stores one Q-value for each state-action pair, producing "
             "a 16 x 4 Q-table."),
        ]
        y = 0.83
        for heading, body in page1:
            fig.text(0.10, y, heading, fontsize=13, weight="bold", va="top")
            fig.text(0.10, y - 0.035, body, fontsize=10.5, va="top",
                     wrap=True, linespacing=1.5)
            y -= 0.16
        fig.text(0.10, 0.12,
                 "Implementation: Python, Gymnasium, NumPy and Matplotlib.",
                 fontsize=10)
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Algorithm and hyperparameters
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "5. Method and Algorithm",
                 fontsize=17, weight="bold")
        fig.text(0.08, 0.885,
                 "The agent uses tabular Q-Learning, an off-policy temporal-difference "
                 "method. It explores actions with an epsilon-greedy policy and "
                 "updates the Q-table after each transition.",
                 fontsize=11, wrap=True, va="top", linespacing=1.5)
        fig.text(0.08, 0.79, "Q-learning update rule", fontsize=13, weight="bold")
        fig.text(0.10, 0.745,
                 "Q(s,a) <- Q(s,a) + alpha [r + gamma max Q(s',a') - Q(s,a)]",
                 fontsize=11)
        fig.text(0.08, 0.68,
                 "The future-value term is omitted when the episode terminates.",
                 fontsize=10.5)
        fig.text(0.08, 0.61, "Training hyperparameters",
                 fontsize=13, weight="bold")
        rows = [
            ["Parameter", "Value"],
            ["Training episodes", "20,000"],
            ["Maximum steps per episode", "100"],
            ["Learning rate (alpha)", "0.10"],
            ["Discount factor (gamma)", "0.99"],
            ["Initial epsilon", "1.00"],
            ["Minimum epsilon", "0.02"],
            ["Epsilon decay", "0.9995 per episode"],
            ["Random seed", "42"],
        ]
        table = plt.table(
            cellText=rows[1:],
            colLabels=rows[0],
            cellLoc="left",
            colLoc="left",
            bbox=[0.08, 0.25, 0.84, 0.31],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        fig.text(0.08, 0.17,
                 "Exploration gradually decreases so the agent shifts from "
                 "trying actions to using learned Q-values.",
                 fontsize=10.5, wrap=True)
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: Actual results and plots
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.95, "6. Results and Hyperparameter Comparison",
                 fontsize=16, weight="bold")
        result_text = (
            f"Evaluation episodes: {evaluation['evaluation_episodes']}\n"
            f"Successful episodes: {evaluation['successful_episodes']}\n"
            f"Success rate: {evaluation['success_rate']:.1%}\n"
            f"Average steps per episode: {evaluation['average_steps']:.2f}"
        )
        fig.text(0.10, 0.88, result_text, fontsize=12, va="top",
                 linespacing=1.7)
        if TRAINING_PLOT.exists():
            image = plt.imread(TRAINING_PLOT)
            ax = fig.add_axes([0.12, 0.48, 0.76, 0.25])
            ax.imshow(image)
            ax.axis("off")
            ax.set_title("Training reward curve", fontsize=11)
        if COMPARISON_PLOT.exists():
            image = plt.imread(COMPARISON_PLOT)
            ax = fig.add_axes([0.12, 0.13, 0.76, 0.25])
            ax.imshow(image)
            ax.axis("off")
            ax.set_title("Learning-rate comparison", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: Limitations, failure modes and conclusion
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.94, "7. Discussion, Failure Modes and Conclusion",
                 fontsize=15, weight="bold")
        discussion = [
            ("Observed result",
             f"The trained agent achieved a {evaluation['success_rate']:.1%} "
             f"success rate across {evaluation['evaluation_episodes']} evaluation "
             f"episodes, with {evaluation['average_steps']:.2f} average steps."),
            ("Failure modes and limitations",
             "An untrained agent may fall into holes because exploration includes "
             "random actions. Learning may be slower with an unsuitable learning "
             "rate, insufficient episodes or a poor exploration schedule. The "
             "tabular method does not scale well to very large or continuous "
             "state spaces. Results on this non-slippery map do not establish "
             "performance on stochastic FrozenLake."),
            ("Hyperparameter comparison",
             "Three learning rates (0.05, 0.10 and 0.30) were trained and tested "
             "using the same environment type. Their measured evaluation success "
             "rates are shown in the comparison chart; the best setting is the "
             "one with the highest measured success rate."),
            ("Conclusion",
             "The project implements Q-Learning from scratch, saves the learned "
             "Q-table, records a GIF of the greedy policy, and produces training "
             "and comparison plots. The evaluation indicates that the agent "
             "learned a successful route in the deterministic 4x4 environment."),
            ("Reproducibility and artifacts",
             "The project includes source files, requirements, Q-table, training "
             "plots, evaluation JSON, hyperparameter results, a GIF, and an "
             "executed Jupyter notebook."),
        ]
        y = 0.86
        for heading, body in discussion:
            fig.text(0.09, y, heading, fontsize=12, weight="bold", va="top")
            fig.text(0.09, y - 0.035, body, fontsize=10.3, va="top",
                     wrap=True, linespacing=1.45)
            y -= 0.15
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def make_executed_notebook(evaluation, comparison):
    import nbformat
    from nbclient import NotebookClient

    cells = [
        nbformat.v4.new_markdown_cell(
            "# FrozenLake-v1 Q-Learning Agent\n"
            "Executed project notebook. This notebook loads the artifacts "
            "generated by the training script and presents evaluation results."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import numpy as np\n"
            "from IPython.display import Image, display\n"
            "ROOT = Path.cwd()\n"
            "with open(ROOT / 'outputs/evaluation_results.json', encoding='utf-8') as f:\n"
            "    evaluation = json.load(f)\n"
            "with open(ROOT / 'outputs/hyperparameter_results.json', encoding='utf-8') as f:\n"
            "    comparison = json.load(f)\n"
            "q_table = np.load(ROOT / 'models/q_table.npy')\n"
            "print('Environment: FrozenLake-v1, 4x4, non-slippery')\n"
            "print('Evaluation results:', evaluation)\n"
            "print('Q-table shape:', q_table.shape)\n"
            "print('Q-table:')\n"
            "print(np.round(q_table, 3))"
        ),
        nbformat.v4.new_markdown_cell("## Training progress"),
        nbformat.v4.new_code_cell(
            "from IPython.display import display, Image\n"
            "display(Image(filename=str(ROOT / 'outputs/training_rewards.png')))"
        ),
        nbformat.v4.new_markdown_cell("## Learning-rate comparison"),
        nbformat.v4.new_code_cell(
            "print(comparison)\n"
            "display(Image(filename=str(ROOT / 'outputs/hyperparameter_comparison.png')))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Saved policy demonstration\n"
            "The trained greedy policy was recorded as `outputs/trained_agent.gif`. "
            "The evaluation success rate is based on the saved evaluation JSON."
        ),
        nbformat.v4.new_code_cell(
            "print('GIF exists:', (ROOT / 'outputs/trained_agent.gif').exists())\n"
            "print('Evaluation success rate:', f\"{evaluation['success_rate']:.1%}\")\n"
            "print('Average steps:', round(evaluation['average_steps'], 2))"
        ),
    ]

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
    )
    NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.write(notebook, NOTEBOOK_PATH)


def main():
    if not Q_TABLE_PATH.exists() or not EVAL_PATH.exists():
        raise FileNotFoundError(
            "Training artifacts are missing. Run `python main.py` first."
        )

    with EVAL_PATH.open(encoding="utf-8") as file:
        evaluation = json.load(file)

    print("Creating hyperparameter comparison...")
    comparison = run_comparison()

    print("Creating four-page PDF report...")
    make_pdf(evaluation, comparison)

    print("Creating and executing Jupyter notebook...")
    try:
        make_executed_notebook(evaluation, comparison)
        notebook_status = "executed notebook created"
    except Exception as error:
        notebook_status = f"Notebook execution failed: {error}"
        print(notebook_status)
        print("Install notebook dependencies if missing, then rerun this script.")

    print("\nSubmission generation finished.")
    print("PDF:", PDF_PATH)
    print("Notebook:", NOTEBOOK_PATH)
    print("GIF:", GIF_PATH)
    print("Q-table:", Q_TABLE_PATH)
    print("Training plot:", TRAINING_PLOT)
    print("Comparison plot:", COMPARISON_PLOT)
    print(notebook_status)


if __name__ == "__main__":
    main()

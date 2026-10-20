@'
# Custom Reinforcement Learning Agent: FrozenLake Q-Learning

## Objective
Implement tabular Q-Learning from scratch using Python and Farama Gymnasium.

## Environment
- Environment: FrozenLake-v1
- Map: 4x4
- Observation: Discrete state index (0 to 15)
- Actions: Left (0), Down (1), Right (2), Up (3)
- Rewards: +1 for reaching the goal; 0 otherwise
- Variant: is_slippery=False for deterministic, reproducible experiments

## Algorithm
The agent uses epsilon-greedy exploration and the Q-Learning update:

Q(s,a) <- Q(s,a) + alpha [r + gamma max Q(s',a') - Q(s,a)]

For terminal transitions, the future-value term is omitted.

## Project Files
- main.py: training, evaluation, hyperparameter experiments and GIF recording
- agent.py: Q-Learning implementation from scratch
- utils.py: plots, GIF and JSON helpers
- notebooks/: executed Jupyter notebook
- models/q_table.npy: saved learned Q-table
- outputs/: graphs, evaluation metrics and GIF
- report.pdf: technical report, maximum four pages

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
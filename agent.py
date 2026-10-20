
import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent implemented from scratch."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.9995,
        seed=42,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        self.q_table = np.zeros(
            (state_size, action_size),
            dtype=np.float64,
        )

    def choose_action(self, state, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))

        values = self.q_table[state]
        best_actions = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best_actions))

    def learn(self, state, action, reward, next_state, terminated):
        target = reward

        if not terminated:
            target += self.gamma * np.max(self.q_table[next_state])

        old_value = self.q_table[state, action]
        self.q_table[state, action] = (
            old_value
            + self.learning_rate * (target - old_value)
        )

    def decay_exploration(self):
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

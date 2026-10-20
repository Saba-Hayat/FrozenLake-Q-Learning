
from pathlib import Path
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from agent import QLearningAgent

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

EPISODES = 20000
MAX_STEPS = 100
SEED = 42


def train():
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False
    )

    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.9995,
        seed=SEED
    )

    rewards = []

    for episode in range(EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0

        for _ in range(MAX_STEPS):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            agent.learn(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        agent.decay_exploration()
        rewards.append(total_reward)

        if (episode + 1) % 2000 == 0:
            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"Recent success rate: {np.mean(rewards[-500:]):.1%}"
            )

    env.close()
    np.save(MODELS / "q_table.npy", agent.q_table)

    plt.figure(figsize=(9, 5))
    plt.plot(rewards, alpha=0.25, label="Episode reward")
    window = 500
    if len(rewards) >= window:
        smooth = np.convolve(
            rewards, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            np.arange(len(smooth)) + window - 1,
            smooth,
            label="500-episode moving average"
        )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("FrozenLake Q-Learning Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS / "training_rewards.png", dpi=150)
    plt.close()

    return agent


def evaluate(agent, episodes=1000):
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False
    )
    successes = 0
    steps_taken = []

    for episode in range(episodes):
        state, _ = env.reset(seed=100000 + episode)
        steps = 0

        for steps in range(1, MAX_STEPS + 1):
            action = agent.choose_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                successes += int(reward > 0)
                break

        steps_taken.append(steps)

    env.close()

    success_rate = successes / episodes
    print(f"\nEvaluation episodes: {episodes}")
    print(f"Successful episodes: {successes}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average steps: {np.mean(steps_taken):.2f}")

    results = {
        "environment": "FrozenLake-v1 (4x4, non-slippery)",
        "evaluation_episodes": episodes,
        "successful_episodes": successes,
        "success_rate": success_rate,
        "average_steps": float(np.mean(steps_taken))
    }

    import json
    with (OUTPUTS / "evaluation_results.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    return results


def record_gif(agent):
    env = gym.make(
        "FrozenLake-v1",
        map_name="4x4",
        is_slippery=False,
        render_mode="rgb_array"
    )
    state, _ = env.reset(seed=7)
    frames = [env.render()]

    for _ in range(MAX_STEPS):
        action = agent.choose_action(state, explore=False)
        state, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())

        if terminated or truncated:
            break

    env.close()
    imageio.mimsave(
        OUTPUTS / "trained_agent.gif",
        frames,
        duration=500,
        loop=0
    )
    print(f"GIF saved: {OUTPUTS / 'trained_agent.gif'}")
    print(f"GIF episode reward: {reward}")


def main():
    print("Starting FrozenLake Q-Learning training...")
    agent = train()
    evaluate(agent)
    record_gif(agent)
    print("\nTraining and evaluation completed.")
    print("Next, run: python create_submission.py")


if __name__ == "__main__":
    main()

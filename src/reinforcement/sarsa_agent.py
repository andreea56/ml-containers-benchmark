import os
import time
import numpy as np
import gymnasium as gym

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utile import save_metrics, SEED, ResourceMonitor

ENV_NAME = os.environ.get("RL_ENV", "FrozenLake-v1")
EPISODES = int(os.environ.get("EPISODES", 5000))
ALPHA = float(os.environ.get("ALPHA", 0.1))
GAMMA = float(os.environ.get("GAMMA", 0.99))
EPSILON_START = float(os.environ.get("EPSILON_START", 1.0))
EPSILON_MIN = float(os.environ.get("EPSILON_MIN", 0.01))
EPSILON_DECAY = float(os.environ.get("EPSILON_DECAY", 0.9995))


def epsilon_greedy(Q, state, n_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[state]))


def main():
    start = time.time()
    np.random.seed(SEED)
    monitor = ResourceMonitor()
    monitor.start()

    env = gym.make(ENV_NAME, is_slippery=True)
    env.action_space.seed(SEED)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    epsilon = EPSILON_START
    rewards_history = []

    for episode in range(EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        action = epsilon_greedy(Q, state, n_actions, epsilon)
        terminated = truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, n_actions, epsilon)

            td_target = reward + GAMMA * Q[next_state, next_action] * (not terminated)
            Q[state, action] += ALPHA * (td_target - Q[state, action])

            state, action = next_state, next_action
            total_reward += reward

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        rewards_history.append(total_reward)

    env.close()

    last_window = rewards_history[-500:] if len(rewards_history) >= 500 else rewards_history
    resource_stats = monitor.stop()
    elapsed = round(time.time() - start, 3)

    metrics = {
        "algorithm": "SARSA",
        "environment": ENV_NAME,
        "episodes": EPISODES,
        "final_epsilon": round(epsilon, 4),
        "success_rate_last_500_episodes": round(float(np.mean(last_window)), 4),
        "avg_reward_all_episodes": round(float(np.mean(rewards_history)), 4),
        "training_time_seconds": elapsed,
        "throughput_episodes_per_second": round(EPISODES / elapsed, 2),
        **resource_stats,
    }
    save_metrics("sarsa", metrics)


if __name__ == "__main__":
    main()
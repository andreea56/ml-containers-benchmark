import os
import time
import random
from collections import deque

import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers, Model

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utile import save_metrics, SEED, ResourceMonitor

ENV_NAME = os.environ.get("RL_ENV", "CartPole-v1")
EPISODES = int(os.environ.get("EPISODES", 300))
GAMMA = float(os.environ.get("GAMMA", 0.99))
LR = float(os.environ.get("LR", 1e-3))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))
BUFFER_SIZE = int(os.environ.get("BUFFER_SIZE", 20000))
EPSILON_START = float(os.environ.get("EPSILON_START", 1.0))
EPSILON_MIN = float(os.environ.get("EPSILON_MIN", 0.01))
EPSILON_DECAY = float(os.environ.get("EPSILON_DECAY", 0.995))
TARGET_UPDATE_EVERY = int(os.environ.get("TARGET_UPDATE_EVERY", 10))


def build_q_network(state_dim: int, n_actions: int) -> Model:
    inputs = layers.Input(shape=(state_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(n_actions, activation="linear")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
    return model


def main():
    start = time.time()
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    monitor = ResourceMonitor()
    monitor.start()

    env = gym.make(ENV_NAME)
    env.action_space.seed(SEED)
    state_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)

    q_network = build_q_network(state_dim, n_actions)
    target_network = build_q_network(state_dim, n_actions)
    target_network.set_weights(q_network.get_weights())

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    epsilon = EPSILON_START
    rewards_history = []

    for episode in range(EPISODES):
        state, _ = env.reset(seed=SEED + episode)
        terminated = truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = q_network.predict(state[np.newaxis, :], verbose=0)
                action = int(np.argmax(q_values[0]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, terminated))
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= BATCH_SIZE:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                states = np.array([b[0] for b in batch])
                actions = np.array([b[1] for b in batch])
                rewards = np.array([b[2] for b in batch])
                next_states = np.array([b[3] for b in batch])
                dones = np.array([b[4] for b in batch])

                target_q = q_network.predict(states, verbose=0)
                next_q = target_network.predict(next_states, verbose=0)
                max_next_q = np.max(next_q, axis=1)

                for i in range(BATCH_SIZE):
                    target_q[i, actions[i]] = rewards[i] + GAMMA * max_next_q[i] * (not dones[i])

                q_network.fit(states, target_q, epochs=1, verbose=0)

        if episode % TARGET_UPDATE_EVERY == 0:
            target_network.set_weights(q_network.get_weights())

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        rewards_history.append(total_reward)

    env.close()

    last_window = rewards_history[-50:] if len(rewards_history) >= 50 else rewards_history
    resource_stats = monitor.stop()
    elapsed = round(time.time() - start, 3)

    metrics = {
        "algorithm": "DQN",
        "environment": ENV_NAME,
        "episodes": EPISODES,
        "final_epsilon": round(epsilon, 4),
        "avg_reward_last_50_episodes": round(float(np.mean(last_window)), 2),
        "avg_reward_all_episodes": round(float(np.mean(rewards_history)), 2),
        "training_time_seconds": elapsed,
        "throughput_episodes_per_second": round(EPISODES / elapsed, 2),
        **resource_stats,
    }

    save_metrics("dqn", metrics)


if __name__ == "__main__":
    main()
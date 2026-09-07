"""
PyTorch DQN trainer for snake_engine.SnakeGame.

Run:
    python train_snake.py

Output:
    weights.npz       Final model after training
    best_weights.npz  Model with the highest score reached during training

The saved weights are compatible with main.py's NeuralNetworkAgent,
provided that main.py uses:

    input_size=22
    hidden_size=64
    output_size=3
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_engine import SnakeConfig, SnakeGame


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 7

WEIGHTS_PATH = "weights.npz"
BEST_WEIGHTS_PATH = "best_weights.npz"

NUM_EPISODES = 12_000
MAX_STEPS_PER_EPISODE = 2_000

REPLAY_CAPACITY = 50_000
BATCH_SIZE = 128
MIN_REPLAY_SIZE = 2_000

LEARNING_RATE = 0.001
GAMMA = 0.95

TARGET_UPDATE_EVERY = 500
TRAIN_EVERY = 1

EPSILON_START = 1.0
EPSILON_END = 0.03
EPSILON_DECAY_STEPS = 100_000

# SnakeGame.get_state() now returns 22 features.
INPUT_SIZE = 22
HIDDEN_SIZE = 64
OUTPUT_SIZE = 3


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class SnakeDQN(nn.Module):
    """
    Architecture:

        22 inputs
          ↓
        Linear(64) + ReLU
          ↓
        Linear(3)

    Actions:

        0 = straight
        1 = turn right
        2 = turn left
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                float(done),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

def get_epsilon(step: int) -> float:
    """
    Linearly decreases exploration from EPSILON_START to EPSILON_END.
    """
    fraction = min(step / EPSILON_DECAY_STEPS, 1.0)

    return EPSILON_START + fraction * (
        EPSILON_END - EPSILON_START
    )


def choose_action(
    model: SnakeDQN,
    state: np.ndarray,
    epsilon: float,
) -> int:
    """
    Epsilon-greedy action selection.
    """
    if random.random() < epsilon:
        return random.randrange(OUTPUT_SIZE)

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)

    return int(torch.argmax(q_values, dim=1).item())


# ---------------------------------------------------------------------------
# One optimization step
# ---------------------------------------------------------------------------

def optimize_model(
    online_model: SnakeDQN,
    target_model: SnakeDQN,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
) -> float | None:
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return None

    (
        states,
        actions,
        rewards,
        next_states,
        dones,
    ) = replay_buffer.sample(BATCH_SIZE)

    states_tensor = torch.as_tensor(
        states,
        dtype=torch.float32,
        device=DEVICE,
    )

    actions_tensor = torch.as_tensor(
        actions,
        dtype=torch.int64,
        device=DEVICE,
    ).unsqueeze(1)

    rewards_tensor = torch.as_tensor(
        rewards,
        dtype=torch.float32,
        device=DEVICE,
    )

    next_states_tensor = torch.as_tensor(
        next_states,
        dtype=torch.float32,
        device=DEVICE,
    )

    dones_tensor = torch.as_tensor(
        dones,
        dtype=torch.float32,
        device=DEVICE,
    )

    # Q-values for the actions actually taken.
    current_q_values = online_model(states_tensor)
    current_q_values = current_q_values.gather(
        1,
        actions_tensor,
    ).squeeze(1)

    # Target:
    #
    # reward                                  if terminal
    # reward + gamma * max(next Q-values)     otherwise
    with torch.no_grad():
        next_q_values = target_model(next_states_tensor)
        best_next_q_values = next_q_values.max(dim=1).values

        target_q_values = rewards_tensor + (
            GAMMA
            * best_next_q_values
            * (1.0 - dones_tensor)
        )

    loss = nn.functional.smooth_l1_loss(
        current_q_values,
        target_q_values,
    )

    optimizer.zero_grad()
    loss.backward()

    # Prevent unusually large parameter updates.
    torch.nn.utils.clip_grad_norm_(
        online_model.parameters(),
        max_norm=5.0,
    )

    optimizer.step()

    return float(loss.item())


# ---------------------------------------------------------------------------
# Saving weights for main.py
# ---------------------------------------------------------------------------

def save_weights(
    model: SnakeDQN,
    path: str,
):
    """
    Converts PyTorch tensors to NumPy arrays and saves the exact names
    expected by main.py's NeuralNetworkAgent.load_weights().

    PyTorch Linear weights are stored as:

        first layer:  (hidden_size, input_size)
        second layer: (output_size, hidden_size)

    main.py expects:

        W1: (input_size, hidden_size)
        W2: (hidden_size, output_size)

    Therefore both weight matrices are transposed here.
    """
    state_dict = model.state_dict()

    W1 = (
        state_dict["network.0.weight"]
        .detach()
        .cpu()
        .numpy()
        .T
    )

    b1 = (
        state_dict["network.0.bias"]
        .detach()
        .cpu()
        .numpy()
    )

    W2 = (
        state_dict["network.2.weight"]
        .detach()
        .cpu()
        .numpy()
        .T
    )

    b2 = (
        state_dict["network.2.bias"]
        .detach()
        .cpu()
        .numpy()
    )

    np.savez(
        path,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
    )

    print(f"Saved model to {path}")
    print(f"  W1 shape: {W1.shape}")
    print(f"  b1 shape: {b1.shape}")
    print(f"  W2 shape: {W2.shape}")
    print(f"  b2 shape: {b2.shape}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    config = SnakeConfig()
    game = SnakeGame(config)

    online_model = SnakeDQN().to(DEVICE)
    target_model = SnakeDQN().to(DEVICE)

    target_model.load_state_dict(
        online_model.state_dict()
    )
    target_model.eval()

    optimizer = optim.Adam(
        online_model.parameters(),
        lr=LEARNING_RATE,
    )

    replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

    total_steps = 0
    best_score = -1

    recent_scores = deque(maxlen=100)
    recent_rewards = deque(maxlen=100)

    last_epsilon = EPSILON_START

    for episode in range(1, NUM_EPISODES + 1):
        state = game.reset().astype(np.float32)

        # Confirm that snake_engine.py is returning the expanded
        # 22-feature observation.
        if state.shape != (INPUT_SIZE,):
            raise ValueError(
                f"Expected initial state shape {(INPUT_SIZE,)}, "
                f"but received {state.shape}. "
                f"Update SnakeGame.get_state() in snake_engine.py."
            )

        episode_reward = 0.0
        episode_steps = 0

        terminated = False
        truncated = False

        while not terminated and not truncated:
            epsilon = get_epsilon(total_steps)
            last_epsilon = epsilon

            action = choose_action(
                model=online_model,
                state=state,
                epsilon=epsilon,
            )

            next_state, reward, terminated, truncated, info = (
                game.step(action)
            )

            next_state = np.asarray(
                next_state,
                dtype=np.float32,
            )

            if next_state.shape != (INPUT_SIZE,):
                raise ValueError(
                    f"Expected next state shape {(INPUT_SIZE,)}, "
                    f"but received {next_state.shape}. "
                    f"Update SnakeGame.get_state() in snake_engine.py."
                )

            done = bool(terminated or truncated)

            replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            state = next_state
            episode_reward += float(reward)
            episode_steps += 1
            total_steps += 1

            if total_steps % TRAIN_EVERY == 0:
                optimize_model(
                    online_model=online_model,
                    target_model=target_model,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                )

            if total_steps % TARGET_UPDATE_EVERY == 0:
                target_model.load_state_dict(
                    online_model.state_dict()
                )

            # Prevent pathological episodes from running indefinitely.
            if episode_steps >= MAX_STEPS_PER_EPISODE:
                truncated = True

        # SnakeGame.score is the number of food items eaten in the episode.
        score = int(game.score)

        recent_scores.append(score)
        recent_rewards.append(episode_reward)

        # Save immediately when a new best score is achieved.
        if score > best_score:
            best_score = score

            save_weights(
                model=online_model,
                path=BEST_WEIGHTS_PATH,
            )

            print(
                f"New best model | "
                f"episode={episode} | "
                f"score={score} | "
                f"steps={total_steps} | "
                f"saved={BEST_WEIGHTS_PATH}"
            )

        if episode == 1 or episode % 100 == 0:
            average_score = (
                sum(recent_scores) / len(recent_scores)
            )

            average_reward = (
                sum(recent_rewards) / len(recent_rewards)
            )

            print(
                f"Episode {episode:5d} | "
                f"steps {total_steps:8d} | "
                f"score {score:4d} | "
                f"best {best_score:4d} | "
                f"average score {average_score:6.2f} | "
                f"average reward {average_reward:8.2f} | "
                f"epsilon {last_epsilon:.3f}"
            )

    # Save the final state of the model after all training episodes.
    save_weights(
        model=online_model,
        path=WEIGHTS_PATH,
    )

    print()
    print("Training complete.")
    print(f"Best score:       {best_score}")
    print(f"Final model:      {WEIGHTS_PATH}")
    print(f"Best model:       {BEST_WEIGHTS_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()

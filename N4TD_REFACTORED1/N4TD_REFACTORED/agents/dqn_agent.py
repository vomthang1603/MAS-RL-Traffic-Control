# =============================================================================
# dqn_agent.py — Deep Q-Network Agent
#
# Kiến trúc:
#   - Online Network  : dùng để chọn action và train
#   - Target Network  : bản sao của Online, cập nhật định kỳ → ổn định training
#   - Experience Replay Buffer: lưu (s, a, r, s', done), sample mini-batch ngẫu nhiên
# =============================================================================

import os
import sys
import random
import numpy as np
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    ACTIONS, ACTION_SIZE,
    ALPHA, GAMMA, EPSILON,
    DQN_HIDDEN_UNITS, DQN_LEARNING_RATE,
    REPLAY_BUFFER_SIZE, BATCH_SIZE, TARGET_UPDATE_FREQ,
    DQN_WEIGHTS_DIR,
)
from agents.base_agent import BaseAgent


class ReplayBuffer:
    """Circular buffer lưu trữ experience (s, a, r, s', done)."""

    def __init__(self, max_size: int = REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """
    DQN Agent cho 1 nút đèn giao thông.

    Params
    ------
    state_size : int   — chiều dài state vector
    name       : str   — 'GL1' hoặc 'GL2'
    epsilon    : float — exploration rate ban đầu
    """

    def __init__(self, state_size: int, name: str = 'agent',
                 epsilon: float = EPSILON):
        super().__init__(state_size=state_size, action_size=ACTION_SIZE,
                         name=name, epsilon=epsilon)
        self.train_step = 0

        self.online_net = self._build_network(state_size)
        self.target_net = self._build_network(state_size)
        self._sync_target()

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    # ------------------------------------------------------------------
    # Xây dựng mạng neural
    # ------------------------------------------------------------------

    @staticmethod
    def _build_network(state_size: int) -> keras.Model:
        """
        Feedforward network:
            Input(state_size) → Dense(64, relu) → Dense(64, relu) → Dense(2, linear)
        """
        model = keras.Sequential(name='dqn')
        model.add(layers.Input(shape=(state_size,)))
        for units in DQN_HIDDEN_UNITS:
            model.add(layers.Dense(units, activation='relu',
                                   kernel_initializer='he_uniform'))
            model.add(layers.BatchNormalization())
        model.add(layers.Dense(ACTION_SIZE, activation='linear'))
        model.compile(
            loss=keras.losses.Huber(),
            optimizer=keras.optimizers.Adam(learning_rate=DQN_LEARNING_RATE),
        )
        return model

    def _sync_target(self):
        """Copy weights từ online → target network."""
        self.target_net.set_weights(self.online_net.get_weights())

    # ------------------------------------------------------------------
    # Action selection (ε-greedy)
    # ------------------------------------------------------------------

    def get_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q_vals = self.online_net(state.reshape(1, -1), training=False).numpy()[0]
        return int(np.argmax(q_vals))

    # ------------------------------------------------------------------
    # Experience replay
    # ------------------------------------------------------------------

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Train từ replay buffer
    # ------------------------------------------------------------------

    def train(self, *args, **kwargs):
        """
        Sample mini-batch từ Replay Buffer, tính TD-target bằng Target Network,
        rồi train Online Network. Trả về loss hoặc None nếu buffer chưa đủ.
        """
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(BATCH_SIZE)

        next_q    = self.target_net(next_states, training=False).numpy()
        td_targets = rewards + GAMMA * np.max(next_q, axis=1) * (1 - dones)

        current_q = self.online_net(states, training=False).numpy()
        current_q[np.arange(actions.shape[0]), actions] = td_targets

        loss = float(self.online_net.train_on_batch(states, current_q))

        self.train_step += 1
        if self.train_step % TARGET_UPDATE_FREQ == 0:
            self._sync_target()
            print(f"    [{self.name}] Target synced @ step={self.train_step}")

        return loss

    # ------------------------------------------------------------------
    # Save / Load — lưu vào models/dqn_weights/
    # ------------------------------------------------------------------

    def save(self, scenario: str) -> str:
        path = os.path.join(DQN_WEIGHTS_DIR, f'dqn_{self.name}_{scenario}.keras')
        self.online_net.save(path)
        print(f"  [{self.name}] Model saved → {path}")
        return path

    def load(self, path: str):
        self.online_net = keras.models.load_model(path)
        self._sync_target()
        print(f"  [{self.name}] Model loaded ← {path}")

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.online_net(state.reshape(1, -1), training=False).numpy()[0]

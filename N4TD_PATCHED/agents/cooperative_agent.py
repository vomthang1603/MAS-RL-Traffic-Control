"""
agents/cooperative_agent.py
----------------------------
DQN Agent với phối hợp MAS:
  - GL1: state_size = 25 (18 local + 7 từ GL2 qua MessageBus)
  - GL2: state_size = 18 (giữ nguyên, publish lên bus)
  - Không dùng BatchNorm (fix lỗi dqn_agent.py gốc)
  - Double DQN + Prioritized Experience Replay (optional)
"""

import random
import numpy as np
from collections import deque
from typing import Optional, List, Tuple

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not found. CooperativeAgent will run in stub mode.")


# ──────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Experience Replay đơn giản (uniform sampling)."""

    def __init__(self, capacity: int = 50_000):
        self._buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self._buffer, batch_size)

    def __len__(self):
        return len(self._buffer)


# ──────────────────────────────────────────────────────────────────────────────
# Cooperative DQN Agent
# ──────────────────────────────────────────────────────────────────────────────

class CooperativeAgent:
    """
    DQN Agent hỗ trợ phối hợp MAS.

    Args:
        agent_id:        "GL1" hoặc "GL2"
        state_size:      25 (GL1) hoặc 18 (GL2)
        action_size:     số pha đèn (GL1=12, GL2=8)
        message_bus:     instance MessageBus (None nếu không dùng)
        learning_rate:   lr Adam
        gamma:           discount factor
        epsilon_*:       tham số ε-greedy
        tau:             soft update target network
    """

    def __init__(
        self,
        agent_id: str,
        state_size: int,
        action_size: int,
        message_bus=None,
        learning_rate: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 50_000,
        target_update_freq: int = 200,   # steps
        tau: float = 0.005,              # soft update coefficient
    ):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.bus = message_bus

        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.target_update_freq = target_update_freq

        self._step_count = 0

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        if TF_AVAILABLE:
            self.q_net = self._build_network()
            self.target_net = self._build_network()
            self._hard_update_target()
            self.optimizer = optimizers.Adam(learning_rate=self.lr)
        else:
            self.q_net = None
            self.target_net = None

    # ------------------------------------------------------------------
    # Network architecture (NO BatchNorm — phù hợp với online RL)
    # ------------------------------------------------------------------

    def _build_network(self):
        """
        MLP 3 lớp với Layer Normalization (thay BatchNorm).
        BatchNorm không phù hợp với DQN vì batch nhỏ và non-stationary.
        """
        inp = layers.Input(shape=(self.state_size,))
        x = layers.Dense(128, activation="relu")(inp)
        x = layers.LayerNormalization()(x)          # thay thế BatchNorm
        x = layers.Dense(128, activation="relu")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(self.action_size, activation="linear")(x)
        model = models.Model(inputs=inp, outputs=out)
        return model

    # ------------------------------------------------------------------
    # State augmentation (GL1 nhận thêm state từ GL2)
    # ------------------------------------------------------------------

    def augment_state(self, local_state: np.ndarray) -> np.ndarray:
        """
        Ghép state cục bộ với state vector nhận từ bus.
        Chỉ GL1 mới cần augment; GL2 dùng local_state trực tiếp.
        """
        if self.agent_id == "GL1" and self.bus is not None:
            extra = np.array(self.bus.get_state_vector("GL2"), dtype=np.float32)
            return np.concatenate([local_state, extra])
        return local_state

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy action selection."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        if not TF_AVAILABLE or self.q_net is None:
            return random.randrange(self.action_size)

        s = state.reshape(1, -1).astype(np.float32)
        q_values = self.q_net(s, training=False).numpy()[0]
        return int(np.argmax(q_values))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        """
        Double DQN update.
        Returns: loss value (float) hoặc None nếu buffer chưa đủ.
        """
        if not TF_AVAILABLE or len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = np.array(states,      dtype=np.float32)
        actions     = np.array(actions,     dtype=np.int32)
        rewards     = np.array(rewards,     dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones       = np.array(dones,       dtype=np.float32)

        # Double DQN: chọn action bằng q_net, đánh giá bằng target_net
        next_actions = np.argmax(
            self.q_net(next_states, training=False).numpy(), axis=1
        )
        next_q = self.target_net(next_states, training=False).numpy()
        next_q_selected = next_q[np.arange(self.batch_size), next_actions]

        targets_full = self.q_net(states, training=False).numpy()
        targets_full[np.arange(self.batch_size), actions] = (
            rewards + self.gamma * next_q_selected * (1.0 - dones)
        )

        with tf.GradientTape() as tape:
            q_pred = self.q_net(states, training=True)
            loss = tf.reduce_mean(tf.square(targets_full - q_pred))

        grads = tape.gradient(loss, self.q_net.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) for g in grads]   # gradient clipping
        self.optimizer.apply_gradients(
            zip(grads, self.q_net.trainable_variables)
        )

        # Soft update target network
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self._soft_update_target()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.numpy())

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def _hard_update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def _soft_update_target(self):
        """τ·θ_online + (1-τ)·θ_target"""
        for t_var, q_var in zip(
            self.target_net.trainable_variables,
            self.q_net.trainable_variables,
        ):
            t_var.assign(self.tau * q_var + (1.0 - self.tau) * t_var)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        if TF_AVAILABLE and self.q_net:
            self.q_net.save_weights(path)

    def load(self, path: str):
        if TF_AVAILABLE and self.q_net:
            self.q_net.load_weights(path)
            self._hard_update_target()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"CooperativeAgent(id={self.agent_id}, "
            f"state={self.state_size}, action={self.action_size}, "
            f"ε={self.epsilon:.3f}, steps={self._step_count})"
        )

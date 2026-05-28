from .base_agent import BaseAgent
from .q_learning_agent import QLearningAgent

# DQN is lazy-loaded so importing Q-learning does not require TensorFlow.

__all__ = ['BaseAgent', 'QLearningAgent', 'DQNAgent', 'ReplayBuffer']


def __getattr__(name: str):
    if name == 'DQNAgent':
        from .dqn_agent import DQNAgent
        return DQNAgent
    if name == 'ReplayBuffer':
        from .dqn_agent import ReplayBuffer
        return ReplayBuffer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__

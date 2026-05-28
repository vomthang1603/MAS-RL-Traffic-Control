# Lazy exports so Q-learning / evaluate paths work without TensorFlow (DQN only).

__all__ = ['train_q', 'evaluate_q', 'train_dqn', 'evaluate_dqn', 'benchmark']


def __getattr__(name: str):
    if name == 'train_q':
        from .train_q import train as train_q
        return train_q
    if name == 'evaluate_q':
        from .train_q import evaluate as evaluate_q
        return evaluate_q
    if name == 'train_dqn':
        from .train_dqn import train as train_dqn
        return train_dqn
    if name == 'evaluate_dqn':
        from .train_dqn import evaluate as evaluate_dqn
        return evaluate_dqn
    if name == 'benchmark':
        from .evaluate import benchmark
        return benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__

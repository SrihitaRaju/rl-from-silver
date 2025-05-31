import numpy as np


def decay_schedule(
    init_value: float,
    min_value: float,
    decay_ratio: float,
    max_steps: int,
    log_start: float = -2,
    log_base: float = 10,
) -> np.ndarray:
    """
    Creates a decay schedule that smoothly transitions from init_value to min_value.

    Args:
        init_value: Initial value of the parameter
        min_value: Minimum value to decay to
        decay_ratio: Proportion of max_steps over which to decay
        max_steps: Total number of steps
        log_start: Starting point for logarithmic decay (default: -2)
        log_base: Base for logarithmic decay (default: 10)

    Returns:
        Array of values that smoothly decay from init_value to min_value
    """
    # Calculate number of steps for decay
    decay_steps = int(max_steps * decay_ratio)
    remaining_steps = max_steps - decay_steps

    # Generate logarithmic decay values
    decay_values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[
        ::-1
    ]

    # Normalize decay values to [0, 1]
    decay_values = (decay_values - decay_values.min()) / (
        decay_values.max() - decay_values.min()
    )

    # Scale decay values to [min_value, init_value]
    decay_values = (init_value - min_value) * decay_values + min_value

    # Pad remaining steps with min_value
    return np.pad(decay_values, (0, remaining_steps), "edge")


# Example usage
if __name__ == "__main__":
    # Example: Decay epsilon from 1.0 to 0.01 over 1000 steps
    epsilon_schedule = decay_schedule(
        init_value=1.0, min_value=0.01, decay_ratio=0.8, max_steps=1000
    )

    # Plot the decay schedule
    import matplotlib.pyplot as plt

    plt.plot(epsilon_schedule)
    plt.title("Epsilon Decay Schedule")
    plt.xlabel("Step")
    plt.ylabel("Epsilon")
    plt.grid(True)
    plt.show()

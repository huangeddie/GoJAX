import jax.numpy as jnp
from jax import lax


def new_state(board_size, batch_size=1, squeeze=True):
    """
    :param board_size:
    :param batch_size:
    :param squeeze: whether to remove the batch dimension if it's 1
    :return: Generates new JAX array representing new Go game
    """
    state = jnp.zeros((batch_size, 6, board_size, board_size), dtype=bool)
    if squeeze and batch_size <= 1:
        return lax.squeeze(state, dimensions=[0])
    return state

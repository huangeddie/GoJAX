import jax.numpy as jnp
from jax import lax
from jax import jit
from functools import partial


@partial(jit, static_argnums=[0, 1, 2])
def new_states(board_size, batch_size=1):
    """
    :param board_size:
    :param batch_size:
    :return: Generates new JAX array representing new Go game
    """
    state = jnp.zeros((batch_size, 6, board_size, board_size), dtype=bool)
    return state


@jit
def next_states(states, indicator_actions):
    """
    :param states:
    :param indicator_actions: A sparse array of the same shape that represents the new actions. For each state
    in the batch, there should be at most one non-zero element representing the move. If all elements are 0,
    then it's considered a pass. :return: The next states of the board
    """
    next_states = lax.max(states, indicator_actions)
    return next_states.at[:, 2].set(True)


def to_indicator_actions(actions, states):
    """
    :param actions: A list of actions. Each element is either pass (None), or a tuple of integers representing a row, column coordinate.
    :param states: The corresponding list of states.
    :return: A sparse array representing indicator actions for each state.
    """
    turns = get_turns(states)
    indicator_actions = jnp.zeros_like(states)
    for i, (action, turn) in enumerate(zip(actions, turns)):
        if action is None:
            continue
        indicator_actions = indicator_actions.at[i, jnp.int8(turn), action[0], action[1]].set(True)
    return indicator_actions


def get_turns(states):
    """
    :param states:
    :return: A boolean list indicating whose turn it is for each state
    """
    return list(map(lambda s: jnp.all(s[2] == 1), states))

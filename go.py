from functools import partial

import jax.numpy as jnp
from jax import jit
from jax import lax

import go_constants


@partial(jit, static_argnums=[0, 1, 2])
def new_states(board_size, batch_size=1):
    """
    :param board_size:
    :param batch_size:
    :return: Generates new JAX array representing new Go game
    """
    state = jnp.zeros((batch_size, go_constants.NUM_CHANNELS, board_size, board_size), dtype=bool)
    return state


@jit
def next_states(states, indicator_actions):
    """
    :param states:
    :param indicator_actions: A sparse array of the same shape as states that represents the new actions. For each state
    in the batch, there should be at most one non-zero element representing the move. If all elements are 0,
    then it's considered a pass. :return: The next states of the board
    """
    states = lax.max(states, indicator_actions)
    # Change the turn
    states = states.at[:, go_constants.TURN_CHANNEL_INDEX].set(~states[:, go_constants.TURN_CHANNEL_INDEX])

    # Get passed states
    previously_passed = states[:, go_constants.PASS_CHANNEL_INDEX]
    passed = jnp.alltrue(lax.eq(indicator_actions, jnp.zeros_like(indicator_actions)), axis=(1, 2, 3))

    # Set pass
    states = states.at[:, go_constants.PASS_CHANNEL_INDEX].set(passed)

    # Set game ended
    states = states.at[:, go_constants.END_CHANNEL_INDEX].set(previously_passed & passed)

    return states


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


def decode_state(encode_str: str, turn: bool = go_constants.BLACKS_TURN, passed: bool = False, komi=None):
    """
    Creates a game board from the human-readable encoded string.
    :param encode_str:
    :param turn:
    :param passed:
    :param komi:
    :return:
    """
    if encode_str[0] == '\n':
        encode_str = encode_str[1:]
    if encode_str[-1] == '\n':
        encode_str = encode_str[:-1]
    lines = encode_str.splitlines()
    board_size = len(lines)
    state = new_states(board_size, batch_size=1)
    for i, line in enumerate(lines):
        for j, char in enumerate(line.split()):
            if char == 'B':
                state = state.at[0, go_constants.BLACK_CHANNEL_INDEX, i, j].set(True)
            elif char == 'W':
                state = state.at[0, go_constants.WHITE_CHANNEL_INDEX, i, j].set(True)

    # Set the turn
    state = state.at[0, go_constants.TURN_CHANNEL_INDEX].set(turn)

    # TODO: Set the invalid channel
    state = state.at[0, go_constants.INVALID_CHANNEL_INDEX].set(
        jnp.zeros_like(state[0, go_constants.INVALID_CHANNEL_INDEX]))

    # Set if passed
    if passed:
        state = state.at[0, go_constants.PASS_CHANNEL_INDEX].set(True)

    return state

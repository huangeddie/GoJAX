import textwrap
from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
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


def get_at_pieces_per_turn(states, turns):
    return states.at[jnp.arange(states.shape[0]), jnp.array(turns, dtype=int)]


def get_pieces_per_turn(states, turns):
    return states[jnp.arange(states.shape[0]), jnp.array(turns, dtype=int)]


def next_states(states, indicator_actions):
    """
    :param states:
    :param indicator_actions: A (N x B x B) sparse array of the same shape as states that represents the new actions. For each state
    in the batch, there should be at most one non-zero element representing the move. If all elements are 0,
    then it's considered a pass. :return: The next states of the board
    """
    turns = get_turns(states)
    opponents = ~turns

    # Add the piece
    states = get_at_pieces_per_turn(states, turns).max(indicator_actions)

    # Remove trapped pieces
    states = get_at_pieces_per_turn(states, opponents).set(get_free_groups(states, opponents))

    # Change the turn
    states = states.at[:, go_constants.TURN_CHANNEL_INDEX].set(~states[:, go_constants.TURN_CHANNEL_INDEX])

    # Get passed states
    previously_passed = jnp.alltrue(states[:, go_constants.PASS_CHANNEL_INDEX], axis=(1, 2), keepdims=True)
    passed = jnp.alltrue(~indicator_actions, axis=(1, 2), keepdims=True)

    # Set pass
    states = states.at[:, go_constants.PASS_CHANNEL_INDEX].set(passed)

    # Set invalid moves
    states = states.at[:, go_constants.INVALID_CHANNEL_INDEX].set(
        jnp.sum(states[:, [go_constants.BLACK_CHANNEL_INDEX, go_constants.WHITE_CHANNEL_INDEX]], axis=1, dtype=bool))

    # Set game ended
    states = states.at[:, go_constants.END_CHANNEL_INDEX].set(previously_passed & passed)

    return states


def to_indicator_actions(actions, states):
    """
    :param actions: A list of actions. Each element is either pass (None), or a tuple of integers representing a row, column coordinate.
    :param states: The corresponding list of states.
    :return: A (N x B x B) sparse array representing indicator actions for each state.
    """
    indicator_actions = jnp.zeros((states.shape[0], states.shape[2], states.shape[3]), dtype=bool)
    for i, action in enumerate(actions):
        if action is None:
            continue
        indicator_actions = indicator_actions.at[i, action[0], action[1]].set(True)
    return indicator_actions


def get_turns(states):
    """
    :param states:
    :return: A boolean list indicating whose turn it is for each state
    """

    return jnp.alltrue(states[:, go_constants.TURN_CHANNEL_INDEX], axis=(1, 2))


def decode_state(encode_str: str, turn: bool = go_constants.BLACKS_TURN, passed: bool = False, komi=None):
    """
    Creates a game board from the human-readable encoded string.
    :param encode_str:
    :param turn:
    :param passed:
    :param komi:
    :return:
    """
    encode_str = textwrap.dedent(encode_str)
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


def get_free_groups(states, turns):
    pieces = get_pieces_per_turn(states, turns)
    free_spaces = ~jnp.sum(states[:, [0, 1]], axis=1, dtype=bool)
    kernel = jnp.array([[[False, True, False],
                         [True, True, True],
                         [False, True, False]]])
    free_pieces = jnp.logical_and(jsp.signal.convolve(free_spaces, kernel, mode='same'), pieces)
    next_free_pieces = jnp.logical_and(jsp.signal.convolve(free_pieces, kernel, mode='same'), pieces)

    last_two_states_free_pieces = jnp.stack([free_pieces, next_free_pieces], axis=1)

    def cond_fun(x):
        return jnp.any(x[:, 0] != x[:, 1])

    def body_fun(x):
        x = x.at[:, 0].set(x[:, 1])  # Copy the second state to the first state
        return x.at[:, 1].set(jnp.logical_and(jsp.signal.convolve(x[:, 1], kernel, mode='same'), pieces))

    return lax.while_loop(cond_fun, body_fun, last_two_states_free_pieces)[:, 1]

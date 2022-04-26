from gojax import constants
from jax import numpy as jnp


def get_pieces_per_turn(states, turns):
    """
    Slices the black/white pieces of the states.

    See `at_pieces_per_turn` to get an update reference view of the pieces.

    :param states: an array of N Go games.
    :param turns: a boolean array of length N indicating which pieces to reference per state.
    :return: an array of shape N x B x B.
    """
    return states[jnp.arange(states.shape[0]), jnp.array(turns, dtype=int)]


def get_turns(states):
    """
    Gets the turn for each state in states.

    :param states: a batch array of N Go games.
    :return: a boolean array of length N indicating whose turn it is for each state.
    """

    return jnp.alltrue(states[:, constants.TURN_CHANNEL_INDEX], axis=(1, 2))


def get_invalids(states):
    """
    Gets the invalid moves for each state in states.

    :param states: a batch array of N Go games.
    :return: an N x B x B boolean array.
    """

    return states[:, constants.INVALID_CHANNEL_INDEX]


def get_passes(states):
    """
    Gets passes for each state in states.

    :param states: a batch array of N Go games.
    :return: a boolean array of length N indicating which state was passed.
    """

    return jnp.alltrue(states[:, constants.PASS_CHANNEL_INDEX], axis=(1, 2))


def get_ended(states):
    """
    Indicates which states have ended.

    :param states: a batch array of N Go games.
    :return: a boolean array of length N indicating which state ended.
    """
    return jnp.alltrue(states[:, constants.END_CHANNEL_INDEX], axis=(1, 2))


def get_empty_spaces(states):
    """
    Gets the empty spaces for each state.

    :param states: a batch array of N Go games.
    :return: an N x B x B boolean array.
    """
    return ~jnp.sum(states[:, [constants.BLACK_CHANNEL_INDEX, constants.WHITE_CHANNEL_INDEX]], axis=1, dtype=bool)


def get_occupied_spaces(states):
    """
    Gets the occupied spaces for each state (i.e. any black or white piecee).

    :param states: a batch array of N Go games.
    :return: an N x B x B boolean array.
    """
    return jnp.sum(states[:, [constants.BLACK_CHANNEL_INDEX, constants.WHITE_CHANNEL_INDEX]], axis=1, dtype=bool)

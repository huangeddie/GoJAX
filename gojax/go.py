"""Main Go game functions."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from gojax import constants
from gojax import state_index


def new_states(board_size, batch_size=1):
    """
    Returns a batch array of new Go games.

    :param board_size: board size (B).
    :param batch_size: batch size (N).
    :return: An N x 6 x B x B JAX zero-array of representing new Go games.
    """
    state = jnp.zeros((batch_size, constants.NUM_CHANNELS, board_size, board_size), dtype=bool)
    return state


def paint_fill(seeds, areas):
    """
    Paint fills the seeds to expand as much area as they can expand to in all 4 cardinal directions.

    Analogous to the Microsoft paint fill feature.

    Note that the seeds must intersect a location of an area in order to fill it. It cannot be
    adjacent to an area.

    :param seeds: an N x 1 x B x B boolean array where the True entries are the seeds.
    :param areas: an N x 1 x B x B boolean array where the True entries are areas.
    :return: an N x 1 x B x B boolean array.
    """
    second_expansion = jnp.logical_and(
        lax.conv(seeds.astype('bfloat16'), constants.CARDINALLY_CONNECTED_KERNEL,
                 window_strides=(1, 1), padding='same'), areas)

    def _last_expansion_no_change(last_two_expansions_):
        return jnp.any(last_two_expansions_[0] != last_two_expansions_[1])

    def _expand(last_two_expansions_):
        return last_two_expansions_[1], jnp.logical_and(
            lax.conv(last_two_expansions_[1].astype('bfloat16'),
                     constants.CARDINALLY_CONNECTED_KERNEL, window_strides=(1, 1), padding='same'),
            areas)

    return lax.while_loop(_last_expansion_no_change, _expand, (seeds, second_expansion))[1]


def compute_free_groups(states, turns):
    """
    Computes the free groups for each turn in the state of states.

    Free groups are the opposite of surrounded groups which are to be removed.

    :param states: a batch array of N Go games.
    :param turns: a boolean array of length N.
    :return: an N x B x B boolean array.
    """
    pieces = jnp.expand_dims(state_index.get_pieces_per_turn(states, turns), 1)
    empty_spaces = state_index.get_empty_spaces(states, keepdims=True)  # N x 1 x B x B array.
    immediate_free_pieces = jnp.logical_and(
        lax.conv(empty_spaces.astype('bfloat16'), constants.CARDINALLY_CONNECTED_KERNEL, (1, 1),
                 padding='same'), pieces)

    return jnp.squeeze(paint_fill(immediate_free_pieces, pieces), 1)


def compute_areas(states):
    """
    Compute the black and white areas of the states.

    An area is defined as the set of points where the point is either the player's piece or part
    of an empty group that
    is completely surrounded by the player's pieces (i.e. is not adjacent to any of the
    opponent's pieces).

    :param states: a batch array of N Go games.
    :return: an N x 2 x B x B boolean array, where the 0th and 1st indices of the 2nd dimension
    represent the black and
    white areas respectively.
    """
    black_pieces = states[:, constants.BLACK_CHANNEL_INDEX]
    white_pieces = states[:, constants.WHITE_CHANNEL_INDEX]
    empty_spaces = state_index.get_empty_spaces(states, keepdims=True)

    immediately_connected_to_black_pieces = jnp.logical_and(
        lax.conv(jnp.expand_dims(black_pieces, 1).astype('bfloat16'),
                 constants.CARDINALLY_CONNECTED_KERNEL, (1, 1), padding="same"), empty_spaces)
    immediately_connected_to_white_pieces = jnp.logical_and(
        lax.conv(jnp.expand_dims(white_pieces, 1).astype('bfloat16'),
                 constants.CARDINALLY_CONNECTED_KERNEL, (1, 1), padding="same"), empty_spaces)
    connected_to_black_pieces = paint_fill(immediately_connected_to_black_pieces, empty_spaces)
    connected_to_white_pieces = paint_fill(immediately_connected_to_white_pieces, empty_spaces)

    connected_to_pieces = jnp.concatenate((connected_to_black_pieces, connected_to_white_pieces), 1)
    pieces = states[:, (constants.BLACK_CHANNEL_INDEX, constants.WHITE_CHANNEL_INDEX)]
    return jnp.logical_or(jnp.logical_and(connected_to_pieces, ~connected_to_pieces[:, ::-1]),
                          pieces)


def compute_area_sizes(states):
    """
    Compute the size of the black and white areas (i.e. the number of pieces and empty spaces
    controlled by each player).

    :param states: a batch array of N Go games.
    :return: an N x 2 integer array.
    """
    return jnp.sum(compute_areas(states), axis=(2, 3))


def compute_winning(states):
    """
    Computes which player has the higher amount of area.

    1 = black is winning
    0 = tie
    -1 = white is winning

    :param states: a batch array of N Go games.
    :return: an N integer array.
    """
    return lax.clamp(-1,
                     -jnp.squeeze(jnp.diff(jnp.sum(compute_areas(states), axis=(2, 3))), axis=1), 1)


def compute_indicator_actions_are_invalid(states, indicator_actions):
    """
    Computes whether the given actions are valid for each state.

    An action is invalid if any of the following are met:
    • The space is occupied by a piece.
    • The action does not remove any opponent groups and the resulting group has no liberties.
    • The move is blocked by Komi.

    Komi is defined as a special type of invalid move where the following criteria are met:
    • The previous move by the opponent killed exactly one of our pieces.
    • The move would 'revive' said single killed piece, that is the move is the same location
    where our piece died.
    • The move would kill exactly one of the opponent's pieces.

    :param states: a batch array of N Go games.
    :param indicator_actions: an N x B x B partial one-hot boolean array of actions.
    the action would be `row x B + col`. The actions are in 1D form so that this function can be
    `jax.vmap`-ed.
    :return:
        • a boolean array of length N indicating whether the moves are invalid,
        • a batch array of N partial next Go states with the piece set, opponents removed,
        and killed channel set.
            • would need to update the turn, pass, and end channels.
    """
    turns = state_index.get_turns(states)
    opponents = ~turns
    piece_added = state_index.at_pieces_per_turn(states, turns).set(
        indicator_actions | state_index.get_pieces_per_turn(states, turns))
    piece_added_and_opponents_removed = state_index.at_pieces_per_turn(piece_added, opponents).set(
        compute_free_groups(piece_added, opponents))
    ghost_killed = jnp.logical_xor(state_index.get_pieces_per_turn(piece_added, opponents),
                                   state_index.get_pieces_per_turn(
                                       piece_added_and_opponents_removed, opponents))
    previously_killed_pieces = states[:, constants.KILLED_CHANNEL_INDEX]
    num_casualties = jnp.sum(previously_killed_pieces, axis=(1, 2))
    num_ghost_kills = jnp.sum(ghost_killed, axis=(1, 2))
    komi = (num_ghost_kills == 1) & jnp.sum(previously_killed_pieces & indicator_actions,
                                            axis=(1, 2)) & (num_casualties == 1)
    occupied = jnp.sum(
        jnp.sum(states[:, [constants.BLACK_CHANNEL_INDEX, constants.WHITE_CHANNEL_INDEX]], axis=1,
                dtype=bool) & indicator_actions, axis=(1, 2))
    no_liberties = jnp.sum(
        jnp.logical_xor(compute_free_groups(piece_added_and_opponents_removed, turns),
                        state_index.get_pieces_per_turn(piece_added_and_opponents_removed, turns)),
        axis=(1, 2), dtype=bool)

    partial_next_states = piece_added_and_opponents_removed.at[:,
                          constants.KILLED_CHANNEL_INDEX].set(ghost_killed)
    return jnp.logical_or(jnp.logical_or(occupied, no_liberties), komi), partial_next_states


def compute_actions1d_are_invalid(states, actions_1d):
    """
    Computes whether the given actions are valid for each state.

    An action is invalid if any of the following are met:
    • The space is occupied by a piece.
    • The action does not remove any opponent groups and the resulting group has no liberties.
    • The move is blocked by Komi.

    Komi is defined as a special type of invalid move where the following criteria are met:
    • The previous move by the opponent killed exactly one of our pieces.
    • The move would 'revive' said single killed piece, that is the move is the same location
    where our piece died.
    • The move would kill exactly one of the opponent's pieces.

    :param states: a batch array of N Go games.
    :param actions_1d: 1D action index. For a given action `(row, col)` in a Go game with board
    size `B`, the 1D form of
    the action would be `row x B + col`. The actions are in 1D form so that this function can be
    `jax.vmap`-ed.
    previous state.
    :return: a boolean array of length N indicating whether the moves are invalid.
    """
    rows = jnp.floor_divide(actions_1d, states.shape[2])
    cols = jnp.remainder(actions_1d, states.shape[3])
    n_indices = jnp.arange(len(states))
    passed = (actions_1d == np.prod(states.shape[-2:]))
    turns = state_index.get_turns(states)
    opponents = ~turns
    piece_added = states.at[n_indices, turns.astype('uint8'), rows, cols].set(
        ~passed | states[n_indices, turns.astype('uint8'), rows, cols])
    piece_added_and_opponents_removed = state_index.at_pieces_per_turn(piece_added, opponents).set(
        compute_free_groups(piece_added, opponents))
    ghost_killed = jnp.logical_xor(state_index.get_pieces_per_turn(piece_added, opponents),
                                   state_index.get_pieces_per_turn(
                                       piece_added_and_opponents_removed, opponents))
    previously_killed_pieces = states[:, constants.KILLED_CHANNEL_INDEX]
    num_casualties = jnp.sum(previously_killed_pieces, axis=(1, 2))
    num_ghost_kills = jnp.sum(ghost_killed, axis=(1, 2))
    komi = (num_ghost_kills == 1) & previously_killed_pieces[n_indices, rows, cols] & ~passed & (
            num_casualties == 1)
    occupied = \
        jnp.sum(states[:, [constants.BLACK_CHANNEL_INDEX, constants.WHITE_CHANNEL_INDEX]], axis=1,
                dtype=bool)[n_indices, rows, cols] & ~passed
    no_liberties = jnp.sum(
        jnp.logical_xor(compute_free_groups(piece_added_and_opponents_removed, turns),
                        state_index.get_pieces_per_turn(piece_added_and_opponents_removed, turns)),
        axis=(1, 2), dtype=bool)
    partial_next_states = piece_added_and_opponents_removed.at[:,
                          constants.KILLED_CHANNEL_INDEX].set(ghost_killed)
    return jnp.logical_or(jnp.logical_or(occupied, no_liberties), komi), partial_next_states


def compute_invalid_actions(states):
    """
    Computes the invalid moves for the turns of each state.

    :param states: a batch of N Go games.
    :return: an N x B x B indicator array of invalid moves.
    """

    invalid_moves, _ = jax.vmap(compute_actions1d_are_invalid, (None, 0), 1)(states, jnp.arange(
        states.shape[2] * states.shape[3]))
    return jnp.reshape(invalid_moves, (states.shape[0], states.shape[2], states.shape[3]))


def next_states(states, indicator_actions):
    """
    Compute the next batch of states in Go.

    :param states: a batch array of N Go games.
    :param indicator_actions: A (N x B x B) indicator array. For each state
    in the batch, there should be at most one non-zero element representing the move. If all
    elements are 0,
    then it's considered a pass.
    :return: an N x C x B x B boolean array.
    """
    invalid_actions, partial_next_states = compute_indicator_actions_are_invalid(states,
                                                                                 indicator_actions)

    # Set turn, pass, and end channels.
    partial_next_states = _change_turns(partial_next_states)
    previously_passed = jnp.alltrue(partial_next_states[:, constants.PASS_CHANNEL_INDEX],
                                    axis=(1, 2), keepdims=True)
    passed = jnp.alltrue(~indicator_actions, axis=(1, 2), keepdims=True)
    partial_next_states = partial_next_states.at[:, constants.PASS_CHANNEL_INDEX].set(passed)
    next_states_ = partial_next_states.at[:, constants.END_CHANNEL_INDEX].set(
        previously_passed & passed)

    # If the action is invalid or the game ended, set the move to pass, otherwise return what
    # would be the next state.
    return jnp.where(jnp.expand_dims(invalid_actions | state_index.get_ended(states), (1, 2, 3)),
                     _change_turns(states).at[:, constants.PASS_CHANNEL_INDEX].set(True),
                     next_states_)


def next_states_v2(states, actions_1d):
    """
    Compute the next batch of states in Go.

    :param states: a batch array of N Go games.
    :param actions_1d: An array of N integers in range [0, B^2].
    :return: an N x C x B x B boolean array.
    """
    invalid_actions, partial_next_states = compute_actions1d_are_invalid(states, actions_1d)

    # Set turn, pass, and end channels.
    partial_next_states = _change_turns(partial_next_states)
    previously_passed = jnp.alltrue(partial_next_states[:, constants.PASS_CHANNEL_INDEX],
                                    axis=(1, 2), keepdims=True)
    passed = jnp.expand_dims(actions_1d == np.prod(states.shape[-2:]), (1, 2))
    partial_next_states = partial_next_states.at[:, constants.PASS_CHANNEL_INDEX].set(passed)
    next_states_ = partial_next_states.at[:, constants.END_CHANNEL_INDEX].set(
        previously_passed & passed)

    # If the action is invalid or the game ended, set the move to pass, otherwise return what
    # would be the next state.
    return jnp.where(jnp.expand_dims(invalid_actions | state_index.get_ended(states), (1, 2, 3)),
                     _change_turns(states).at[:, constants.PASS_CHANNEL_INDEX].set(True),
                     next_states_)


def _change_turns(states):
    """
    Changes the turn for each state in states.

    :param states: a batch array of N Go games.
    :return: a boolean array with the same shape as states.
    """
    return states.at[:, constants.TURN_CHANNEL_INDEX].set(~states[:, constants.TURN_CHANNEL_INDEX])


def swap_perspectives(states):
    """
    Returns the same states but with the turns and pieces swapped.

    :param states: a batch array of N Go games.
    :return: a boolean array with the same shape as states.
    """
    swapped_pieces = states.at[:,
                     [constants.BLACK_CHANNEL_INDEX, constants.WHITE_CHANNEL_INDEX]].set(
        states[:, [constants.WHITE_CHANNEL_INDEX, constants.BLACK_CHANNEL_INDEX]])
    return _change_turns(swapped_pieces)

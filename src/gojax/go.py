import textwrap

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax

from gojax import constants
from gojax.state_info import get_pieces_per_turn, get_turns, get_invalids, get_ended, get_empty_spaces


def new_states(board_size, batch_size=1):
    """
    Returns a batch array of new Go games.

    :param board_size: board size (B).
    :param batch_size: batch size (N).
    :return: An N x 6 x B x B JAX zero-array of representing new Go games.
    """
    state = jnp.zeros((batch_size, constants.NUM_CHANNELS, board_size, board_size), dtype=bool)
    return state


def at_pieces_per_turn(states, turns):
    """
    Update reference to the black/white pieces of the states.

    See `get_pieces_per_turn` to get a read-only view of the pieces.

    :param states: an array of N Go games.
    :param turns: a boolean array of length N indicating which pieces to reference per state.
    :return: an update reference array of shape N x B x B.
    """
    return states.at[jnp.arange(states.shape[0]), jnp.array(turns, dtype=int)]


def at_location_per_turn(states, turns, row, col):
    """
    Update reference to specific turn-wise locations of the states.

    A more specific version of `at_pieces_per_turn`.

    :param states: an array of N Go games.
    :param turns: a boolean array of length N indicating which pieces to reference per state.
    :param row: integer row index.
    :param col: integer column index.
    :return: a scalar update reference.
    """
    return states.at[
        jnp.arange(states.shape[0]), jnp.array(turns, dtype=int), jnp.full(states.shape[0], row), jnp.full(
            states.shape[0], col)]


def to_indicator_actions(actions, states):
    """
    Converts a list of actions into their sparse indicator array form.

    :param actions: a list of N actions. Each element is either pass (None), or a tuple of integers representing a row,
    column coordinate.
    :param states: a batch array of N Go games.
    :return: a (N x B x B) sparse array representing indicator actions for each state.
    """
    indicator_actions = jnp.zeros(
        (states.shape[0], states.shape[2], states.shape[3]), dtype=bool)
    for i, action in enumerate(actions):
        if action is None:
            continue
        indicator_actions = indicator_actions.at[i, action[0], action[1]].set(
            True)
    return indicator_actions


def _paint_fill(seeds, areas):
    """
    Paint fills the seeds to expand as much area as they can expand to in all 4 cardinal directions.

    Analogous to the Microsoft paint fill feature.

    Note that the seeds must intersect a location of an area in order to fill it. It cannot be adjacent to an area.

    :param seeds: an (xN)* x B x B boolean array where the True entries are the seeds.
    :param areas: an (xN)* x B x B boolean array where the True entries are areas.
    :return: an (xN)* x B x B boolean array.
    """
    kernel = _get_cardinally_connected_kernel(jnp.ndim(seeds) - 2)
    second_expansion = jnp.logical_and(
        jsp.signal.convolve(seeds, kernel, mode='same'), areas)
    last_two_expansions = jnp.stack([seeds, second_expansion], axis=0)

    def _last_expansion_no_change(x):
        return jnp.any(x[0] != x[1])

    def _expand(x):
        x = x.at[0].set(x[1])  # Copy the second state to the first state
        return x.at[1].set(jnp.logical_and(jsp.signal.convolve(x[1], kernel, mode='same'), areas))

    return lax.while_loop(_last_expansion_no_change, _expand, last_two_expansions)[1]


def _get_cardinally_connected_kernel(num_extra_dims):
    """
    Returns a kernel used to in convolution used to expand a batch of 2D boolean arrays in all four cardinal directions.

    :param num_extra_dims: the number of extra dimensions before the 2D canvas.
    :return: an x[num_extra_dims] x 3 x 3 boolean array.
    """
    kernel = jnp.expand_dims(jnp.array([[False, True, False],
                                        [True, True, True],
                                        [False, True, False]]), tuple(range(num_extra_dims)))
    return kernel


def compute_free_groups(states, turns):
    """
    Computes the free groups for each turn in the state of states.

    Free groups are the opposite of surrounded groups which are to be removed.

    :param states: a batch array of N Go games.
    :param turns: a boolean array of length N.
    :return: an N x B x B boolean array.
    """
    pieces = get_pieces_per_turn(states, turns)
    empty_spaces = get_empty_spaces(states)
    kernel = _get_cardinally_connected_kernel(jnp.ndim(pieces) - 2)
    immediate_free_pieces = jnp.logical_and(
        jsp.signal.convolve(empty_spaces, kernel, mode='same'), pieces)

    return _paint_fill(immediate_free_pieces, pieces)


def compute_areas(states):
    """
    Compute the black and white areas of the states.

    An area is defined as the set of points where the point is either the player's piece or part of an empty group that
    is completely surrounded by the player's pieces (i.e. is not adjacent to any of the opponent's pieces).

    :param states: a batch array of N Go games.
    :return: an N x 2 x B x B boolean array, where the 0th and 1st indices of the 2nd dimension represent the black and
    white areas respectively.
    """
    pieces = states[:, (constants.BLACK_CHANNEL_INDEX, constants.WHITE_CHANNEL_INDEX)]
    kernel = _get_cardinally_connected_kernel(jnp.ndim(pieces) - 2)
    empty_spaces = get_empty_spaces(states, keepdims=True)

    immediately_connected_to_pieces = jnp.logical_and(jsp.signal.convolve(pieces, kernel, mode='same'),
                                                      empty_spaces)
    connected_to_pieces = _paint_fill(immediately_connected_to_pieces, empty_spaces)
    return jnp.logical_or(jnp.logical_and(connected_to_pieces, ~connected_to_pieces[:, ::-1]), pieces)


def compute_area_sizes(states):
    """
    Compute the size of the black and white areas (i.e. the number of pieces and empty spaces controlled by each player).

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
    return lax.clamp(-1, -jnp.squeeze(jnp.diff(jnp.sum(compute_areas(states), axis=(2, 3))), axis=1), 1)


def compute_actions_are_invalid(states, action_1d, my_killed_pieces):
    """
    Computes whether the given actions are valid for each state.

    An action is invalid if any of the following are met:
    • The space is occupied by a piece.
    • The action does not remove any opponent groups and the resulting group has no liberties.
    • The move is blocked by Komi.

    Komi is defined as a special type of invalid move where the following criteria are met:
    • The previous move by the opponent killed exactly one of our pieces.
    • The move would 'revive' said single killed piece, that is the move is the same location where our piece died.
    • The move would kill exactly one of the opponent's pieces.

    :param states: a batch array of N Go games.
    :param action_1d: 1D action index. For a given action `(row, col)` in a Go game with board size `B`, the 1D form of
    the action would be `row x B + col`. The actions are in 1D form so that this function can be `jax.vmap`-ed.
    :param my_killed_pieces: an N x B x B indicator array for pieces that were killed from the previous state.
    :return: a boolean array of length N indicating whether the moves are invalid.
    """
    row = jnp.floor_divide(action_1d, states.shape[2])
    col = jnp.remainder(action_1d, states.shape[3])
    turns = get_turns(states)
    opponents = ~turns
    ghost_next_states = at_location_per_turn(states, turns, row, col).set(True)
    ghost_maybe_kill = at_pieces_per_turn(ghost_next_states, opponents).set(
        compute_free_groups(ghost_next_states, opponents))
    ghost_killed = jnp.logical_xor(get_pieces_per_turn(ghost_next_states, opponents),
                                   get_pieces_per_turn(ghost_maybe_kill, opponents))
    num_casualties = jnp.sum(my_killed_pieces, axis=(1, 2))
    single_casualties = num_casualties == jnp.ones_like(num_casualties)
    move_is_revival = my_killed_pieces[:, row, col]
    num_ghost_kills = jnp.sum(ghost_killed, axis=(1, 2))
    single_ghost_kills = num_ghost_kills == jnp.ones_like(num_ghost_kills)
    komi = single_ghost_kills & move_is_revival & single_casualties
    occupied = jnp.sum(states[:, [constants.BLACK_CHANNEL_INDEX, constants.WHITE_CHANNEL_INDEX], row, col],
                       dtype=bool)
    no_liberties = jnp.sum(
        jnp.logical_xor(compute_free_groups(ghost_maybe_kill, turns),
                        get_pieces_per_turn(ghost_maybe_kill, turns)),
        axis=(1, 2), dtype=bool)
    return jnp.logical_or(jnp.logical_or(occupied, no_liberties), komi)


def compute_invalid_actions(states, my_killed_pieces):
    """
    Computes the invalid moves for the turns of each state.

    :param states: a batch of N Go games.
    :param my_killed_pieces: an N x B x B indicator array for pieces that were killed from the previous state.
    :return: an N x B x B indicator array of invalid moves.
    """

    invalid_moves = jax.vmap(compute_actions_are_invalid, (None, 0, None), 1)(states,
                                                                              jnp.arange(
                                                                                  states.shape[2] * states.shape[3]),
                                                                              my_killed_pieces)
    return jnp.reshape(invalid_moves, (states.shape[0], states.shape[2], states.shape[3]))


def _move_is_invalid(states, indicator_actions):
    """
    Gets whether the actions are invalid for the corresponding states.

    :param states: a batch array of N Go games.
    :param indicator_actions: A (N x B x B) indicator array. For each state
    in the batch, there should be at most one non-zero element representing the move. If all elements are 0,
    then it's considered a pass.
    :return: a boolean array of length N.
    """
    return jnp.sum(get_invalids(states) & indicator_actions, axis=(1, 2), dtype=bool)


def next_states(states, indicator_actions):
    """
    Compute the next batch of states in Go.

    :param states: a batch array of N Go games.
    :param indicator_actions: A (N x B x B) indicator array. For each state
    in the batch, there should be at most one non-zero element representing the move. If all elements are 0,
    then it's considered a pass.
    :return: an N x C x B x B boolean array.
    """
    # Get the players
    turns = get_turns(states)
    opponents = ~turns

    # Change the turn
    states = _change_turns(states)

    # Save the current state after changing the turn because we may need to revert back to this checkpoint for some of them.
    checkpoint_states = states

    # Add the piece
    states = at_pieces_per_turn(states, turns).max(indicator_actions)

    # Remove trapped pieces
    kill_pieces = jnp.logical_xor(get_pieces_per_turn(
        states, opponents), compute_free_groups(states, opponents))
    states = at_pieces_per_turn(states, opponents).set(
        compute_free_groups(states, opponents))

    # Get passed states
    previously_passed = jnp.alltrue(
        states[:, constants.PASS_CHANNEL_INDEX], axis=(1, 2), keepdims=True)
    passed = jnp.alltrue(~indicator_actions, axis=(1, 2), keepdims=True)

    # Set pass
    states = states.at[:, constants.PASS_CHANNEL_INDEX].set(passed)

    # Set invalid moves
    states = states.at[:, constants.INVALID_CHANNEL_INDEX].set(
        compute_invalid_actions(states, kill_pieces))

    # Set game ended
    states = states.at[:, constants.END_CHANNEL_INDEX].set(
        previously_passed & passed)

    # Revert back to original state if the original state already ended or the action is invalid
    states = jnp.where(jnp.expand_dims(get_ended(checkpoint_states) | _move_is_invalid(
        checkpoint_states, indicator_actions), (1, 2, 3)), checkpoint_states, states)

    return states


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
    swapped_pieces = states.at[:, [constants.BLACK_CHANNEL_INDEX, constants.WHITE_CHANNEL_INDEX]].set(
        states[:, [constants.WHITE_CHANNEL_INDEX, constants.BLACK_CHANNEL_INDEX]])
    return _change_turns(swapped_pieces)


def decode_state(encode_str: str, turn: bool = constants.BLACKS_TURN, passed: bool = False, komi=None,
                 ended: bool = False):
    """
    Creates a game board from the human-readable encoded string.

    Example encoding:
    ```
    B W
    W _
    ```

    :param encode_str: string representation of the Go game.
    :param turn: boolean turn indicator.
    :param passed: boolean indicator if the previous move was passed.
    :param komi: 2d action (tuple of 2 integers) or None.
    :param ended: whether the game ended.
    :return: a 1 x C x B X B boolean array.
    """
    encode_str = textwrap.dedent(encode_str)
    if encode_str[0] == '\n':
        encode_str = encode_str[1:]
    if encode_str[-1] == '\n':
        encode_str = encode_str[:-1]
    lines = encode_str.splitlines()
    board_size = len(lines)
    states = new_states(board_size, batch_size=1)
    for i, line in enumerate(lines):
        for j, char in enumerate(line.split()):
            if char == 'B':
                states = states.at[0, constants.BLACK_CHANNEL_INDEX, i, j].set(
                    True)
            elif char == 'W':
                states = states.at[0, constants.WHITE_CHANNEL_INDEX, i, j].set(
                    True)

    # Set the turn
    states = states.at[0, constants.TURN_CHANNEL_INDEX].set(turn)

    # Set invalid moves
    states = states.at[:, constants.INVALID_CHANNEL_INDEX].set(
        compute_invalid_actions(states, jnp.zeros_like(states[:, constants.BLACK_CHANNEL_INDEX])))
    if komi:
        states = states.at[0, constants.INVALID_CHANNEL_INDEX,
                           komi[0], komi[1]].set(True)

    # Set passed
    states = states.at[0, constants.PASS_CHANNEL_INDEX].set(passed)

    # Set ended
    states = states.at[0, constants.END_CHANNEL_INDEX].set(ended)

    return states


def get_pretty_string(state):
    """
    Creates a human-friendly string of the given state.

    :param state: C x B x B boolean array .
    :return: string representing the state.
    """
    board_str = ''

    size = state.shape[1]
    board_str += '\t'
    for i in range(size):
        board_str += '{}'.format(i).ljust(2, ' ')
    board_str += '\n'
    for i in range(size):
        board_str += '{}\t'.format(i)
        for j in range(size):
            if state[0, i, j] == 1:
                board_str += '○'
                if j != size - 1:
                    if i == 0 or i == size - 1:
                        board_str += '═'
                    else:
                        board_str += '─'
            elif state[1, i, j] == 1:
                board_str += '●'
                if j != size - 1:
                    if i == 0 or i == size - 1:
                        board_str += '═'
                    else:
                        board_str += '─'
            else:
                if i == 0:
                    if j == 0:
                        board_str += '╔═'
                    elif j == size - 1:
                        board_str += '╗'
                    else:
                        board_str += '╤═'
                elif i == size - 1:
                    if j == 0:
                        board_str += '╚═'
                    elif j == size - 1:
                        board_str += '╝'
                    else:
                        board_str += '╧═'
                else:
                    if j == 0:
                        board_str += '╟─'
                    elif j == size - 1:
                        board_str += '╢'
                    else:
                        board_str += '┼─'
        board_str += '\n'

    # TODO: Include empty spaces surrounded by either all black or all white pieces.
    black_area, white_area = jnp.sum(state[constants.BLACK_CHANNEL_INDEX]), jnp.sum(
        state[constants.WHITE_CHANNEL_INDEX])
    done = jnp.alltrue(state[constants.END_CHANNEL_INDEX])
    previous_player_passed = jnp.alltrue(state[constants.PASS_CHANNEL_INDEX])
    turn = jnp.alltrue(state[constants.TURN_CHANNEL_INDEX])
    if done:
        game_state = 'END'
    elif previous_player_passed:
        game_state = 'PASSED'
    else:
        game_state = 'ONGOING'
    board_str += f"\tTurn: {'BLACK' if turn == 0 else 'WHITE'}, Game State: {game_state}\n"
    board_str += f'\tBlack Area: {black_area}, White Area: {white_area}\n'
    return board_str

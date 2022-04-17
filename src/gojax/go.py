import textwrap

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax

from gojax import constants


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


def get_pieces_per_turn(states, turns):
    """
    Slices the black/white pieces of the states.

    See `at_pieces_per_turn` to get an update reference view of the pieces.

    :param states: an array of N Go games.
    :param turns: a boolean array of length N indicating which pieces to reference per state.
    :return: an array of shape N x B x B.
    """
    return states[jnp.arange(states.shape[0]), jnp.array(turns, dtype=int)]


def to_indicator_actions(actions, states):
    """
    Converts a list of actions into their sparse indicator array form.

    :param actions: a list of N actions. Each element is either pass (None), or a tuple of integers representing a row,
    column coordinate.
    :param states: a batch array of N Go games.
    :return: a (N x B x B) sparse array representing indicator actions for each state.
    """
    indicator_actions = jnp.zeros((states.shape[0], states.shape[2], states.shape[3]), dtype=bool)
    for i, action in enumerate(actions):
        if action is None:
            continue
        indicator_actions = indicator_actions.at[i, action[0], action[1]].set(True)
    return indicator_actions


def get_turns(states):
    """
    Gets the turn for each state in states.

    :param states: a batch array of N Go games.
    :return: a boolean array of length N indicating whose turn it is for each state.
    """

    return jnp.alltrue(states[:, constants.TURN_CHANNEL_INDEX], axis=(1, 2))


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


def get_free_groups(states, turns):
    """
    Gets the free groups for each turn in the state of states.

    Free groups are the opposite of surrounded groups which are to be removed.

    :param states: a batch array of N Go games.
    :param turns: a boolean array of length N.
    :return: an N x B x B boolean array.
    """
    pieces = get_pieces_per_turn(states, turns)
    free_spaces = ~jnp.sum(states[:, [0, 1]], axis=1, dtype=bool)
    kernel = jnp.array([[[False, True, False],
                         [True, True, True],
                         [False, True, False]]])
    free_pieces = jnp.logical_and(jsp.signal.convolve(free_spaces, kernel, mode='same'), pieces)
    next_free_pieces = jnp.logical_and(jsp.signal.convolve(free_pieces, kernel, mode='same'), pieces)

    last_two_states_free_pieces = jnp.stack([free_pieces, next_free_pieces], axis=1)

    def _cond_fun(x):
        return jnp.any(x[:, 0] != x[:, 1])

    def _body_fun(x):
        x = x.at[:, 0].set(x[:, 1])  # Copy the second state to the first state
        return x.at[:, 1].set(jnp.logical_and(jsp.signal.convolve(x[:, 1], kernel, mode='same'), pieces))

    return lax.while_loop(_cond_fun, _body_fun, last_two_states_free_pieces)[:, 1]


def get_action_is_invalid(action_1d, states, my_killed_pieces):
    """
    Computes whether the given actions are valid for each state.

    Komi is defined as a special type of invalid move where the following criteria are met:
    • The previous move by the opponent killed exactly one of our pieces.
    • The move would 'revive' said single killed piece, that is the move is the same location where our piece died.
    • The move would kill exactly one of the opponent's pieces.

    :param action_1d: 1D action index. For a given action `(row, col)` in a Go game with board size `B`, the 1D form of
    the action would be `row x B + col`. The actions are in 1D form so that this function can be `jax.vmap`-ed.
    :param states: a batch array of N Go games.
    :param my_killed_pieces: an N x B x B indicator array for pieces that were killed from the previous state.
    :return: a boolean array of length N indicating whether the moves are invalid.
    """
    row = jnp.floor_divide(action_1d, states.shape[2])
    col = jnp.remainder(action_1d, states.shape[3])
    turns = get_turns(states)
    opponents = ~turns
    ghost_next_states = at_location_per_turn(states, turns, row, col).set(True)
    ghost_maybe_kill = at_pieces_per_turn(ghost_next_states, opponents).set(
        get_free_groups(ghost_next_states, opponents))
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
        jnp.logical_xor(get_free_groups(ghost_maybe_kill, turns), get_pieces_per_turn(ghost_maybe_kill, turns)),
        axis=(1, 2), dtype=bool)
    return jnp.logical_or(jnp.logical_or(occupied, no_liberties), komi)


def get_invalid_moves(states, my_killed_pieces):
    """
    Computes the invalid moves for the turns of each state.

    :param states: a batch of N Go games.
    :param my_killed_pieces: an N x B x B indicator array for pieces that were killed from the previous state.
    :return: an N x B x B indicator array of invalid moves.
    """

    invalid_moves = jax.vmap(get_action_is_invalid, (0, None, None), 1)(jnp.arange(states.shape[2] * states.shape[3]),
                                                                        states, my_killed_pieces)
    return jnp.reshape(invalid_moves, (states.shape[0], states.shape[2], states.shape[3]))


def next_states(states, indicator_actions):
    """
    Compute the next batch of states in Go.

    :param states: a batch array of N Go games.
    :param indicator_actions: A (N x B x B) indicator array. For each state
    in the batch, there should be at most one non-zero element representing the move. If all elements are 0,
    then it's considered a pass.
    :return: an N x C x B x B boolean array.
    """
    turns = get_turns(states)
    opponents = ~turns

    # Add the piece
    states = at_pieces_per_turn(states, turns).max(indicator_actions)

    # Remove trapped pieces
    kill_pieces = jnp.logical_xor(get_pieces_per_turn(states, opponents), get_free_groups(states, opponents))
    states = at_pieces_per_turn(states, opponents).set(get_free_groups(states, opponents))

    # Change the turn
    states = states.at[:, constants.TURN_CHANNEL_INDEX].set(~states[:, constants.TURN_CHANNEL_INDEX])

    # Get passed states
    previously_passed = jnp.alltrue(states[:, constants.PASS_CHANNEL_INDEX], axis=(1, 2), keepdims=True)
    passed = jnp.alltrue(~indicator_actions, axis=(1, 2), keepdims=True)

    # Set pass
    states = states.at[:, constants.PASS_CHANNEL_INDEX].set(passed)

    # Set invalid moves
    states = states.at[:, constants.INVALID_CHANNEL_INDEX].set(get_invalid_moves(states, kill_pieces))

    # Set game ended
    states = states.at[:, constants.END_CHANNEL_INDEX].set(previously_passed & passed)

    return states


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
    state = new_states(board_size, batch_size=1)
    for i, line in enumerate(lines):
        for j, char in enumerate(line.split()):
            if char == 'B':
                state = state.at[0, constants.BLACK_CHANNEL_INDEX, i, j].set(True)
            elif char == 'W':
                state = state.at[0, constants.WHITE_CHANNEL_INDEX, i, j].set(True)

    # Set the turn
    state = state.at[0, constants.TURN_CHANNEL_INDEX].set(turn)

    # Set invalid moves
    state = state.at[:, constants.INVALID_CHANNEL_INDEX].set(
        get_invalid_moves(state, jnp.zeros_like(state[:, constants.BLACK_CHANNEL_INDEX])))
    if komi:
        state = state.at[0, constants.INVALID_CHANNEL_INDEX, komi[0], komi[1]].set(True)

    # Set passed
    state = state.at[0, constants.PASS_CHANNEL_INDEX].set(passed)

    # Set ended
    state = state.at[0, constants.END_CHANNEL_INDEX].set(ended)

    return state


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

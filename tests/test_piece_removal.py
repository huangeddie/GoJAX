"""
Tests Go piece removal logic.

Piece removal is arguably the more complicated part of the Go game logic, so we decided it
deserved its own dedicated test file.
"""

# pylint: disable=missing-function-docstring,too-many-public-methods,no-self-use,duplicate-code

import unittest

import jax.numpy as jnp

import gojax


class PieceRemovalTestCase(unittest.TestCase):
    """Tests Go piece removal logic."""

    def test_single_piece(self):
        state_str = """
                    W _ _ _
                    B _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str, gojax.BLACKS_TURN)

        next_state = gojax.next_states(state, gojax.to_indicator_actions([(0, 1)], state))

        # Check that the white piece is gone and the black piece is added
        delta_board = jnp.logical_xor(next_state[0, [0, 1]], state[0, [0, 1]])
        # Only have two changes
        self.assertEqual(jnp.sum(delta_board), 2)
        # The black piece is added
        self.assertTrue(delta_board[gojax.BLACK_CHANNEL_INDEX, 0, 1])
        self.assertTrue(next_state[0, gojax.BLACK_CHANNEL_INDEX, 0, 1])
        # White piece removed
        self.assertTrue(delta_board[gojax.WHITE_CHANNEL_INDEX, 0, 0])
        self.assertFalse(next_state[0, gojax.WHITE_CHANNEL_INDEX, 0, 0])

    def test_two_connected_pieces(self):
        state_str = """
                    W W _ _
                    B B _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str, gojax.BLACKS_TURN)
        next_state = gojax.next_states(state, gojax.to_indicator_actions([(0, 2)], state))
        self.assertTrue(jnp.alltrue(~next_state[0, gojax.WHITE_CHANNEL_INDEX]))
        self.assertTrue(
            jnp.alltrue(
                next_state[0, gojax.BLACK_CHANNEL_INDEX] == jnp.array([[False, False, True, False],
                                                                       [True, True, False, False],
                                                                       [False, False, False, False],
                                                                       [False, False, False,
                                                                        False]])),
            next_state[0, gojax.BLACK_CHANNEL_INDEX])

    def test_two_disjoint_pieces(self):
        state_str = """
                    W _ W B
                    B _ B _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str, gojax.BLACKS_TURN)
        next_state = gojax.next_states(state, gojax.to_indicator_actions([(0, 1)], state))
        self.assertTrue(jnp.alltrue(~next_state[0, gojax.WHITE_CHANNEL_INDEX]))
        self.assertTrue(
            jnp.alltrue(
                next_state[0, gojax.BLACK_CHANNEL_INDEX] == jnp.array([[False, True, False, True],
                                                                       [True, False, True, False],
                                                                       [False, False, False, False],
                                                                       [False, False, False,
                                                                        False]])))

    def test_donut(self):
        state_str = """
                    B B B W
                    B _ B W
                    B B B W
                    W W W _
                    """
        state = gojax.decode_state(state_str, gojax.WHITES_TURN)
        next_state = gojax.next_states(state, gojax.to_indicator_actions([(1, 1)], state))
        self.assertTrue(jnp.alltrue(~next_state[0, gojax.BLACK_CHANNEL_INDEX]))
        self.assertTrue(
            jnp.alltrue(
                next_state[0, gojax.WHITE_CHANNEL_INDEX] == jnp.array([[False, False, False, True],
                                                                       [False, True, False, True],
                                                                       [False, False, False, True],
                                                                       [True, True, True, False]])))


if __name__ == '__main__':
    unittest.main()

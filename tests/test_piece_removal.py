import unittest

import jax.numpy as jnp

from gojax import go, constants


class PieceRemovalTestCase(unittest.TestCase):
    def test_single_piece(self):
        state_str = """
                    W _ _ _
                    B _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, constants.BLACKS_TURN)

        next_state = go.next_states(state, go.to_indicator_actions([(0, 1)], state))

        # Check that the white piece is gone and the black piece is added
        delta_board = jnp.logical_xor(next_state[0, [0, 1]], state[0, [0, 1]])
        # Only have two changes
        self.assertEqual(jnp.sum(delta_board), 2)
        # The black piece is added
        self.assertTrue(delta_board[constants.BLACK_CHANNEL_INDEX, 0, 1])
        self.assertTrue(next_state[0, constants.BLACK_CHANNEL_INDEX, 0, 1])
        # White piece removed
        self.assertTrue(delta_board[constants.WHITE_CHANNEL_INDEX, 0, 0])
        self.assertFalse(next_state[0, constants.WHITE_CHANNEL_INDEX, 0, 0])

    def test_two_connected_pieces(self):
        state_str = """
                    W W _ _
                    B B _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, constants.BLACKS_TURN)
        next_state = go.next_states(state, go.to_indicator_actions([(0, 2)], state))
        self.assertTrue(jnp.alltrue(~next_state[0, constants.WHITE_CHANNEL_INDEX]))
        self.assertTrue(
            jnp.alltrue(next_state[0, constants.BLACK_CHANNEL_INDEX] == jnp.array([[False, False, True, False],
                                                                                   [True, True, False, False],
                                                                                   [False, False, False, False],
                                                                                   [False, False, False, False]])),
            next_state[0, constants.BLACK_CHANNEL_INDEX])

    def test_two_disjoint_pieces(self):
        state_str = """
                    W _ W B
                    B _ B _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, constants.BLACKS_TURN)
        next_state = go.next_states(state, go.to_indicator_actions([(0, 1)], state))
        self.assertTrue(jnp.alltrue(~next_state[0, constants.WHITE_CHANNEL_INDEX]))
        self.assertTrue(
            jnp.alltrue(next_state[0, constants.BLACK_CHANNEL_INDEX] == jnp.array([[False, True, False, True],
                                                                                   [True, False, True, False],
                                                                                   [False, False, False, False],
                                                                                   [False, False, False, False]])))

    def test_donut(self):
        state_str = """
                    B B B W
                    B _ B W
                    B B B W
                    W W W _
                    """
        state = go.decode_state(state_str, constants.WHITES_TURN)
        next_state = go.next_states(state, go.to_indicator_actions([(1, 1)], state))
        self.assertTrue(jnp.alltrue(~next_state[0, constants.BLACK_CHANNEL_INDEX]))
        self.assertTrue(
            jnp.alltrue(next_state[0, constants.WHITE_CHANNEL_INDEX] == jnp.array([[False, False, False, True],
                                                                                   [False, True, False, True],
                                                                                   [False, False, False, True],
                                                                                   [True, True, True, False]])))


if __name__ == '__main__':
    unittest.main()

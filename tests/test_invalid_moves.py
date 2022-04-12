import unittest

from jax import numpy as jnp

import constants
import go


class InvalidMovesTestCase(unittest.TestCase):
    def test_space_occupied_by_opponent_pieces(self):
        state = go.new_states(2)
        next_state = go.next_states(state, go.to_indicator_actions([(0, 0)], state))
        self.assertTrue(next_state[0, constants.INVALID_CHANNEL_INDEX, 0, 0])

    def test_space_occupied_by_own_pieces(self):
        state = go.new_states(2)
        state = go.next_states(state, go.to_indicator_actions([(0, 0)], state))
        state = go.next_states(state, go.to_indicator_actions([None], state))
        self.assertTrue(state[0, constants.INVALID_CHANNEL_INDEX, 0, 0])

    def test_single_hole_no_liberties_manual_build(self):
        """
        X B _ _
        B _ _ _
        _ _ _ _
        _ _ _ _
        """
        state = go.new_states(4)
        state = go.next_states(state, go.to_indicator_actions([[0, 1]], state))
        state = go.next_states(state, go.to_indicator_actions([None], state))
        state = go.next_states(state, go.to_indicator_actions([[1, 0]], state))
        self.assertTrue(state[:, constants.INVALID_CHANNEL_INDEX, 0, 0])

    def test_single_hole_no_liberties(self):
        state_str = """
                    X B _ _
                    B _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, constants.WHITES_TURN)
        self.assertTrue(state[:, constants.INVALID_CHANNEL_INDEX, 0, 0])

    def test_no_liberties_connect_to_group(self):
        state_str = """
                    B X W _
                    W W _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, constants.BLACKS_TURN)
        self.assertTrue(state[0, constants.INVALID_CHANNEL_INDEX, 0, 1])

    def test_get_action_is_invalid_false(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, constants.BLACKS_TURN)
        action1d = 0
        my_killed_pieces = jnp.zeros((1, 4, 4), dtype=bool)
        self.assertFalse(go.get_action_is_invalid(action1d, state, my_killed_pieces))

    def test_get_action_is_invalid_komi(self):
        state_str = """
                    _ B W _
                    B W _ W
                    _ B W _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, constants.BLACKS_TURN)
        action1d = 6
        my_killed_pieces = jnp.array([[[False, False, False, False],
                                       [False, False, True, False],
                                       [False, False, False, False],
                                       [False, False, False, False]]])
        self.assertTrue(go.get_action_is_invalid(action1d, state, my_killed_pieces))

    def test_komi(self):
        state_str = """
                    _ B W _
                    B W _ W
                    _ B W _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, constants.BLACKS_TURN)
        next_state = go.next_states(state, go.to_indicator_actions([(1, 2)], state))
        self.assertTrue(next_state[:, constants.INVALID_CHANNEL_INDEX, 1, 1])


if __name__ == '__main__':
    unittest.main()

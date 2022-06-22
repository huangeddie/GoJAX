"""
Tests Go invalid move logic.

Computing invalid moves is arguably the more complicated part of the Go game logic, so we decided it
deserved its own dedicated test file.
"""

# pylint: disable=missing-function-docstring,too-many-public-methods,no-self-use,duplicate-code

import unittest

import numpy as np
from jax import numpy as jnp

import gojax
import serialize


class InvalidMovesTestCase(unittest.TestCase):
    """Tests Go invalid move logic."""

    def test_space_occupied_by_opponent_pieces(self):
        state = gojax.new_states(2)
        next_state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([(0, 0)], state))
        self.assertTrue(next_state[0, gojax.KILLED_CHANNEL_INDEX, 0, 0])

    def test_space_occupied_by_own_pieces(self):
        state = gojax.new_states(2)
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([(0, 0)], state))
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([None], state))
        self.assertTrue(state[0, gojax.KILLED_CHANNEL_INDEX, 0, 0])

    def test_single_hole_no_liberties_manual_build(self):
        """
        X B _ _
        B _ _ _
        _ _ _ _
        _ _ _ _
        """
        state = gojax.new_states(4)
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([[0, 1]], state))
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([None], state))
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([[1, 0]], state))
        self.assertTrue(state[:, gojax.KILLED_CHANNEL_INDEX, 0, 0])

    def test_single_hole_no_liberties(self):
        state_str = """
                    X B _ _
                    B _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = serialize.decode_states(state_str, gojax.WHITES_TURN)
        self.assertTrue(state[:, gojax.KILLED_CHANNEL_INDEX, 0, 0])

    def test_no_liberties_connect_to_group(self):
        state_str = """
                    B X W _
                    W W _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = serialize.decode_states(state_str, gojax.BLACKS_TURN)
        self.assertTrue(state[0, gojax.KILLED_CHANNEL_INDEX, 0, 1])

    def test_get_action_is_invalid_false(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = serialize.decode_states(state_str, gojax.BLACKS_TURN)
        action1d = 0
        my_killed_pieces = jnp.zeros((1, 4, 4), dtype=bool)
        self.assertFalse(gojax.compute_actions1d_are_invalid(state, action1d, my_killed_pieces))

    def test_get_action_is_invalid_komi(self):
        state_str = """
                    _ B W _
                    B W _ W
                    _ B W _
                    _ _ _ _
                    """
        state = serialize.decode_states(state_str, gojax.BLACKS_TURN)
        action1d = 6
        my_killed_pieces = jnp.array([[[False, False, False, False], [False, False, True, False],
                                       [False, False, False, False], [False, False, False, False]]])
        self.assertTrue(gojax.compute_actions1d_are_invalid(state, action1d, my_killed_pieces))

    def test_komi(self):
        state_str = """
                    _ B W _
                    B W _ W
                    _ B W _
                    _ _ _ _
                    """
        state = serialize.decode_states(state_str, gojax.BLACKS_TURN)
        next_state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([(1, 2)], state))
        self.assertTrue(next_state[:, gojax.KILLED_CHANNEL_INDEX, 1, 1])

    def test_invalid_move_no_op_pieces(self):
        state = serialize.decode_states("""
                                _ _ _
                                _ W _
                                _ _ _
                                """, gojax.BLACKS_TURN)
        next_state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([(1, 1)], state))
        np.testing.assert_array_equal(
            state[0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]],
            next_state[0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]])
        np.testing.assert_array_equal(gojax.get_turns(state), [gojax.BLACKS_TURN])
        np.testing.assert_array_equal(gojax.get_turns(next_state), [gojax.WHITES_TURN])

    def test_compute_invalid_actions_for_two_states(self):
        states = jnp.concatenate((serialize.decode_states("""
                                                          B _ _
                                                          _ _ _
                                                          _ _ _
                                                          """, turn=gojax.WHITES_TURN),
                                  serialize.decode_states("""
                                                          _ B _
                                                          _ _ _
                                                          _ _ _
                                                          """, turn=gojax.WHITES_TURN)), axis=0)
        np.testing.assert_array_equal(
            gojax.compute_invalid_actions(states, my_killed_pieces=jnp.zeros_like(states[:, 0])),
            [[[True, False, False], [False, False, False], [False, False, False]],
             [[False, True, False], [False, False, False], [False, False, False]]])


if __name__ == '__main__':
    unittest.main()

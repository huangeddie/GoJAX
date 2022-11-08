"""
Tests Go invalid move logic.

Computing invalid moves is arguably the more complicated part of the Go game logic, so we decided it
deserved its own dedicated test file.
"""

# pylint: disable=missing-function-docstring,too-many-public-methods,duplicate-code

import unittest

import numpy as np
from jax import numpy as jnp

import gojax




class InvalidMovesTestCase(unittest.TestCase):
    """Tests Go invalid move logic."""

    def test_compute_actions1d_are_invalid_false(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str, gojax.BLACKS_TURN)
        action1d = 0
        np.testing.assert_array_equal(gojax.compute_actions1d_are_invalid(state, action1d)[0],
                                      [False])

    def test_compute_actions1d_are_invalid_komi(self):
        state_str = """
                    _ B W _
                    B W X W
                    _ B W _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str, gojax.BLACKS_TURN)
        np.testing.assert_array_equal(gojax.compute_actions1d_are_invalid(state, actions_1d=6)[0],
                                      [True])

    def test_komi(self):
        state_str = """
                    _ B W _
                    B W _ W
                    _ B W _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str, gojax.BLACKS_TURN)
        next_state = gojax.next_states_legacy(state, gojax.action_2d_to_indicator([(1, 2)], state))
        self.assertTrue(next_state[:, gojax.KILLED_CHANNEL_INDEX, 1, 1])

    def test_invalid_move_no_op_pieces(self):
        state = gojax.decode_states("""
                                _ _ _
                                _ W _
                                _ _ _
                                """, gojax.BLACKS_TURN)
        next_state = gojax.next_states_legacy(state, gojax.action_2d_to_indicator([(1, 1)], state))
        np.testing.assert_array_equal(
            state[0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]],
            next_state[0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]])
        np.testing.assert_array_equal(gojax.get_turns(state), [gojax.BLACKS_TURN])
        np.testing.assert_array_equal(gojax.get_turns(next_state), [gojax.WHITES_TURN])
        np.testing.assert_array_equal(gojax.get_passes(next_state), [True])

    def test_compute_invalid_actions_for_two_states(self):
        states = jnp.concatenate((gojax.decode_states("""
                                                          B _ _
                                                          _ _ _
                                                          _ _ _
                                                          """, turn=gojax.WHITES_TURN),
                                  gojax.decode_states("""
                                                          _ B _
                                                          _ _ _
                                                          _ _ _
                                                          """, turn=gojax.WHITES_TURN)), axis=0)
        np.testing.assert_array_equal(gojax.compute_invalid_actions(states), [
            [[True, False, False], [False, False, False], [False, False, False]],
            [[False, True, False], [False, False, False], [False, False, False]]])


if __name__ == '__main__':
    unittest.main()

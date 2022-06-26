import unittest

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

import gojax


class StateIndexTestCase(chex.TestCase):
    @parameterized.named_parameters(
        {'testcase_name': 'pass', 'indicator_actions': [[[False, False], [False, False]]],
         'expected_indices': 4},
        {'testcase_name': 'first_index', 'indicator_actions': [[[True, False], [False, False]]],
         'expected_indices': 0}, {'testcase_name': 'pass_and_move',
                                  'indicator_actions': [[[False, False], [False, False]],
                                                        [[True, False], [False, False]]],
                                  'expected_indices': (4, 0)}, )
    def test_action_indicators_to_indices_(self, indicator_actions, expected_indices):
        np.testing.assert_array_equal(
            gojax.action_indicators_to_indices(jnp.array(indicator_actions)), expected_indices)

    @parameterized.named_parameters(
        {'testcase_name': 'small', 'board_size': 3, 'batch_size': 1, 'expected_action_size': 10},
        {'testcase_name': 'intermediate', 'board_size': 7, 'batch_size': 4,
         'expected_action_size': 50},
        {'testcase_name': 'big', 'board_size': 19, 'batch_size': 8, 'expected_action_size': 362}, )
    def test_get_action_size_(self, board_size, batch_size, expected_action_size):
        states = gojax.new_states(board_size, batch_size)
        self.assertEqual(gojax.get_action_size(states), expected_action_size)

    @parameterized.named_parameters(
        {'testcase_name': 'pass', 'states': gojax.new_states(board_size=2), 'actions': [None],
         'expected_output': jnp.zeros((1, 2, 2))},
        {'testcase_name': 'black_move', 'states': gojax.new_states(board_size=2),
         'actions': [(0, 0)], 'expected_output': [[[True, False], [False, False]]]},
        {'testcase_name': 'white_move',
         'states': gojax.new_states(board_size=2).at[:, gojax.TURN_CHANNEL_INDEX].set(
             gojax.WHITES_TURN), 'actions': [(0, 0)],
         'expected_output': [[[True, False], [False, False]]]},
        {'testcase_name': 'pass_and_move', 'states': gojax.new_states(board_size=2, batch_size=2),
         'actions': [None, (0, 0)],
         'expected_output': [[[False, False], [False, False]], [[True, False], [False, False]]]}, )
    def test_action2d_indices_to_indicator_(self, states, actions, expected_output):
        indicator_actions = gojax.action_2d_indices_to_indicator(actions, states)
        np.testing.assert_array_equal(indicator_actions, expected_output)
        chex.assert_type(indicator_actions, bool)

    def test_action_1d_indices_to_indicator(self):
        indicator_actions = gojax.action_1d_indices_to_indicator([5], gojax.new_states(board_size=3,
                                                                                       batch_size=1))
        np.testing.assert_array_equal(indicator_actions, [
            [[False, False, False], [False, False, True], [False, False, False]]])

    def test_action_1d_indices_to_indicator_pass_with_integer(self):
        indicator_actions = gojax.action_1d_indices_to_indicator([10],
                                                                 gojax.new_states(board_size=3,
                                                                                  batch_size=1))
        np.testing.assert_array_equal(indicator_actions, [
            [[False, False, False], [False, False, False], [False, False, False]]])

    def test_action_1d_indices_to_indicator_pass_with_none(self):
        indicator_actions = gojax.action_1d_indices_to_indicator([None],
                                                                 gojax.new_states(board_size=3,
                                                                                  batch_size=1))
        np.testing.assert_array_equal(indicator_actions, [
            [[False, False, False], [False, False, False], [False, False, False]]])

    def test_get_turns_channel_index(self):
        states = gojax.new_states(2, batch_size=2)
        states = states.at[0, gojax.TURN_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(gojax.get_turns(states) == jnp.array([True, False])))

    def test_get_killed_channel_index(self):
        states = gojax.new_states(2, batch_size=2)
        states = states.at[1, gojax.KILLED_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(gojax.get_killed(states) == jnp.array(
            [[[False, False], [False, False]], [[True, True], [True, True]]])))

    def test_get_passes_channel_index(self):
        states = gojax.new_states(2, batch_size=2)
        states = states.at[0, gojax.PASS_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(gojax.get_passes(states) == jnp.array([True, False])))

    def test_get_ended_channel_index(self):
        states = gojax.new_states(2, batch_size=2)
        states = states.at[0, gojax.END_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(gojax.get_ended(states) == jnp.array([True, False])))

    def test_get_occupied_spaces(self):
        state_str = """
            _ B _ _
            B _ B _
            _ B _ _
            _ _ W _
            """
        state = gojax.decode_states(state_str, komi=(0, 0))
        occupied_spaces = gojax.get_occupied_spaces(state)
        np.testing.assert_array_equal(occupied_spaces, [
            [[False, True, False, False], [True, False, True, False], [False, True, False, False],
             [False, False, True, False]]])

    def test_get_empty_spaces_shape_squeeze(self):
        empty_spaces = gojax.get_empty_spaces(gojax.new_states(board_size=3))
        chex.assert_shape(empty_spaces, (1, 3, 3))

    def test_get_empty_spaces_shape_keepdims(self):
        empty_spaces = gojax.get_empty_spaces(gojax.new_states(board_size=3), keepdims=True)
        chex.assert_shape(empty_spaces, (1, 1, 3, 3))

    def test_get_empty_spaces(self):
        state_str = """
            _ B _ _
            B _ B _
            _ B _ _
            _ _ W _
            """
        state = gojax.decode_states(state_str, komi=(0, 0))
        empty_spaces = gojax.get_empty_spaces(state)
        np.testing.assert_array_equal(empty_spaces, [
            [[True, False, True, True], [False, True, False, True], [True, False, True, True],
             [True, True, False, True]]])


if __name__ == '__main__':
    unittest.main()

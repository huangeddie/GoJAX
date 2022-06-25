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
    def test_action2d_indicies_to_indicator_(self, states, actions, expected_output):
        indicator_actions = gojax.action_2d_indices_to_indicator(actions, states)
        np.testing.assert_array_equal(indicator_actions, expected_output)
        chex.assert_type(indicator_actions, bool)


if __name__ == '__main__':
    unittest.main()

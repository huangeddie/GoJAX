"""Tests general Go functions."""

# pylint: disable=missing-function-docstring,too-many-public-methods,no-self-use,duplicate-code

import unittest

import chex
import gojax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax import lax
from jax import nn


class ActionIndicatorsToIndicesTestCase(chex.TestCase):
    @parameterized.named_parameters(
        {'testcase_name': 'pass', 'indicator_actions': [[[False, False],
                                                         [False, False]]], 'expected_indices': 4},
        {'testcase_name': 'first_index', 'indicator_actions': [[[True, False],
                                                                [False, False]]],
         'expected_indices': 0},
        {'testcase_name': 'pass_and_move', 'indicator_actions': [[[False, False],
                                                                  [False, False]],
                                                                 [[True, False],
                                                                  [False, False]]],
         'expected_indices': (4, 0)},
    )
    def test_(self, indicator_actions, expected_indices):
        np.testing.assert_array_equal(
            gojax.action_indicators_to_indices(jnp.array(indicator_actions)),
            expected_indices)


class NewStatesTestCase(chex.TestCase):
    @parameterized.named_parameters(
        {'testcase_name': 'board_size_3_batch_size_1', 'board_size': 3, 'batch_size': 1,
         'expected_output': jnp.zeros((1, gojax.NUM_CHANNELS, 3, 3), dtype=bool)},
        {'testcase_name': 'board_size_4_batch_size_1', 'board_size': 4, 'batch_size': 1,
         'expected_output': jnp.zeros((1, gojax.NUM_CHANNELS, 4, 4), dtype=bool)},
        {'testcase_name': 'board_size_3_batch_size_2', 'board_size': 3, 'batch_size': 2,
         'expected_output': jnp.zeros((2, gojax.NUM_CHANNELS, 3, 3), dtype=bool)},
    )
    def test_(self, board_size, batch_size, expected_output):
        new_states = gojax.new_states(board_size, batch_size)
        np.testing.assert_array_equal(new_states, expected_output)
        chex.assert_type(new_states, bool)


class Action2DIndicesToIndicatorTestCase(chex.TestCase):
    @parameterized.named_parameters(
        {'testcase_name': 'pass',
         'states': gojax.new_states(board_size=2),
         'actions': [None],
         'expected_output': jnp.zeros((1, 2, 2))},
        {'testcase_name': 'black_move',
         'states': gojax.new_states(board_size=2),
         'actions': [(0, 0)],
         'expected_output': [[[True, False],
                              [False, False]]]},
        {'testcase_name': 'white_move',
         'states': gojax.new_states(board_size=2).at[:, gojax.TURN_CHANNEL_INDEX].set(
             gojax.WHITES_TURN),
         'actions': [(0, 0)],
         'expected_output': [[[True, False],
                              [False, False]]]},
        {'testcase_name': 'pass_and_move',
         'states': gojax.new_states(board_size=2, batch_size=2),
         'actions': [None, (0, 0)],
         'expected_output': [[[False, False],
                              [False, False]],
                             [[True, False],
                              [False, False]]]},
    )
    def test_(self, states, actions, expected_output):
        indicator_actions = gojax.action_2d_indices_to_indicator(actions, states)
        np.testing.assert_array_equal(indicator_actions, expected_output)
        chex.assert_type(indicator_actions, bool)


class LegacyGeneralTestCase(unittest.TestCase):
    """Tests general Go functions."""

    def test_get_turns(self):
        states = gojax.new_states(2, batch_size=2)
        states = states.at[0, gojax.TURN_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(gojax.get_turns(states) == jnp.array([True, False])))

    def test_get_invalids(self):
        states = gojax.new_states(2, batch_size=2)
        states = states.at[1, gojax.INVALID_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(gojax.get_invalids(states) == jnp.array([[[False, False],
                                                                              [False, False]],
                                                                             [[True, True],
                                                                              [True, True]]
                                                                             ])))

    def test_get_passes(self):
        states = gojax.new_states(2, batch_size=2)
        states = states.at[0, gojax.PASS_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(gojax.get_passes(states) == jnp.array([True, False])))

    def test_get_ended_false(self):
        states = gojax.new_states(2, batch_size=2)
        states = states.at[0, gojax.END_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(gojax.get_ended(states) == jnp.array([True, False])))

    def test_white_moves_second(self):
        state = gojax.new_states(4)
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([(0, 0)], state))
        self.assertTrue(jnp.alltrue(state[0, gojax.TURN_CHANNEL_INDEX]))

    def test_black_moves_third(self):
        state = gojax.new_states(4)
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([None], state))
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([None], state))
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, gojax.TURN_CHANNEL_INDEX],
                               jnp.zeros_like(state[0, gojax.TURN_CHANNEL_INDEX]))))

    def test_black_and_white_moves_in_batch(self):
        states = gojax.new_states(4, batch_size=2)
        states = states.at[0, gojax.TURN_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(gojax.get_turns(states) == jnp.array([True, False])),
                        gojax.get_turns(states))
        states = gojax.next_states(states,
                                   gojax.action_2d_indices_to_indicator([None, None], states))
        self.assertTrue(jnp.alltrue(gojax.get_turns(states) == jnp.array([False, True])),
                        gojax.get_turns(states))

    def test_pass_changes_turn(self):
        state = gojax.new_states(2)
        self.assertTrue(jnp.alltrue(gojax.get_turns(state) == jnp.array([False])))
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([None], state))
        self.assertTrue(jnp.alltrue(gojax.get_turns(state) == jnp.array([True])))

    def test_pass_sets_pass_layer(self):
        state = gojax.new_states(2)
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([None], state))
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, gojax.PASS_CHANNEL_INDEX],
                               jnp.ones_like(state[0, gojax.PASS_CHANNEL_INDEX]))))

    def test_two_consecutive_passes_ends_game(self):
        state = gojax.new_states(2)
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.END_CHANNEL_INDEX],
                                           jnp.zeros_like(state[0, gojax.END_CHANNEL_INDEX]))))
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([None], state))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.END_CHANNEL_INDEX],
                                           jnp.zeros_like(state[0, gojax.END_CHANNEL_INDEX]))))
        state = gojax.next_states(state, gojax.action_2d_indices_to_indicator([None], state))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.END_CHANNEL_INDEX],
                                           jnp.ones_like(state[0, gojax.END_CHANNEL_INDEX]))))

    def test_game_end_no_op_pieces(self):
        state = gojax.decode_state("""
                                _ _ _
                                _ _ _
                                _ _ _
                                """,
                                   ended=True)
        next_state = gojax.next_states(
            state, gojax.action_2d_indices_to_indicator([(1, 1)], state))
        np.testing.assert_array_equal(
            state[0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]],
            next_state[0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]])
        np.testing.assert_array_equal(gojax.get_turns(state), [gojax.BLACKS_TURN])
        np.testing.assert_array_equal(gojax.get_turns(next_state), [gojax.WHITES_TURN])
        np.testing.assert_array_equal(gojax.get_ended(state), [True])
        np.testing.assert_array_equal(gojax.get_ended(next_state), [True])

    def test_no_op_second_state(self):
        first_state = gojax.decode_state("""
                                      _ _ _
                                      _ _ _
                                      _ _ _
                                      """)
        second_state = gojax.decode_state("""
                                       _ _ _
                                       _ _ _
                                       _ _ _
                                       """,
                                          ended=True)
        states = jnp.concatenate((first_state, second_state), axis=0)
        next_states = gojax.next_states(
            states, gojax.action_2d_indices_to_indicator([(0, 0), (0, 0)], states))
        self.assertEqual(
            jnp.sum(
                jnp.logical_xor(states[0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]],
                                next_states[
                                    0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]])), 1)
        np.testing.assert_array_equal(
            states[1, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]],
            next_states[1, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]])
        np.testing.assert_array_equal(gojax.get_turns(states),
                                      [gojax.BLACKS_TURN, gojax.BLACKS_TURN])
        np.testing.assert_array_equal(gojax.get_turns(next_states),
                                      [gojax.WHITES_TURN, gojax.WHITES_TURN])
        np.testing.assert_array_equal(gojax.get_ended(states), [False, True])
        np.testing.assert_array_equal(gojax.get_ended(next_states), [False, True])

    def test_decode_state_shape(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str)
        self.assertEqual((1, 6, 4, 4), state.shape)

    def test_decode_new_state(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str)
        self.assertTrue(jnp.alltrue(lax.eq(state, jnp.zeros_like(state))))

    def test_decode_state_turn(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str, gojax.WHITES_TURN)
        self.assertTrue(jnp.alltrue(
            lax.eq(state[0, gojax.TURN_CHANNEL_INDEX],
                   jnp.ones_like(state[0, gojax.TURN_CHANNEL_INDEX]))))

    def test_decode_state_one_piece(self):
        state_str = """
                    B _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str)
        self.assertTrue(state[0, gojax.BLACK_CHANNEL_INDEX, 0, 0])

    def test_decode_state_two_pieces(self):
        state_str = """
                    B _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ W
                    """
        state = gojax.decode_state(state_str)
        self.assertTrue(state[0, gojax.BLACK_CHANNEL_INDEX, 0, 0])
        self.assertTrue(state[0, gojax.WHITE_CHANNEL_INDEX, 3, 3])

    def test_decode_state_pass_default_false(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str)
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, gojax.PASS_CHANNEL_INDEX],
                               jnp.zeros_like(state[0, gojax.PASS_CHANNEL_INDEX]))))

    def test_decode_state_pass(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str, passed=True)
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, gojax.PASS_CHANNEL_INDEX],
                               jnp.ones_like(state[0, gojax.PASS_CHANNEL_INDEX]))))

    def test_decode_state_ended_default_false(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str)
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, gojax.END_CHANNEL_INDEX],
                               jnp.zeros_like(state[0, gojax.END_CHANNEL_INDEX]))))

    def test_decode_state_ended(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str, ended=True)
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, gojax.END_CHANNEL_INDEX],
                               jnp.ones_like(state[0, gojax.END_CHANNEL_INDEX]))))

    def test_decode_state_komi(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str, komi=(0, 0))
        self.assertTrue(state[0, gojax.INVALID_CHANNEL_INDEX, 0, 0])

    def test_get_occupied_spaces(self):
        state_str = """
                    _ B _ _
                    B _ B _
                    _ B _ _
                    _ _ W _
                    """
        state = gojax.decode_state(state_str, komi=(0, 0))
        occupied_spaces = gojax.get_occupied_spaces(state)
        np.testing.assert_array_equal(occupied_spaces, [[[False, True, False, False],
                                                         [True, False, True, False],
                                                         [False, True, False, False],
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
        state = gojax.decode_state(state_str, komi=(0, 0))
        empty_spaces = gojax.get_empty_spaces(state)
        np.testing.assert_array_equal(empty_spaces, [[[True, False, True, True],
                                                      [False, True, False, True],
                                                      [True, False, True, True],
                                                      [True, True, False, True]]])

    def test_get_free_groups_shape(self):
        state_str = """
                    _ _
                    _ _ 
                    """
        state = gojax.decode_state(state_str)
        free_black_groups = gojax.compute_free_groups(state, [gojax.BLACKS_TURN])
        self.assertEqual((1, 2, 2), free_black_groups.shape)

    def test_get_free_groups_free_single_piece(self):
        state_str = """
                    B _
                    _ _ 
                    """
        state = gojax.decode_state(state_str)
        free_black_groups = gojax.compute_free_groups(state, [gojax.BLACKS_TURN])
        self.assertTrue(
            jnp.alltrue(jnp.array([[True, False], [False, False]]) == free_black_groups))

    def test_get_free_groups_non_free_single_piece(self):
        state_str = """
                    B W
                    W _ 
                    """
        state = gojax.decode_state(state_str)
        free_black_groups = gojax.compute_free_groups(state, [gojax.BLACKS_TURN])
        self.assertTrue(
            jnp.alltrue(jnp.array([[False, False], [False, False]]) == free_black_groups),
            free_black_groups)

    def test_get_free_groups_free_chain(self):
        state_str = """
                    _ W _ _ _
                    W B W _ _
                    W B W _ _
                    W B W _ _
                    _ _ _ _ _
                    """
        state = gojax.decode_state(state_str)
        free_black_groups = gojax.compute_free_groups(state, [gojax.BLACKS_TURN])
        self.assertTrue(jnp.alltrue(jnp.array([[False, False, False, False, False],
                                               [False, True, False, False, False],
                                               [False, True, False, False, False],
                                               [False, True, False, False, False],
                                               [False, False, False, False,
                                                False], ]) == free_black_groups),
                        free_black_groups)

    def test_get_free_groups_white(self):
        state_str = """
                    B _
                    _ W 
                    """
        state = gojax.decode_state(state_str)
        free_white_groups = gojax.compute_free_groups(state, [gojax.WHITES_TURN])
        np.testing.assert_array_equal(free_white_groups, [[[False, False], [False, True]]])

    def test_get_pretty_string(self):
        state_str = """
                    B _ _ _
                    _ W _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_state(state_str)
        with open('tests/expected_pretty_string.txt', 'r', encoding='utf8') as file:
            expected_str = file.read()
        self.assertEqual(expected_str, gojax.get_pretty_string(state[0]))

    def test_compute_areas_empty(self):
        state = gojax.decode_state("""
                                    _ _ _
                                    _ _ _
                                    _ _ _
                                    """)
        np.testing.assert_array_equal(gojax.compute_areas(state), [[[[False, False, False],
                                                                     [False, False, False],
                                                                     [False, False, False]],
                                                                    [[False, False, False],
                                                                     [False, False, False],
                                                                     [False, False, False]]]])

    def test_compute_areas_pieces(self):
        state_str = """
                    B _ _
                    _ _ _
                    _ _ W
                    """
        state = gojax.decode_state(state_str)
        areas = gojax.compute_areas(state)
        np.testing.assert_array_equal(areas, [[[[True, False, False],
                                                [False, False, False],
                                                [False, False, False]],
                                               [[False, False, False],
                                                [False, False, False],
                                                [False, False, True]]]])

    def test_compute_areas_single_piece_controls_all(self):
        state_str = """
                    B _ _
                    _ _ _
                    _ _ _
                    """
        state = gojax.decode_state(state_str)
        areas = gojax.compute_areas(state)
        np.testing.assert_array_equal(areas, [[[[True, True, True],
                                                [True, True, True],
                                                [True, True, True]],
                                               [[False, False, False],
                                                [False, False, False],
                                                [False, False, False]]]])

    def test_compute_areas_donut(self):
        state_str = """
                    _ _ _ _ _
                    _ W B _ _
                    _ B _ B _
                    _ _ B _ _
                    _ _ _ _ _
                    """
        state = gojax.decode_state(state_str)
        areas = gojax.compute_areas(state)
        np.testing.assert_array_equal(areas, [[[[False, False, False, False, False],
                                                [False, False, True, False, False],
                                                [False, True, True, True, False],
                                                [False, False, True, False, False],
                                                [False, False, False, False, False]],
                                               [[False, False, False, False, False],
                                                [False, True, False, False, False],
                                                [False, False, False, False, False],
                                                [False, False, False, False, False],
                                                [False, False, False, False, False]]]])

    def test_compute_areas_batch_size_two(self):
        states = jnp.concatenate((gojax.decode_state("""
                                                    _ _ _
                                                    _ _ _
                                                    _ _ _
                                                    """),
                                  gojax.decode_state("""
                                                    _ _ _
                                                    _ B _
                                                    _ _ _
                                                    """)), axis=0)

        np.testing.assert_array_equal(gojax.compute_areas(states), [[[[False, False, False],
                                                                      [False, False, False],
                                                                      [False, False, False]],
                                                                     [[False, False, False],
                                                                      [False, False, False],
                                                                      [False, False, False]]],
                                                                    [[[True, True, True],
                                                                      [True, True, True],
                                                                      [True, True, True]],
                                                                     [[False, False, False],
                                                                      [False, False, False],
                                                                      [False, False, False]]]
                                                                    ])

    def test_compute_areas_batch_size_three(self):
        states = jnp.concatenate((gojax.decode_state("""
                                                    _ _ _
                                                    _ _ _
                                                    _ _ _
                                                    """),
                                  gojax.decode_state("""
                                                    _ _ _
                                                    _ B _
                                                    _ _ _
                                                    """),
                                  gojax.decode_state("""
                                                      _ _ _
                                                      _ W _
                                                      _ _ _
                                                      """)))

        np.testing.assert_array_equal(gojax.compute_areas(states), [[[[False, False, False],
                                                                      [False, False, False],
                                                                      [False, False, False]],
                                                                     [[False, False, False],
                                                                      [False, False, False],
                                                                      [False, False, False]]],
                                                                    [[[True, True, True],
                                                                      [True, True, True],
                                                                      [True, True, True]],
                                                                     [[False, False, False],
                                                                      [False, False, False],
                                                                      [False, False, False]]],
                                                                    [[[False, False, False],
                                                                      [False, False, False],
                                                                      [False, False, False]],
                                                                     [[True, True, True],
                                                                      [True, True, True],
                                                                      [True, True, True]]]
                                                                    ])

    def test_compute_area_sizes_pieces(self):
        state = gojax.decode_state("""
                                B _ _
                                _ _ _
                                _ _ W
                                """)
        np.testing.assert_array_equal(gojax.compute_area_sizes(state), [[1, 1]])

    def test_compute_area_sizes_single_piece_controls_all(self):
        state = gojax.decode_state("""
                                B _ _
                                _ _ _
                                _ _ _
                                """)
        np.testing.assert_array_equal(gojax.compute_area_sizes(state), [[9, 0]])

    def test_compute_area_sizes_donut(self):
        state = gojax.decode_state("""
                                _ _ _ _ _
                                _ W B _ _
                                _ B _ B _
                                _ _ B _ _
                                _ _ _ _ _
                                """)
        np.testing.assert_array_equal(gojax.compute_area_sizes(state), [[5, 1]])

    def test_compute_area_sizes_batch_size_two(self):
        states = jnp.concatenate((gojax.decode_state("""
                                                    _ _ _
                                                    _ _ _
                                                    _ _ _
                                                    """),
                                  gojax.decode_state("""
                                                    _ _ _
                                                    _ B _
                                                    _ _ _
                                                    """)))
        np.testing.assert_array_equal(gojax.compute_area_sizes(states), [[0, 0], [9, 0]])

    def test_compute_area_sizes_batch_size_three(self):
        states = jnp.concatenate((gojax.decode_state("""
                                                    _ _ _
                                                    _ _ _
                                                    _ _ _
                                                    """),
                                  gojax.decode_state("""
                                                    _ _ _
                                                    _ B _
                                                    _ _ _
                                                    """),
                                  gojax.decode_state("""
                                                      _ _ _
                                                      _ W _
                                                      _ _ _
                                                      """)))
        np.testing.assert_array_equal(gojax.compute_area_sizes(states), [[0, 0], [9, 0], [0, 9]])

    def test_compute_winning_new_state_tie(self):
        state = gojax.decode_state("""
                                _ _ _
                                _ _ _
                                _ _ _
                                """)
        np.testing.assert_array_equal(gojax.compute_winning(state), [0])

    def test_compute_winning_single_black(self):
        state = gojax.decode_state("""
                                _ _ _
                                _ B _
                                _ _ _
                                """)
        np.testing.assert_array_equal(gojax.compute_winning(state), [1])

    def test_compute_winning_single_white(self):
        state = gojax.decode_state("""
                                _ _ _
                                _ W _
                                _ _ _
                                """)
        np.testing.assert_array_equal(gojax.compute_winning(state), [-1])

    def test_compute_winning_tie_single_pieces(self):
        state = gojax.decode_state("""
                                _ _ _
                                _ W _
                                _ _ B
                                """)
        np.testing.assert_array_equal(gojax.compute_winning(state), [0])

    def test_compute_winning_black_has_more_area_with_empty_space(self):
        state = gojax.decode_state("""
                                W W _ _ _
                                W W B _ _
                                _ B _ B _
                                _ _ B _ _
                                _ _ _ _ _
                                """)
        np.testing.assert_array_equal(gojax.compute_winning(state), [1])

    def test_compute_winning_batch_size_two(self):
        states = jnp.concatenate((gojax.decode_state("""
                                                    _ _ _
                                                    _ _ _
                                                    _ _ _
                                                    """),
                                  gojax.decode_state("""
                                                    _ _ _
                                                    _ B _
                                                    _ _ _
                                                    """)))
        np.testing.assert_array_equal(gojax.compute_winning(states), [0, 1])

    def test_swap_perspectives_black_to_white(self):
        state = gojax.decode_state("""
                                B _ _
                                _ _ _
                                _ _ W
                                """, turn=gojax.BLACKS_TURN)
        swapped_perspective = gojax.swap_perspectives(state)
        np.testing.assert_array_equal(swapped_perspective[0, gojax.BLACK_CHANNEL_INDEX],
                                      [[False, False, False],
                                       [False, False, False],
                                       [False, False, True]])
        np.testing.assert_array_equal(swapped_perspective[0, gojax.WHITE_CHANNEL_INDEX],
                                      [[True, False, False],
                                       [False, False, False],
                                       [False, False, False]])
        np.testing.assert_array_equal(gojax.get_turns(swapped_perspective), [gojax.WHITES_TURN])

    def test_swap_perspectives_white_to_black(self):
        state = gojax.decode_state("""
                                B _ _
                                _ _ _
                                _ _ W
                                """, turn=gojax.WHITES_TURN)
        swapped_perspective = gojax.swap_perspectives(state)
        np.testing.assert_array_equal(swapped_perspective[0, gojax.BLACK_CHANNEL_INDEX],
                                      [[False, False, False],
                                       [False, False, False],
                                       [False, False, True]])
        np.testing.assert_array_equal(swapped_perspective[0, gojax.WHITE_CHANNEL_INDEX],
                                      [[True, False, False],
                                       [False, False, False],
                                       [False, False, False]])
        np.testing.assert_array_equal(gojax.get_turns(swapped_perspective), [gojax.BLACKS_TURN])

    def test_next_two_states(self):
        action_size = 2
        board_size = 3
        states = gojax.new_states(board_size, action_size)
        indicator_actions = jnp.reshape(
            nn.one_hot(jnp.arange(action_size), num_classes=board_size ** 2, dtype=bool),
            (action_size, board_size, board_size))
        children = gojax.next_states(states, indicator_actions)
        expected_children = jnp.concatenate((gojax.decode_state("""
                                                          B _ _
                                                          _ _ _
                                                          _ _ _
                                                          """,
                                                                turn=gojax.WHITES_TURN),
                                             gojax.decode_state("""
                                                          _ B _
                                                          _ _ _
                                                          _ _ _
                                                          """,
                                                                turn=gojax.WHITES_TURN)
                                             ), axis=0)

        np.testing.assert_array_equal(children, expected_children)

    def test_get_all_children(self):
        action_size = 10
        board_size = 3
        states = gojax.new_states(board_size, batch_size=action_size)
        indicator_actions = jnp.reshape(
            nn.one_hot(jnp.arange(action_size), num_classes=board_size ** 2, dtype=bool),
            (action_size, board_size, board_size))
        children = gojax.next_states(states, indicator_actions)
        expected_children = jnp.concatenate((gojax.decode_state("""
                                                          B _ _
                                                          _ _ _
                                                          _ _ _
                                                          """,
                                                                turn=gojax.WHITES_TURN),
                                             gojax.decode_state("""
                                                          _ B _
                                                          _ _ _
                                                          _ _ _
                                                          """,
                                                                turn=gojax.WHITES_TURN),
                                             gojax.decode_state("""
                                                          _ _ B
                                                          _ _ _
                                                          _ _ _
                                                          """,
                                                                turn=gojax.WHITES_TURN),
                                             gojax.decode_state("""
                                                          _ _ _
                                                          B _ _
                                                          _ _ _
                                                          """,
                                                                turn=gojax.WHITES_TURN),
                                             gojax.decode_state("""
                                                          _ _ _
                                                          _ B _
                                                          _ _ _
                                                          """,
                                                                turn=gojax.WHITES_TURN),
                                             gojax.decode_state("""
                                                          _ _ _
                                                          _ _ B
                                                          _ _ _
                                                          """,
                                                                turn=gojax.WHITES_TURN),
                                             gojax.decode_state("""
                                                          _ _ _
                                                          _ _ _
                                                          B _ _
                                                          """,
                                                                turn=gojax.WHITES_TURN),
                                             gojax.decode_state("""
                                                          _ _ _
                                                          _ _ _
                                                          _ B _
                                                          """,
                                                                turn=gojax.WHITES_TURN),
                                             gojax.decode_state("""
                                                          _ _ _
                                                          _ _ _
                                                          _ _ B
                                                          """,
                                                                turn=gojax.WHITES_TURN),
                                             gojax.decode_state("""
                                                          _ _ _
                                                          _ _ _
                                                          _ _ _
                                                          """,
                                                                turn=gojax.WHITES_TURN,
                                                                passed=True),
                                             )
                                            )

        np.testing.assert_array_equal(children, expected_children)


if __name__ == '__main__':
    unittest.main()

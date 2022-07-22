"""Tests general Go functions."""

# pylint: disable=missing-function-docstring,too-many-public-methods,no-self-use,duplicate-code
import textwrap
import unittest

import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized
from jax import lax
from jax import nn

import gojax
import serialize
import state_index


class GoTestCase(chex.TestCase):
    """Tests general Go functions."""

    @parameterized.named_parameters(
        {'testcase_name': 'board_size_3_batch_size_1', 'board_size': 3, 'batch_size': 1,
         'expected_output': jnp.zeros((1, gojax.NUM_CHANNELS, 3, 3), dtype=bool)},
        {'testcase_name': 'board_size_4_batch_size_1', 'board_size': 4, 'batch_size': 1,
         'expected_output': jnp.zeros((1, gojax.NUM_CHANNELS, 4, 4), dtype=bool)},
        {'testcase_name': 'board_size_3_batch_size_2', 'board_size': 3, 'batch_size': 2,
         'expected_output': jnp.zeros((2, gojax.NUM_CHANNELS, 3, 3), dtype=bool)}, )
    def test_new_states_(self, board_size, batch_size, expected_output):
        new_states = gojax.new_states(board_size, batch_size)
        np.testing.assert_array_equal(new_states, expected_output)
        chex.assert_type(new_states, bool)

    def test_white_moves_second(self):
        state = gojax.new_states(4)
        state = gojax.next_states(state,
                                  state_index.action_2d_indices_to_indicator([(0, 0)], state))
        self.assertTrue(jnp.alltrue(state[0, gojax.TURN_CHANNEL_INDEX]))

    def test_white_moves_second_v2(self):
        state = gojax.new_states(board_size=4)
        state = gojax.next_states_v2(state, actions_1d=jnp.array([0]))
        self.assertTrue(jnp.alltrue(state[0, gojax.TURN_CHANNEL_INDEX]))

    def test_black_moves_third(self):
        state = gojax.new_states(4)
        state = gojax.next_states(state, state_index.action_2d_indices_to_indicator([None], state))
        state = gojax.next_states(state, state_index.action_2d_indices_to_indicator([None], state))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.TURN_CHANNEL_INDEX],
                                           jnp.zeros_like(state[0, gojax.TURN_CHANNEL_INDEX]))))

    def test_black_moves_third_v2(self):
        state = gojax.new_states(board_size=4)
        state = gojax.next_states_v2(state, actions_1d=jnp.array([16]))
        state = gojax.next_states_v2(state, actions_1d=jnp.array([16]))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.TURN_CHANNEL_INDEX],
                                           jnp.zeros_like(state[0, gojax.TURN_CHANNEL_INDEX]))))

    def test_black_and_white_moves_in_batch(self):
        states = gojax.new_states(4, batch_size=2)
        states = states.at[0, gojax.TURN_CHANNEL_INDEX].set(True)
        np.testing.assert_array_equal(gojax.get_turns(states), [True, False])
        states = gojax.next_states(states,
                                   state_index.action_2d_indices_to_indicator([None, None], states))
        np.testing.assert_array_equal(gojax.get_turns(states), [False, True])

    def test_black_and_white_moves_in_batch_v2(self):
        states = gojax.new_states(board_size=4, batch_size=2)
        states = states.at[0, gojax.TURN_CHANNEL_INDEX].set(True)
        np.testing.assert_array_equal(gojax.get_turns(states), [True, False])
        states = gojax.next_states_v2(states, actions_1d=jnp.array([16, 16]))
        np.testing.assert_array_equal(gojax.get_turns(states), [False, True])

    def test_pass_changes_turn(self):
        state = gojax.new_states(2)
        self.assertTrue(jnp.alltrue(gojax.get_turns(state) == jnp.array([False])))
        state = gojax.next_states(state, state_index.action_2d_indices_to_indicator([None], state))
        self.assertTrue(jnp.alltrue(gojax.get_turns(state) == jnp.array([True])))

    def test_pass_changes_turn_v2(self):
        state = gojax.new_states(board_size=2)
        self.assertTrue(jnp.alltrue(gojax.get_turns(state) == jnp.array([False])))
        state = gojax.next_states_v2(state, actions_1d=jnp.array([4]))
        self.assertTrue(jnp.alltrue(gojax.get_turns(state) == jnp.array([True])))

    def test_pass_sets_pass_layer(self):
        state = gojax.new_states(2)
        state = gojax.next_states(state, state_index.action_2d_indices_to_indicator([None], state))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.PASS_CHANNEL_INDEX],
                                           jnp.ones_like(state[0, gojax.PASS_CHANNEL_INDEX]))))

    def test_pass_sets_pass_layer_v2(self):
        state = gojax.new_states(board_size=2)
        state = gojax.next_states_v2(state, actions_1d=jnp.array([4]))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.PASS_CHANNEL_INDEX],
                                           jnp.ones_like(state[0, gojax.PASS_CHANNEL_INDEX]))))

    def test_two_consecutive_passes_ends_game(self):
        state = gojax.new_states(2)
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.END_CHANNEL_INDEX],
                                           jnp.zeros_like(state[0, gojax.END_CHANNEL_INDEX]))))
        state = gojax.next_states(state, state_index.action_2d_indices_to_indicator([None], state))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.END_CHANNEL_INDEX],
                                           jnp.zeros_like(state[0, gojax.END_CHANNEL_INDEX]))))
        state = gojax.next_states(state, state_index.action_2d_indices_to_indicator([None], state))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.END_CHANNEL_INDEX],
                                           jnp.ones_like(state[0, gojax.END_CHANNEL_INDEX]))))

    def test_two_consecutive_passes_ends_game_v2(self):
        state = gojax.new_states(board_size=2)
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.END_CHANNEL_INDEX],
                                           jnp.zeros_like(state[0, gojax.END_CHANNEL_INDEX]))))
        state = gojax.next_states_v2(state, actions_1d=jnp.array([4]))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.END_CHANNEL_INDEX],
                                           jnp.zeros_like(state[0, gojax.END_CHANNEL_INDEX]))))
        state = gojax.next_states_v2(state, actions_1d=jnp.array([4]))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, gojax.END_CHANNEL_INDEX],
                                           jnp.ones_like(state[0, gojax.END_CHANNEL_INDEX]))))

    def test_game_end_no_op_pieces(self):
        state = serialize.decode_states("""
                        _ _ _
                        _ _ _
                        _ _ _
                        """, ended=True)
        next_state = gojax.next_states(state,
                                       state_index.action_2d_indices_to_indicator([(1, 1)], state))
        np.testing.assert_array_equal(
            state[0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]],
            next_state[0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]])
        np.testing.assert_array_equal(gojax.get_turns(state), [gojax.BLACKS_TURN])
        np.testing.assert_array_equal(gojax.get_turns(next_state), [gojax.WHITES_TURN])
        np.testing.assert_array_equal(gojax.get_ended(state), [True])
        np.testing.assert_array_equal(gojax.get_ended(next_state), [True])

    def test_game_end_no_op_pieces_v2(self):
        state = serialize.decode_states("""
                        _ _ _
                        _ _ _
                        _ _ _
                        """, ended=True)
        next_state = gojax.next_states_v2(state, actions_1d=jnp.array([4]))
        np.testing.assert_array_equal(
            state[0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]],
            next_state[0, [gojax.BLACK_CHANNEL_INDEX, gojax.WHITE_CHANNEL_INDEX]])
        np.testing.assert_array_equal(gojax.get_turns(state), [gojax.BLACKS_TURN])
        np.testing.assert_array_equal(gojax.get_turns(next_state), [gojax.WHITES_TURN])
        np.testing.assert_array_equal(gojax.get_ended(state), [True])
        np.testing.assert_array_equal(gojax.get_ended(next_state), [True])

    def test_no_op_second_state(self):
        first_state = serialize.decode_states("""
                              _ _ _
                              _ _ _
                              _ _ _
                              """)
        second_state = serialize.decode_states("""
                               _ _ _
                               _ _ _
                               _ _ _
                               """, ended=True)
        states = jnp.concatenate((first_state, second_state), axis=0)
        next_states = gojax.next_states(states,
                                        state_index.action_2d_indices_to_indicator([(0, 0), (0, 0)],
                                                                                   states))
        self.assertEqual(jnp.sum(
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

    def test_paint_fill(self):
        x = jnp.array([[[[1, 1, 0, 1, 1], [0, 0, 0, 1, 1], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0]]]], dtype=bool)

        seeds = jnp.array([[[[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]]]], dtype=bool)

        expected_fill = jnp.array([[[[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 0, 0, 0],
                                     [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]]]], dtype=bool)
        np.testing.assert_array_equal(gojax.paint_fill(seeds, x), expected_fill)

    def test_compute_free_groups_shape(self):
        state_str = """
            _ _
            _ _ 
            """
        state = serialize.decode_states(state_str)
        free_black_groups = gojax.compute_free_groups(state, jnp.array([gojax.BLACKS_TURN]))
        self.assertEqual((1, 2, 2), free_black_groups.shape)

    def test_compute_free_groups_free_single_piece(self):
        state_str = """
            B _
            _ _ 
            """
        state = serialize.decode_states(state_str)
        free_black_groups = gojax.compute_free_groups(state, jnp.array([gojax.BLACKS_TURN]))
        self.assertTrue(
            jnp.alltrue(jnp.array([[True, False], [False, False]]) == free_black_groups))

    def test_compute_free_groups_non_free_single_piece(self):
        state_str = """
            B W
            W _ 
            """
        state = serialize.decode_states(state_str)
        free_black_groups = gojax.compute_free_groups(state, jnp.array([gojax.BLACKS_TURN]))
        self.assertTrue(
            jnp.alltrue(jnp.array([[False, False], [False, False]]) == free_black_groups),
            free_black_groups)

    def test_compute_free_groups_free_chain(self):
        state_str = """
            _ W _ _ _
            W B W _ _
            W B W _ _
            W B W _ _
            _ _ _ _ _
            """
        state = serialize.decode_states(state_str)
        free_black_groups = gojax.compute_free_groups(state, jnp.array([gojax.BLACKS_TURN]))
        self.assertTrue(jnp.alltrue(jnp.array(
            [[False, False, False, False, False], [False, True, False, False, False],
             [False, True, False, False, False], [False, True, False, False, False],
             [False, False, False, False, False], ]) == free_black_groups), free_black_groups)

    def test_compute_free_groups_white(self):
        state_str = """
            B _
            _ W 
            """
        state = serialize.decode_states(state_str)
        free_white_groups = gojax.compute_free_groups(state, jnp.array([gojax.WHITES_TURN]))
        np.testing.assert_array_equal(free_white_groups, [[[False, False], [False, True]]])

    def test_get_pretty_string(self):
        state_str = """
            B _ _ _
            _ W _ _
            _ _ _ _
            _ _ _ _
            """
        state = serialize.decode_states(state_str)
        expected_str = textwrap.dedent("""\
        \tA B C D 
        0	○═╤═╤═╗
        1	╟─●─┼─╢
        2	╟─┼─┼─╢
        3	╚═╧═╧═╝
        \tTurn: BLACK, Game State: ONGOING
        \tBlack Area: 1, White Area: 1
        """)

        self.assertEqual(serialize.get_pretty_string(state[0]), expected_str)

    def test_compute_areas_empty(self):
        state = serialize.decode_states("""
                            _ _ _
                            _ _ _
                            _ _ _
                            """)
        np.testing.assert_array_equal(gojax.compute_areas(state), [
            [[[False, False, False], [False, False, False], [False, False, False]],
             [[False, False, False], [False, False, False], [False, False, False]]]])

    def test_compute_areas_pieces(self):
        state_str = """
            B _ _
            _ _ _
            _ _ W
            """
        state = serialize.decode_states(state_str)
        areas = gojax.compute_areas(state)
        np.testing.assert_array_equal(areas, [
            [[[True, False, False], [False, False, False], [False, False, False]],
             [[False, False, False], [False, False, False], [False, False, True]]]])

    def test_compute_areas_single_piece_controls_all(self):
        state_str = """
            B _ _
            _ _ _
            _ _ _
            """
        state = serialize.decode_states(state_str)
        areas = gojax.compute_areas(state)
        np.testing.assert_array_equal(areas, [
            [[[True, True, True], [True, True, True], [True, True, True]],
             [[False, False, False], [False, False, False], [False, False, False]]]])

    def test_compute_areas_donut(self):
        state_str = """
            _ _ _ _ _
            _ W B _ _
            _ B _ B _
            _ _ B _ _
            _ _ _ _ _
            """
        state = serialize.decode_states(state_str)
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
        states = jnp.concatenate((serialize.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _
                                            """), serialize.decode_states("""
                                            _ _ _
                                            _ B _
                                            _ _ _
                                            """)), axis=0)

        np.testing.assert_array_equal(gojax.compute_areas(states), [
            [[[False, False, False], [False, False, False], [False, False, False]],
             [[False, False, False], [False, False, False], [False, False, False]]],
            [[[True, True, True], [True, True, True], [True, True, True]],
             [[False, False, False], [False, False, False], [False, False, False]]]])

    def test_compute_areas_batch_size_three(self):
        states = jnp.concatenate((serialize.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _
                                            """), serialize.decode_states("""
                                            _ _ _
                                            _ B _
                                            _ _ _
                                            """), serialize.decode_states("""
                                              _ _ _
                                              _ W _
                                              _ _ _
                                              """)))

        np.testing.assert_array_equal(gojax.compute_areas(states), [
            [[[False, False, False], [False, False, False], [False, False, False]],
             [[False, False, False], [False, False, False], [False, False, False]]],
            [[[True, True, True], [True, True, True], [True, True, True]],
             [[False, False, False], [False, False, False], [False, False, False]]],
            [[[False, False, False], [False, False, False], [False, False, False]],
             [[True, True, True], [True, True, True], [True, True, True]]]])

    def test_compute_area_sizes_pieces(self):
        state = serialize.decode_states("""
                        B _ _
                        _ _ _
                        _ _ W
                        """)
        np.testing.assert_array_equal(gojax.compute_area_sizes(state), [[1, 1]])

    def test_compute_area_sizes_single_piece_controls_all(self):
        state = serialize.decode_states("""
                        B _ _
                        _ _ _
                        _ _ _
                        """)
        np.testing.assert_array_equal(gojax.compute_area_sizes(state), [[9, 0]])

    def test_compute_area_sizes_donut(self):
        state = serialize.decode_states("""
                        _ _ _ _ _
                        _ W B _ _
                        _ B _ B _
                        _ _ B _ _
                        _ _ _ _ _
                        """)
        np.testing.assert_array_equal(gojax.compute_area_sizes(state), [[5, 1]])

    def test_compute_area_sizes_batch_size_two(self):
        states = jnp.concatenate((serialize.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _
                                            """), serialize.decode_states("""
                                            _ _ _
                                            _ B _
                                            _ _ _
                                            """)))
        np.testing.assert_array_equal(gojax.compute_area_sizes(states), [[0, 0], [9, 0]])

    def test_compute_area_sizes_batch_size_three(self):
        states = jnp.concatenate((serialize.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _
                                            """), serialize.decode_states("""
                                            _ _ _
                                            _ B _
                                            _ _ _
                                            """), serialize.decode_states("""
                                              _ _ _
                                              _ W _
                                              _ _ _
                                              """)))
        np.testing.assert_array_equal(gojax.compute_area_sizes(states), [[0, 0], [9, 0], [0, 9]])

    def test_compute_winning_new_state_tie(self):
        state = serialize.decode_states("""
                        _ _ _
                        _ _ _
                        _ _ _
                        """)
        np.testing.assert_array_equal(gojax.compute_winning(state), [0])

    def test_compute_winning_single_black(self):
        state = serialize.decode_states("""
                        _ _ _
                        _ B _
                        _ _ _
                        """)
        np.testing.assert_array_equal(gojax.compute_winning(state), [1])

    def test_compute_winning_single_white(self):
        state = serialize.decode_states("""
                        _ _ _
                        _ W _
                        _ _ _
                        """)
        np.testing.assert_array_equal(gojax.compute_winning(state), [-1])

    def test_compute_winning_tie_single_pieces(self):
        state = serialize.decode_states("""
                        _ _ _
                        _ W _
                        _ _ B
                        """)
        np.testing.assert_array_equal(gojax.compute_winning(state), [0])

    def test_compute_winning_black_has_more_area_with_empty_space(self):
        state = serialize.decode_states("""
                        W W _ _ _
                        W W B _ _
                        _ B _ B _
                        _ _ B _ _
                        _ _ _ _ _
                        """)
        np.testing.assert_array_equal(gojax.compute_winning(state), [1])

    def test_compute_winning_batch_size_two(self):
        states = jnp.concatenate((serialize.decode_states("""
                                            _ _ _
                                            _ _ _
                                            _ _ _
                                            """), serialize.decode_states("""
                                            _ _ _
                                            _ B _
                                            _ _ _
                                            """)))
        np.testing.assert_array_equal(gojax.compute_winning(states), [0, 1])

    def test_swap_perspectives_black_to_white(self):
        state = serialize.decode_states("""
                        B _ _
                        _ _ _
                        _ _ W
                        """, turn=gojax.BLACKS_TURN)
        swapped_perspective = gojax.swap_perspectives(state)
        np.testing.assert_array_equal(swapped_perspective[0, gojax.BLACK_CHANNEL_INDEX],
                                      [[False, False, False], [False, False, False],
                                       [False, False, True]])
        np.testing.assert_array_equal(swapped_perspective[0, gojax.WHITE_CHANNEL_INDEX],
                                      [[True, False, False], [False, False, False],
                                       [False, False, False]])
        np.testing.assert_array_equal(gojax.get_turns(swapped_perspective), [gojax.WHITES_TURN])

    def test_swap_perspectives_white_to_black(self):
        state = serialize.decode_states("""
                        B _ _
                        _ _ _
                        _ _ W
                        """, turn=gojax.WHITES_TURN)
        swapped_perspective = gojax.swap_perspectives(state)
        np.testing.assert_array_equal(swapped_perspective[0, gojax.BLACK_CHANNEL_INDEX],
                                      [[False, False, False], [False, False, False],
                                       [False, False, True]])
        np.testing.assert_array_equal(swapped_perspective[0, gojax.WHITE_CHANNEL_INDEX],
                                      [[True, False, False], [False, False, False],
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
        expected_children = jnp.concatenate((serialize.decode_states("""
                                                  B _ _
                                                  _ _ _
                                                  _ _ _
                                                  """, turn=gojax.WHITES_TURN),
                                             serialize.decode_states("""
                                                  _ B _
                                                  _ _ _
                                                  _ _ _
                                                  """, turn=gojax.WHITES_TURN)), axis=0)

        np.testing.assert_array_equal(children, expected_children)

    def test_get_children(self):
        action_size = 10
        board_size = 3
        batch_size = 1
        states = gojax.new_states(board_size, batch_size)
        children = gojax.get_children(states)
        chex.assert_shape(children,
                          (batch_size, action_size, gojax.NUM_CHANNELS, board_size, board_size))
        expected_children = serialize.decode_states("""
                                                  B _ _
                                                  _ _ _
                                                  _ _ _
                                                  
                                                  _ B _
                                                  _ _ _
                                                  _ _ _
                                                  
                                                  _ _ B
                                                  _ _ _
                                                  _ _ _
                                                  
                                                  _ _ _
                                                  B _ _
                                                  _ _ _
                                                  
                                                  _ _ _
                                                  _ B _
                                                  _ _ _
                                                  
                                                  _ _ _
                                                  _ _ B
                                                  _ _ _
                                                  
                                                  _ _ _
                                                  _ _ _
                                                  B _ _
                                                  
                                                  _ _ _
                                                  _ _ _
                                                  _ B _
                                                  
                                                  _ _ _
                                                  _ _ _
                                                  _ _ B
                                                  
                                                  _ _ _
                                                  _ _ _
                                                  _ _ _
                                                  PASS=T
                                                    """, turn=gojax.WHITES_TURN)

        np.testing.assert_array_equal(children, jnp.expand_dims(expected_children, 0))

    def test_get_children_batches(self):
        """Test get_children works with two states."""
        action_size = 10
        board_size = 3
        batch_size = 2
        states = gojax.new_states(board_size, batch_size)
        children = gojax.get_children(states)
        chex.assert_shape(children,
                          (batch_size, action_size, gojax.NUM_CHANNELS, board_size, board_size))
        expected_children = serialize.decode_states("""
                                                  B _ _
                                                  _ _ _
                                                  _ _ _

                                                  _ B _
                                                  _ _ _
                                                  _ _ _

                                                  _ _ B
                                                  _ _ _
                                                  _ _ _

                                                  _ _ _
                                                  B _ _
                                                  _ _ _

                                                  _ _ _
                                                  _ B _
                                                  _ _ _

                                                  _ _ _
                                                  _ _ B
                                                  _ _ _

                                                  _ _ _
                                                  _ _ _
                                                  B _ _

                                                  _ _ _
                                                  _ _ _
                                                  _ B _

                                                  _ _ _
                                                  _ _ _
                                                  _ _ B

                                                  _ _ _
                                                  _ _ _
                                                  _ _ _
                                                  PASS=T
                                                    """, turn=gojax.WHITES_TURN)
        expected_children = jnp.repeat(jnp.expand_dims(expected_children, 0), batch_size, axis=0)
        np.testing.assert_array_equal(children, expected_children)


if __name__ == '__main__':
    unittest.main()

import unittest

import jax.numpy as jnp
import numpy as np
from jax import lax

from gojax import constants
from gojax import go


class GeneralTestCase(unittest.TestCase):
    def test_new_state_default_single_batch_size(self):
        state = go.new_states(4)
        self.assertEqual(state.shape, (1, constants.NUM_CHANNELS, 4, 4))

    def test_new_state_all_zeros(self):
        state = go.new_states(4)
        self.assertTrue(jnp.alltrue(state == 0))

    def test_new_state_batch_size_two(self):
        state = go.new_states(4, batch_size=2)
        self.assertEqual(state.shape, (2, constants.NUM_CHANNELS, 4, 4))

    def test_to_indicator_actions_pass(self):
        state = go.new_states(2)
        indicator_moves = go.to_indicator_actions([None], state)
        self.assertTrue(jnp.alltrue(lax.eq(indicator_moves, jnp.zeros_like(indicator_moves))))

    def test_to_indicator_actions_black_move(self):
        state = go.new_states(2)
        indicator_actions = go.to_indicator_actions([(0, 0)], state)
        self.assertTrue(jnp.alltrue(lax.eq(indicator_actions, jnp.array([[[True, False],
                                                                          [False, False]]]))))

    def test_to_indicator_actions_white_move(self):
        state = go.new_states(2)
        state = state.at[0, 2].set(True)
        indicator_actions = go.to_indicator_actions([(0, 0)], state)
        self.assertTrue(jnp.alltrue(lax.eq(indicator_actions, jnp.array([[[True, False],
                                                                          [False, False]]]))))

    def test_to_indicator_actions_pass_and_move(self):
        states = go.new_states(2, batch_size=2)
        indicator_moves = go.to_indicator_actions([None, (0, 0)], states)
        self.assertTrue(jnp.alltrue(lax.eq(indicator_moves[0], jnp.zeros_like(indicator_moves[0]))))
        self.assertTrue(jnp.alltrue(lax.eq(indicator_moves[1], jnp.array([[True, False],
                                                                          [False, False]]))))

    def test_get_turns(self):
        states = go.new_states(2, batch_size=2)
        states = states.at[0, constants.TURN_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(go.get_turns(states) == jnp.array([True, False])))

    def test_get_invalids(self):
        states = go.new_states(2, batch_size=2)
        states = states.at[1, constants.INVALID_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(go.get_invalids(states) == jnp.array([[[False, False],
                                                                           [False, False]],
                                                                          [[True, True],
                                                                           [True, True]]
                                                                          ])))

    def test_get_passes(self):
        states = go.new_states(2, batch_size=2)
        states = states.at[0, constants.PASS_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(go.get_passes(states) == jnp.array([True, False])))

    def test_get_ended_false(self):
        states = go.new_states(2, batch_size=2)
        states = states.at[0, constants.END_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(go.get_ended(states) == jnp.array([True, False])))

    def test_white_moves_second(self):
        state = go.new_states(4)
        state = go.next_states(state, go.to_indicator_actions([(0, 0)], state))
        self.assertTrue(jnp.alltrue(state[0, constants.TURN_CHANNEL_INDEX]))

    def test_black_moves_third(self):
        state = go.new_states(4)
        state = go.next_states(state, go.to_indicator_actions([None], state))
        state = go.next_states(state, go.to_indicator_actions([None], state))
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, constants.TURN_CHANNEL_INDEX],
                               jnp.zeros_like(state[0, constants.TURN_CHANNEL_INDEX]))))

    def test_black_and_white_moves_in_batch(self):
        states = go.new_states(4, batch_size=2)
        states = states.at[0, constants.TURN_CHANNEL_INDEX].set(True)
        self.assertTrue(jnp.alltrue(go.get_turns(states) == jnp.array([True, False])), go.get_turns(states))
        states = go.next_states(states, go.to_indicator_actions([None, None], states))
        self.assertTrue(jnp.alltrue(go.get_turns(states) == jnp.array([False, True])), go.get_turns(states))

    def test_pass_changes_turn(self):
        state = go.new_states(2)
        self.assertTrue(jnp.alltrue(go.get_turns(state) == jnp.array([False])))
        state = go.next_states(state, go.to_indicator_actions([None], state))
        self.assertTrue(jnp.alltrue(go.get_turns(state) == jnp.array([True])))

    def test_pass_sets_pass_layer(self):
        state = go.new_states(2)
        state = go.next_states(state, go.to_indicator_actions([None], state))
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, constants.PASS_CHANNEL_INDEX],
                               jnp.ones_like(state[0, constants.PASS_CHANNEL_INDEX]))))

    def test_two_consecutive_passes_ends_game(self):
        state = go.new_states(2)
        self.assertTrue(jnp.alltrue(lax.eq(state[0, constants.END_CHANNEL_INDEX],
                                           jnp.zeros_like(state[0, constants.END_CHANNEL_INDEX]))))
        state = go.next_states(state, go.to_indicator_actions([None], state))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, constants.END_CHANNEL_INDEX],
                                           jnp.zeros_like(state[0, constants.END_CHANNEL_INDEX]))))
        state = go.next_states(state, go.to_indicator_actions([None], state))
        self.assertTrue(jnp.alltrue(lax.eq(state[0, constants.END_CHANNEL_INDEX],
                                           jnp.ones_like(state[0, constants.END_CHANNEL_INDEX]))))

    def test_decode_state_shape(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str)
        self.assertEqual((1, 6, 4, 4), state.shape)

    def test_decode_new_state(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str)
        self.assertTrue(jnp.alltrue(lax.eq(state, jnp.zeros_like(state))))

    def test_decode_state_turn(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, constants.WHITES_TURN)
        self.assertTrue(jnp.alltrue(
            lax.eq(state[0, constants.TURN_CHANNEL_INDEX],
                   jnp.ones_like(state[0, constants.TURN_CHANNEL_INDEX]))))

    def test_decode_state_one_piece(self):
        state_str = """
                    B _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str)
        self.assertTrue(state[0, constants.BLACK_CHANNEL_INDEX, 0, 0])

    def test_decode_state_two_pieces(self):
        state_str = """
                    B _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ W
                    """
        state = go.decode_state(state_str)
        self.assertTrue(state[0, constants.BLACK_CHANNEL_INDEX, 0, 0])
        self.assertTrue(state[0, constants.WHITE_CHANNEL_INDEX, 3, 3])

    def test_decode_state_pass_default_false(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str)
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, constants.PASS_CHANNEL_INDEX],
                               jnp.zeros_like(state[0, constants.PASS_CHANNEL_INDEX]))))

    def test_decode_state_pass(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, passed=True)
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, constants.PASS_CHANNEL_INDEX],
                               jnp.ones_like(state[0, constants.PASS_CHANNEL_INDEX]))))

    def test_decode_state_ended_default_false(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str)
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, constants.END_CHANNEL_INDEX],
                               jnp.zeros_like(state[0, constants.END_CHANNEL_INDEX]))))

    def test_decode_state_ended(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, ended=True)
        self.assertTrue(
            jnp.alltrue(lax.eq(state[0, constants.END_CHANNEL_INDEX],
                               jnp.ones_like(state[0, constants.END_CHANNEL_INDEX]))))

    def test_decode_state_komi(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str, komi=(0, 0))
        self.assertTrue(state[0, constants.INVALID_CHANNEL_INDEX, 0, 0])

    def test_get_occupied_spaces(self):
        state_str = """
                    _ B _ _
                    B _ B _
                    _ B _ _
                    _ _ W _
                    """
        state = go.decode_state(state_str, komi=(0, 0))
        occupied_spaces = go.get_occupied_spaces(state)
        np.testing.assert_array_equal(occupied_spaces, [[[False, True, False, False],
                                                         [True, False, True, False],
                                                         [False, True, False, False],
                                                         [False, False, True, False]]])

    def test_get_empty_spaces(self):
        state_str = """
                    _ B _ _
                    B _ B _
                    _ B _ _
                    _ _ W _
                    """
        state = go.decode_state(state_str, komi=(0, 0))
        empty_spaces = go.get_empty_spaces(state)
        np.testing.assert_array_equal(empty_spaces, [[[True, False, True, True],
                                                      [False, True, False, True],
                                                      [True, False, True, True],
                                                      [True, True, False, True]]])

    def test_get_free_groups_shape(self):
        state_str = """
                    _ _
                    _ _ 
                    """
        state = go.decode_state(state_str)
        free_black_groups = go.compute_free_groups(state, [constants.BLACKS_TURN])
        self.assertEqual((1, 2, 2), free_black_groups.shape)

    def test_get_free_groups_free_single_piece(self):
        state_str = """
                    B _
                    _ _ 
                    """
        state = go.decode_state(state_str)
        free_black_groups = go.compute_free_groups(state, [constants.BLACKS_TURN])
        self.assertTrue(jnp.alltrue(jnp.array([[True, False], [False, False]]) == free_black_groups))

    def test_get_free_groups_non_free_single_piece(self):
        state_str = """
                    B W
                    W _ 
                    """
        state = go.decode_state(state_str)
        free_black_groups = go.compute_free_groups(state, [constants.BLACKS_TURN])
        self.assertTrue(jnp.alltrue(jnp.array([[False, False], [False, False]]) == free_black_groups),
                        free_black_groups)

    def test_get_free_groups_free_chain(self):
        state_str = """
                    _ W _ _ _
                    W B W _ _
                    W B W _ _
                    W B W _ _
                    _ _ _ _ _
                    """
        state = go.decode_state(state_str)
        free_black_groups = go.compute_free_groups(state, [constants.BLACKS_TURN])
        self.assertTrue(jnp.alltrue(jnp.array([[False, False, False, False, False],
                                               [False, True, False, False, False],
                                               [False, True, False, False, False],
                                               [False, True, False, False, False],
                                               [False, False, False, False, False], ]) == free_black_groups),
                        free_black_groups)

    def test_get_free_groups_white(self):
        state_str = """
                    B _
                    _ W 
                    """
        state = go.decode_state(state_str)
        free_white_groups = go.compute_free_groups(state, [constants.WHITES_TURN])
        np.testing.assert_array_equal(free_white_groups, [[[False, False], [False, True]]])

    def test_get_pretty_string(self):
        state_str = """
                    B _ _ _
                    _ W _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = go.decode_state(state_str)
        expected_str = open('expected_pretty_string.txt', 'r').read()
        self.assertEqual(expected_str, go.get_pretty_string(state[0]))

    def test_compute_areas_pieces(self):
        state_str = """
                    B _ _
                    _ _ _
                    _ _ W
                    """
        state = go.decode_state(state_str)
        areas = go.compute_areas(state)
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
        state = go.decode_state(state_str)
        areas = go.compute_areas(state)
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
        state = go.decode_state(state_str)
        areas = go.compute_areas(state)
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

    def test_compute_area_sizes_pieces(self):
        state = go.decode_state("""
                                B _ _
                                _ _ _
                                _ _ W
                                """)
        np.testing.assert_array_equal(go.compute_area_sizes(state), [[1, 1]])

    def test_compute_area_sizes_single_piece_controls_all(self):
        state = go.decode_state("""
                                B _ _
                                _ _ _
                                _ _ _
                                """)
        np.testing.assert_array_equal(go.compute_area_sizes(state), [[9, 0]])

    def test_compute_area_sizes_donut(self):
        state = go.decode_state("""
                                _ _ _ _ _
                                _ W B _ _
                                _ B _ B _
                                _ _ B _ _
                                _ _ _ _ _
                                """)
        np.testing.assert_array_equal(go.compute_area_sizes(state), [[5, 1]])

    def test_compute_winning_new_state_tie(self):
        state = go.decode_state("""
                                _ _ _
                                _ _ _
                                _ _ _
                                """)
        np.testing.assert_array_equal(go.compute_winning(state), [0])

    def test_compute_winning_single_black(self):
        state = go.decode_state("""
                                _ _ _
                                _ B _
                                _ _ _
                                """)
        np.testing.assert_array_equal(go.compute_winning(state), [1])

    def test_compute_winning_single_white(self):
        state = go.decode_state("""
                                _ _ _
                                _ W _
                                _ _ _
                                """)
        np.testing.assert_array_equal(go.compute_winning(state), [-1])

    def test_compute_winning_tie_single_pieces(self):
        state = go.decode_state("""
                                _ _ _
                                _ W _
                                _ _ B
                                """)
        np.testing.assert_array_equal(go.compute_winning(state), [0])

    def test_compute_winning_black_has_more_area_with_empty_space(self):
        state = go.decode_state("""
                                W W _ _ _
                                W W B _ _
                                _ B _ B _
                                _ _ B _ _
                                _ _ _ _ _
                                """)
        np.testing.assert_array_equal(go.compute_winning(state), [1])

    def test_swap_perspectives_black_to_white(self):
        state = go.decode_state("""
                                B _ _
                                _ _ _
                                _ _ W
                                """, turn=constants.BLACKS_TURN)
        swapped_perspective = go.swap_perspectives(state)
        np.testing.assert_array_equal(swapped_perspective[0, constants.BLACK_CHANNEL_INDEX], [[False, False, False],
                                                                                              [False, False, False],
                                                                                              [False, False, True]])
        np.testing.assert_array_equal(swapped_perspective[0, constants.WHITE_CHANNEL_INDEX], [[True, False, False],
                                                                                              [False, False, False],
                                                                                              [False, False, False]])
        np.testing.assert_array_equal(go.get_turns(swapped_perspective), [constants.WHITES_TURN])

    def test_swap_perspectives_white_to_black(self):
        state = go.decode_state("""
                                B _ _
                                _ _ _
                                _ _ W
                                """, turn=constants.WHITES_TURN)
        swapped_perspective = go.swap_perspectives(state)
        np.testing.assert_array_equal(swapped_perspective[0, constants.BLACK_CHANNEL_INDEX], [[False, False, False],
                                                                                              [False, False, False],
                                                                                              [False, False, True]])
        np.testing.assert_array_equal(swapped_perspective[0, constants.WHITE_CHANNEL_INDEX], [[True, False, False],
                                                                                              [False, False, False],
                                                                                              [False, False, False]])
        np.testing.assert_array_equal(go.get_turns(swapped_perspective), [constants.BLACKS_TURN])

    if __name__ == '__main__':
        unittest.main()

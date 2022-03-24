import unittest

import jax.numpy as jnp
from jax import lax

import go


class GoTest(unittest.TestCase):
    def test_new_state_default_single_batch_size(self):
        state = go.new_states(4)
        self.assertEqual(state.shape, (1, 6, 4, 4))

    def test_new_state_all_zeros(self):
        state = go.new_states(4)
        self.assertTrue(jnp.all(state == 0))

    def test_new_state_batch_size_two(self):
        state = go.new_states(4, batch_size=2)
        self.assertEqual(state.shape, (2, 6, 4, 4))

    def test_to_indicator_actions_pass(self):
        self.assertEqual(True, False)  # add assertion here

    def test_to_indicator_actions_move(self):
        state = go.new_states(2)
        indicator_actions = go.to_indicator_actions([(0, 0)], state)
        self.assertTrue(jnp.all(lax.eq(indicator_actions, jnp.array([[[[True, False],
                                                                       [False, False]],
                                                                      [[False, False],
                                                                       [False, False]],
                                                                      [[False, False],
                                                                       [False, False]],
                                                                      [[False, False],
                                                                       [False, False]],
                                                                      [[False, False],
                                                                       [False, False]],
                                                                      [[False, False],
                                                                       [False, False]]]]))))

    def test_to_indicator_actions_pass_and_move(self):
        self.assertEqual(True, False)  # add assertion here

    def test_get_turns(self):
        self.assertEqual(True, False)  # add assertion here

    def test_white_moves_second(self):
        state = go.new_states(4)
        state = go.next_states(state, go.to_indicator_actions([(0, 0)], state))
        self.assertTrue(jnp.all(state[0, 2]))

    def test_black_and_white_moves_in_batch(self):
        self.assertEqual(True, False)  # add assertion here

    def test_pass(self):
        self.assertEqual(True, False)  # add assertion here

    def test_two_consecutive_passes_ends_game(self):
        self.assertEqual(True, False)  # add assertion here

    def test_invalid_move_space_occupied_by_own_pieces(self):
        self.assertEqual(True, False)  # add assertion here

    def test_invalid_move_space_occupied_by_opponent_pieces(self):
        self.assertEqual(True, False)  # add assertion here

    def test_invalid_move_no_liberties(self):
        self.assertEqual(True, False)  # add assertion here

    def test_invalid_move_komi(self):
        self.assertEqual(True, False)  # add assertion here

    def test_remove_single_piece(self):
        self.assertEqual(True, False)  # add assertion here

    def test_remove_two_connected_pieces(self):
        self.assertEqual(True, False)  # add assertion here

    def test_remove_two_disjoint_pieces(self):
        self.assertEqual(True, False)  # add assertion here

    def test_remove_donut(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()

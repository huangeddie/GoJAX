import unittest

import jax.numpy as jnp

import go


class GoTest(unittest.TestCase):
    def test_new_state_default_squeeze_shape(self):
        state = go.new_state(4)
        self.assertEqual(state.shape, (6, 4, 4))

    def test_new_state_all_zeros(self):
        state = go.new_state(4)
        self.assertTrue(jnp.all(state == 0))

    def test_new_state_nosqueeze_shape(self):
        state = go.new_state(4, squeeze=False)
        self.assertEqual(state.shape, (1, 6, 4, 4))

    def test_new_state_batch_shape(self):
        state = go.new_state(4, batch_size=2)
        self.assertEqual(state.shape, (2, 6, 4, 4))

    def test_black_moves_first(self):
        self.assertEqual(True, False)  # add assertion here

    def test_white_moves_second(self):
        self.assertEqual(True, False)  # add assertion here

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

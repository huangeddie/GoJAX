import unittest


class GoTest(unittest.TestCase):
    def test_init_state_squeeze_default(self):
        self.assertEqual(True, False)  # add assertion here

    def test_init_state_unsqueeze_single(self):
        self.assertEqual(True, False)  # add assertion here

    def test_init_state_batch(self):
        self.assertEqual(True, False)  # add assertion here

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

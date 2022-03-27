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
        self.assertTrue(jnp.alltrue(state == 0))

    def test_new_state_batch_size_two(self):
        state = go.new_states(4, batch_size=2)
        self.assertEqual(state.shape, (2, 6, 4, 4))

    def test_to_indicator_actions_pass(self):
        state = go.new_states(2)
        indicator_moves = go.to_indicator_actions([None], state)
        self.assertTrue(jnp.alltrue(lax.eq(indicator_moves, jnp.zeros_like(indicator_moves))))

    def test_to_indicator_actions_black_move(self):
        state = go.new_states(2)
        indicator_actions = go.to_indicator_actions([(0, 0)], state)
        self.assertTrue(jnp.alltrue(lax.eq(indicator_actions, jnp.array([[[[True, False],
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

    def test_to_indicator_actions_white_move(self):
        state = go.new_states(2)
        state = state.at[0, 2].set(True)
        indicator_actions = go.to_indicator_actions([(0, 0)], state)
        self.assertTrue(jnp.alltrue(lax.eq(indicator_actions, jnp.array([[[[False, False],
                                                                           [False, False]],
                                                                          [[True, False],
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
        states = go.new_states(2, batch_size=2)
        indicator_moves = go.to_indicator_actions([None, (0, 0)], states)
        self.assertTrue(jnp.alltrue(lax.eq(indicator_moves[0], jnp.zeros_like(indicator_moves[0]))))
        self.assertEqual(jnp.alltrue(lax.eq(indicator_moves[1], jnp.array([[[[True, False],
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

    def test_get_turns(self):
        states = go.new_states(2, batch_size=2)
        states = states.at[0, 2].set(True)
        self.assertEqual(go.get_turns(states), [True, False])

    def test_white_moves_second(self):
        state = go.new_states(4)
        state = go.next_states(state, go.to_indicator_actions([(0, 0)], state))
        self.assertTrue(jnp.alltrue(state[0, 2]))

    def test_black_and_white_moves_in_batch(self):
        states = go.new_states(2, batch_size=2)
        states = states.at[0, 2].set(True)
        self.assertEqual(go.get_turns(states), [True, False])
        states = go.next_states(states, go.to_indicator_actions([None, None], states))
        self.assertEqual(go.get_turns(states), [False, True])

    def test_pass_changes_turn(self):
        state = go.new_states(2)
        self.assertEqual(go.get_turns(state), [False])
        state = go.next_states(state, go.to_indicator_actions([None], state))
        self.assertEqual(go.get_turns(state), [True])

    def test_pass_sets_pass_layer(self):
        state = go.new_states(2)
        state = go.next_states(state, go.to_indicator_actions([None], state))
        self.assertTrue(jnp.alltrue(lax.eq(state[4]), jnp.ones_like(state[4])))

    def test_two_consecutive_passes_ends_game(self):
        state = go.new_states(2)
        self.assertTrue(jnp.alltrue(lax.eq(state[5]), jnp.zeros_like(state[5])))
        state = go.next_states(state, go.to_indicator_actions([None], state))
        self.assertTrue(jnp.alltrue(lax.eq(state[5]), jnp.zeros_like(state[5])))
        state = go.next_states(state, go.to_indicator_actions([None], state))
        self.assertTrue(jnp.alltrue(lax.eq(state[5]), jnp.ones_like(state[5])))

    def test_invalid_move_space_occupied_by_opponent_pieces(self):
        state = go.new_states(2)
        state = go.new_states(state, go.to_indicator_actions([(0, 0)], state))
        self.assertEqual(go.get_turns(state), [True])
        next_state = go.new_states(state, go.to_indicator_actions([(0, 0)], state))
        # Invalid moves don't change the state
        self.assertTrue(jnp.alltrue(lax.eq(next_state, state)))

    def test_invalid_move_space_occupied_by_own_pieces(self):
        state = go.new_states(2)
        state = go.new_states(state, go.to_indicator_actions([(0, 0)], state))
        state = go.new_states(state, go.to_indicator_actions([None], state))
        self.assertEqual(go.get_turns(state), [False])
        next_state = go.new_states(state, go.to_indicator_actions([(0, 0)], state))
        # Invalid moves don't change the state
        self.assertTrue(jnp.alltrue(lax.eq(next_state, state)))

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

import unittest

import chex
import jax.numpy as jnp
import jax.random
import numpy as np

import gojax
import rng
import serialize


class RNGTestCase(chex.TestCase):
    def test_sample_actions_from_logits(self):
        sampled_actions = rng.sample_actions(gojax.new_states(board_size=3, batch_size=2),
                                             logits=jnp.array([[10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                               [0, 10, 0, 0, 0, 0, 0, 0, 0, 0]],
                                                              dtype=float),
                                             rng_key=jax.random.PRNGKey(42))
        np.testing.assert_array_equal(sampled_actions, [[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                        [[0, 1, 0], [0, 0, 0], [0, 0, 0]]])
        chex.assert_type(sampled_actions, bool)

    def test_only_piece_action_from_logits(self):
        states = serialize.decode_states("""
                                    B B B
                                    B B B
                                    B B _
                                    TURN=W
                                    """)
        sampled_actions = rng.sample_actions(states, logits=jnp.zeros((1, 10), dtype=float),
                                             rng_key=jax.random.PRNGKey(10))
        np.testing.assert_array_equal(sampled_actions, [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]])

    def test_pass_from_state_with_one_piece_action(self):
        states = serialize.decode_states("""
                                    B B B
                                    B B B
                                    B B _
                                    TURN=W
                                    """)
        sampled_actions = rng.sample_actions(states, logits=jnp.zeros((1, 10), dtype=float),
                                             rng_key=jax.random.PRNGKey(42))
        np.testing.assert_array_equal(sampled_actions, [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

    def test_sample_actions(self):
        states = serialize.decode_states("""
                                    W W W
                                    W W _
                                    W W _
                                    """)
        sampled_actions = rng.sample_actions(states, logits=jnp.zeros((1, 10)),
                                             rng_key=jax.random.PRNGKey(10))
        np.testing.assert_array_equal(sampled_actions, [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]])

    def test_sample_random_state(self):
        state = rng.sample_random_state(board_size=3, batch_size=1, num_steps=4,
                                        logits=jnp.zeros((1, 10)), rng_key=jax.random.PRNGKey(42))
        np.testing.assert_array_equal(state, serialize.decode_states("""
                                                                W B _
                                                                _ _ _
                                                                _ W B
                                                                """))

    def test_sample_random_state(self):
        state = rng.sample_random_state(board_size=9, batch_size=1, num_steps=162,
                                        logits=jnp.zeros((1, 82)), rng_key=jax.random.PRNGKey(42))
        print(gojax.get_pretty_string(state[0]))
        np.testing.assert_array_equal(state, serialize.decode_states("""
                                                                W W W B _ W W B B
                                                                W W _ _ W W B W _
                                                                _ W W W W B B _ W
                                                                W _ W W B B B W W
                                                                _ B W W B B B B B
                                                                B _ W B B B B B B
                                                                _ B W B B B W B B
                                                                _ W W B B B W W W
                                                                _ _ W W _ W W W W
                                                                END=T;PASS=T
                                                                """))


if __name__ == '__main__':
    unittest.main()

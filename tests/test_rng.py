import unittest

import chex
import jax.numpy as jnp
import jax.random
import numpy as np

import gojax
import rng
import serialize


class RNGTestCase(chex.TestCase):
    def test_sample_actions_from_low_entropy_logits(self):
        sampled_actions = rng.sample_actions(gojax.new_states(board_size=3, batch_size=2),
                                             logits=jnp.array([[10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                               [0, 10, 0, 0, 0, 0, 0, 0, 0, 0]],
                                                              dtype=float),
                                             rng_key=jax.random.PRNGKey(42))
        np.testing.assert_array_equal(sampled_actions, [[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                        [[0, 1, 0], [0, 0, 0], [0, 0, 0]]])
        chex.assert_type(sampled_actions, bool)

    def test_sample_from_zero_logits(self):
        sampled_actions = rng.sample_actions(gojax.new_states(board_size=3),
                                             logits=jnp.zeros((1, 10), dtype=float),
                                             rng_key=jax.random.PRNGKey(10))
        np.testing.assert_array_equal(sampled_actions, [[[0, 0, 1], [0, 0, 0], [0, 0, 0]]])

    def test_sample_random_state(self):
        state = rng.sample_random_state(board_size=3, batch_size=1, num_steps=4,
                                        logits=jnp.zeros((1, 10)), rng_key=jax.random.PRNGKey(42))
        np.testing.assert_array_equal(state, serialize.decode_states("""
                                                                _ _ _
                                                                _ _ _
                                                                _ B B
                                                                """))


if __name__ == '__main__':
    unittest.main()

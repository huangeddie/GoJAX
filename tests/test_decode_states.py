import unittest

import gojax
import jax.numpy as jnp
import numpy as np


class DecodeStatesTestCase(unittest.TestCase):
    def test_shape(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str)
        self.assertEqual((1, 6, 4, 4), state.shape)

    def test_decode_multi_states(self):
        states = gojax.decode_states("""
                                   _ _
                                   _ _

                                   _ _
                                   _ _
                                   """)
        np.testing.assert_array_equal(states, jnp.zeros((2, gojax.NUM_CHANNELS, 2, 2)))

    def test_decode_new_state(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str)
        np.testing.assert_array_equal(state, jnp.zeros_like(state))

    def test_turn(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str, gojax.WHITES_TURN)
        np.testing.assert_array_equal(state[0, gojax.TURN_CHANNEL_INDEX],
                                      jnp.ones_like(state[0, gojax.TURN_CHANNEL_INDEX]))

    def test_one_piece(self):
        state_str = """
                    B _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str)
        self.assertTrue(state[0, gojax.BLACK_CHANNEL_INDEX, 0, 0])

    def test_two_pieces(self):
        state_str = """
                    B _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ W
                    """
        state = gojax.decode_states(state_str)
        self.assertTrue(state[0, gojax.BLACK_CHANNEL_INDEX, 0, 0])
        self.assertTrue(state[0, gojax.WHITE_CHANNEL_INDEX, 3, 3])

    def test_pass_default_false(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str)
        np.testing.assert_array_equal(state[0, gojax.PASS_CHANNEL_INDEX],
                                      jnp.zeros_like(state[0, gojax.PASS_CHANNEL_INDEX]))

    def test_pass(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str, passed=True)
        np.testing.assert_array_equal(state[0, gojax.PASS_CHANNEL_INDEX],
                                      jnp.ones_like(state[0, gojax.PASS_CHANNEL_INDEX]))

    def test_ended_default_false(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str)
        np.testing.assert_array_equal(state[0, gojax.END_CHANNEL_INDEX],
                                      jnp.zeros_like(state[0, gojax.END_CHANNEL_INDEX]))

    def test_ended(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str, ended=True)
        np.testing.assert_array_equal(state[0, gojax.END_CHANNEL_INDEX],
                                      jnp.ones_like(state[0, gojax.END_CHANNEL_INDEX]))

    def test_komi(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str, komi=(0, 0))
        self.assertTrue(state[0, gojax.INVALID_CHANNEL_INDEX, 0, 0])


if __name__ == '__main__':
    unittest.main()

"""Tests serialize.py"""
# pylint: disable=missing-function-docstring,missing-class-docstring,too-many-public-methods,duplicate-code
import unittest

import jax.numpy as jnp
import numpy as np

import gojax


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

    def test_multi_states(self):
        states = gojax.decode_states("""
                                   _ _
                                   _ _

                                   _ _
                                   _ _
                                   """)
        np.testing.assert_array_equal(states,
                                      jnp.zeros((2, gojax.NUM_CHANNELS, 2, 2)))

    def test_multi_states_with_macro(self):
        states = gojax.decode_states("""
                                   _ _
                                   _ _

                                   _ _
                                   _ _
                                   TURN=W
                                   """)
        np.testing.assert_array_equal(gojax.get_turns(states), [False, True])

    def test_new_state(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str)
        np.testing.assert_array_equal(state, jnp.zeros_like(state))

    def test_macro_turn(self):
        state = gojax.decode_states("""
                                    _ _
                                    _ _
                                    TURN=WHITE
                                    """)
        np.testing.assert_array_equal(gojax.get_turns(state), [True])

    def test_macro_pass(self):
        state = gojax.decode_states("""
                                    _ _
                                    _ _
                                    PASS=1
                                    """)
        np.testing.assert_array_equal(gojax.get_passes(state), [True])

    def test_macro_end(self):
        state = gojax.decode_states("""
                                    _ _
                                    _ _
                                    END=T
                                    """)
        np.testing.assert_array_equal(gojax.get_ended(state), [True])

    def test_all_macros(self):
        state = gojax.decode_states("""
                                    _ _
                                    _ _
                                    TURN=W;PASS=TRUE;END=True
                                    """)
        np.testing.assert_array_equal(gojax.get_turns(state), [True])
        np.testing.assert_array_equal(gojax.get_passes(state), [True])
        np.testing.assert_array_equal(gojax.get_ended(state), [True])

    def test_turn(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str, gojax.WHITES_TURN)
        np.testing.assert_array_equal(
            state[0, gojax.TURN_CHANNEL_INDEX],
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
        np.testing.assert_array_equal(
            state[0, gojax.PASS_CHANNEL_INDEX],
            jnp.zeros_like(state[0, gojax.PASS_CHANNEL_INDEX]))

    def test_pass(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str, passed=True)
        np.testing.assert_array_equal(
            state[0, gojax.PASS_CHANNEL_INDEX],
            jnp.ones_like(state[0, gojax.PASS_CHANNEL_INDEX]))

    def test_ended_default_false(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str)
        np.testing.assert_array_equal(
            state[0, gojax.END_CHANNEL_INDEX],
            jnp.zeros_like(state[0, gojax.END_CHANNEL_INDEX]))

    def test_ended(self):
        state_str = """
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    _ _ _ _
                    """
        state = gojax.decode_states(state_str, ended=True)
        np.testing.assert_array_equal(
            state[0, gojax.END_CHANNEL_INDEX],
            jnp.ones_like(state[0, gojax.END_CHANNEL_INDEX]))


class EncodeStatesTestCase(unittest.TestCase):

    def test_state(self):
        states = gojax.new_states(3)
        states = states.at[0, gojax.BLACK_CHANNEL_INDEX, 0, 0].set(True)
        states = states.at[0, gojax.WHITE_CHANNEL_INDEX, 0, 1].set(True)
        states = states.at[0, gojax.KILLED_CHANNEL_INDEX, 0, 2].set(True)
        states = states.at[0, gojax.TURN_CHANNEL_INDEX].set(True)
        states = states.at[0, gojax.PASS_CHANNEL_INDEX].set(True)
        states = states.at[0, gojax.END_CHANNEL_INDEX].set(True)
        self.assertEqual(gojax.encode_states(states),
                         'B W X \n_ _ _ \n_ _ _ \nTURN=W;PASS=T;END=T\n')

    def test_three_states(self):
        states = gojax.decode_states("""
                                    _ _ _
                                    _ _ _
                                    _ _ _
                                    
                                    _ _ _
                                    _ B _
                                    _ _ _
                                    TURN=W;PASS=T;END=T
                                    
                                    _ _ _
                                    _ W _
                                    _ _ _
                                    """)
        self.assertEqual(
            gojax.encode_states(states),
            "_ _ _ \n_ _ _ \n_ _ _ \n\n_ _ _ \n_ B _ \n_ _ _ \nTURN=W;PASS=T;END=T\n\n"
            "_ _ _ \n_ W _ \n_ _ _ \n")


if __name__ == '__main__':
    unittest.main()

"""Demo usage of `gojax`."""

from jax import numpy as jnp

from gojax import constants
from gojax import go

if __name__ == '__main__':
  state = go.new_states(7)
  while not jnp.alltrue(state[0, constants.END_CHANNEL_INDEX]):
    print(go.get_pretty_string(state[0]))
    USER_INPUT = input('row col: ').strip()
    # pylint: disable=invalid-name
    action = None
    if USER_INPUT:
      row, col = USER_INPUT.split()
      action = (int(row), int(col))
    state = go.next_states(state, go.action_2d_indices_to_indicator([action], state))

  print(go.get_pretty_string(state[0]))

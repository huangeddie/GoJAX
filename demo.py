from jax import numpy as jnp

import constants
import go

if __name__ == '__main__':
    state = go.new_states(7)
    while not jnp.alltrue(state[0, constants.END_CHANNEL_INDEX]):
        print(go.get_pretty_string(state[0]))
        move = input('row col: ').strip()
        action = None
        if move:
            row, col = move.split()
            action = (int(row), int(col))
        state = go.next_states(state, go.to_indicator_actions([action], state))

    print(go.get_pretty_string(state[0]))

import jax
from jax import numpy as jnp, lax

import gojax


def sample_all_actions(states, logits, rng_key):
    """
    Samples the all actions from the logits with probability equal to softmax(
    raw_action_logits).

    :param states: a batch array of N Go games.
    :param logits: an N x A float array of logits.
    :param rng_key: JAX RNG key.
    :return: an N x B x B boolean one-hot array representing the sampled action (all-false = pass).
    """
    action_1d = jax.random.categorical(rng_key, logits)
    one_hot_action_1d = jax.nn.one_hot(action_1d, logits.shape[1], dtype=bool)
    return jnp.reshape(one_hot_action_1d[:, :-1], (-1, states.shape[2], states.shape[3]))


def sample_non_occupied_indicator_actions(states, logits, rng_key):
    """
    Samples actions that are not currently occupied from the logits with probability equal to
    softmax(
    raw_action_logits).

    Ignores corresponding occupied space logits.

    :param states: a batch array of N Go games.
    :param logits: an N x A float array of logits.
    :param rng_key: JAX RNG key.
    :return: an N x B x B boolean one-hot array representing the sampled action (all-false = pass).
    """
    flattened_occupied = jnp.reshape(gojax.get_occupied_spaces(states),
                                     (-1, states.shape[2] * states.shape[3]))
    action_logits = jnp.where(
        jnp.append(flattened_occupied, jnp.zeros((len(states), 1), dtype=bool), axis=1),
        jnp.full_like(logits, float('-inf')), logits)
    action_1d = jax.random.categorical(rng_key, action_logits)
    one_hot_action_1d = jax.nn.one_hot(action_1d, action_logits.shape[1], dtype=bool)
    return jnp.reshape(one_hot_action_1d[:, :-1], (-1, states.shape[2], states.shape[3]))


def sample_next_states(step, states, logits, rng_key,
                       sampling_fn=sample_non_occupied_indicator_actions):
    """
    Samples the next state with probability softmax(logits) and RNG created by folding rng_key
    with step.

    :param step: an integer.
    :param states: a batch array of N Go games.
    :param logits: an N x A float array of logits.
    :param rng_key: JAX RNG key.
    :param sampling_fn: sampling function. Either sample_all_actions or
    sample_non_occupied_indicator_actions.
    :return: a batch array of N Go games.
    """
    return gojax.next_states(states, sampling_fn(states, logits, jax.random.fold_in(rng_key, step)))


def sample_random_state(board_size, batch_size, num_steps, logits, rng_key,
                        sampling_fn=sample_non_occupied_indicator_actions):
    """
    Samples a random state by with `num_steps` sequential uniform random actions.

    :param board_size: board size B (integer).
    :param batch_size: batch size N (integer).
    :param logits: an N x A float array of logits.
    :param num_steps: number of steps to take (integer).
    :param rng_key: JAX RNG key.
    :param sampling_fn: sampling function. Either sample_all_actions or
    sample_non_occupied_indicator_actions.
    :return: the final state of the game (trajectory).
    """
    return lax.fori_loop(0, num_steps,
                         jax.tree_util.Partial(sample_next_states, logits=logits, rng_key=rng_key,
                                               sampling_fn=sampling_fn),
                         gojax.new_states(board_size, batch_size))

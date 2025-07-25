import time
import jax
import blackjax
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
jnp = jax.numpy


def inference_loop(rng_key, kernel, initial_state, num_samples):

    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


inference_loop_multiple_chains = jax.pmap(
    inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3)
)


def sample_potential_fn(
    potential_fn,
    initial_params,
    bijectors,
    rng_key=None,
    num_warmup_steps=200,
    num_steps=1000,
    n_chains=4,
    target_acceptance_rate=0.8,
):
    if rng_key is None:
        rng_key = generate_key()

    if isinstance(bijectors, list):
        tmp = {}
        for k, bij in zip(initial_params.keys(), bijectors):
            tmp[k] = bij
        bijectors = tmp

    @jax.jit
    def unconstrained_potential_fn(args):
        new_args = {
            k: bijectors[k](args[k]) if k in bijectors else args[k] for k in args
        }

        log_det = sum(
            bijectors[k].forward_log_det_jacobian(args[k]).sum() for k in bijectors
        )
        return potential_fn(new_args) + log_det

    rng_key, warmup_key = jax.random.split(rng_key)
    initial_states, parameters = adapt_nuts(
        unconstrained_potential_fn,
        initial_params,
        num_warmup_steps,
        n_chains,
        target_acceptance_rate,
        warmup_key,
    )
    nuts = blackjax.nuts(unconstrained_potential_fn, **parameters)
    rng_key, sample_key = jax.random.split(rng_key)

    rng_key, sample_key = jax.random.split(rng_key)
    sample_keys = jax.random.split(sample_key, n_chains)

    chain_states = inference_loop_multiple_chains(
        sample_keys, nuts.step, initial_states, num_steps
    )
    for k, bij in bijectors.items():
        chain_states.position[k] = bij(chain_states.position[k])

    chain_states.position[list(chain_states.position)[0]].block_until_ready()

    return chain_states


def generate_key(key=None):
    if key is not None:
        return key
    return jax.random.PRNGKey(int(time.time() * 1e7))


def adapt_nuts(
    unconstrained_potential_fn,
    initial_params,
    num_warmup_steps=200,
    n_chains=4,
    target_acceptance_rate=0.8,
    rng_key=None,
):
    rng_key = generate_key(rng_key)
    adapt = blackjax.window_adaptation(
        blackjax.nuts,
        unconstrained_potential_fn,
        target_acceptance_rate=target_acceptance_rate,
    )

    (last_state, parameters), _ = adapt.run(
        rng_key, {k: v for k, v in initial_params.items()}, num_warmup_steps
    )
    nuts = blackjax.nuts(unconstrained_potential_fn, **parameters)
    initial_states = jax.vmap(nuts.init, in_axes=(0))(
        {
            k: jnp.einsum("...c->c...", v[..., None] * jnp.ones(n_chains))
            for k, v in last_state.position.items()
        }
    )

    return initial_states, parameters

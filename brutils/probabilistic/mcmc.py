import time
import jax
import blackjax
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
jnp = jax.numpy


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
        rng_key = jax.random.PRNGKey(int(time.time()))

    if isinstance(bijectors, list):
        tmp = {}
        for k, bij in zip(initial_params.keys(), bijectors):
            tmp[k] = bij
        bijectors = tmp

    # for k in initial_params.keys() - bijectors.keys():
    #     bijectors[k] = tfb.Identity()

    # bijectors = {k: bijectors[k] for k in initial_params.keys()}
    # bijectors = tfb.JointMap(bijectors)

    @jax.jit
    def unconstrained_potential_fn(args):
        new_args = {
            k: bijectors[k](args[k]) if k in bijectors else args[k] for k in args
        }

        log_det = sum(
            bijectors[k].forward_log_det_jacobian(args[k]).sum() for k in bijectors
        )
        return potential_fn(new_args) + log_det

    adapt = blackjax.window_adaptation(
        blackjax.nuts,
        unconstrained_potential_fn,
        target_acceptance_rate=target_acceptance_rate,
    )

    rng_key, warmup_key = jax.random.split(rng_key)
    (last_state, parameters), _ = adapt.run(
        warmup_key, initial_params, num_warmup_steps
    )
    kernel = jax.jit(blackjax.nuts(unconstrained_potential_fn, **parameters).step)
    # hmc_kernel = jax.jit(nuts.step)

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    keys = jax.random.split(rng_key, n_chains)
    chain_states = jax.vmap(inference_loop, in_axes=(0, None, None, None))(
        keys, kernel, last_state, num_steps
    )

    for k, bij in bijectors.items():
        chain_states.position[k] = bij(chain_states.position[k])

    chain_states.position[list(chain_states.position)[0]].block_until_ready()

    return chain_states


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

    def inference_loop_multiple_chains(
        rng_key, kernel, initial_state, num_samples, num_chains
    ):

        @jax.jit
        def one_step(states, rng_key):
            keys = jax.random.split(rng_key, num_chains)
            states, _ = jax.vmap(kernel)(keys, states)
            return states, states

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    rng_key, warmup_key = jax.random.split(rng_key)
    rng_key, nuts, initial_states = adapt_nuts(
        unconstrained_potential_fn,
        initial_params,
        num_warmup_steps,
        n_chains,
        target_acceptance_rate,
        warmup_key,
    )

    rng_key, sample_key = jax.random.split(rng_key)
    chain_states = inference_loop_multiple_chains(
        sample_key, jax.jit(nuts.step), initial_states, num_steps, n_chains
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

    return rng_key, nuts, initial_states


def combine_blackjax_hmc_results(results):
    keys = results[0].position.keys()
    position = {}
    logdensity_grad = {}
    for key in keys:
        position[key] = jnp.concat([r.position[key] for r in results], axis=0)
        logdensity_grad[key] = jnp.concat(
            [r.logdensity_grad[key] for r in results], axis=0
        )
    log_density = jnp.concat([r.logdensity for r in results], axis=0)
    return blackjax.mcmc.hmc.HMCState(position, log_density, logdensity_grad)

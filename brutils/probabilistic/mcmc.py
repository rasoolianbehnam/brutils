import time
import jax
import blackjax
import arviz as az
from tensorflow_probability.substrates import jax as tfp

tfb = tfp.bijectors
jnp = jax.numpy


def check_rhat(trace, *, thresh=None, ignore_nan=False, bijectors=None):
    trace = dict(trace)
    if bijectors is not None:
        for k in bijectors:
            trace[k] = bijectors[k].inverse(trace[k])

    def fn(x):
        y = jnp.abs(tfp.mcmc.potential_scale_reduction(x) - 1)
        out = jnp.array(y).squeeze()
        if ignore_nan and jnp.isnan(out).sum() > 0:
            out = jnp.where(jnp.isnan(out), 0, out)
        return out.max()

    out = jax.tree.map(fn, trace)
    if thresh is not None:
        out = jax.tree.map(lambda x: x < thresh, out)
    return out


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
    bijectors=None,
    rng_key=None,
    num_warmup_steps=200,
    num_steps=1000,
    num_chains=4,
    target_acceptance_rate=0.8,
):
    if rng_key is None:
        rng_key = generate_key()

    if isinstance(bijectors, (list, tuple)):
        tmp = {}
        for k, bij in zip(initial_params.keys(), bijectors):
            tmp[k] = bij
        bijectors = tmp

    for k in initial_params:
        initial_params[k] = bijectors[k].inverse(initial_params[k])

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
        num_chains,
        target_acceptance_rate,
        warmup_key,
    )
    nuts = blackjax.nuts(unconstrained_potential_fn, **parameters)
    rng_key, sample_key = jax.random.split(rng_key)

    rng_key, sample_key = jax.random.split(rng_key)
    sample_keys = jax.random.split(sample_key, num_chains)

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
    num_chains=4,
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
            k: jnp.einsum("...c->c...", v[..., None] * jnp.ones(num_chains))
            for k, v in last_state.position.items()
        }
    )

    return initial_states, parameters


def sample_tfp(target, initial_states=None, bijectors=None, **sampling_kwargs):
    if bijectors is None:
        bijectors = target.experimental_default_event_space_bijector().bijectors
    if initial_states is None:
        initial_states = target.sample(seed=generate_key())._asdict()

    log_prob_fn = lambda kwargs: target.log_prob(**kwargs)
    out = sample_potential_fn(log_prob_fn, initial_states, bijectors, **sampling_kwargs)
    r_hat = check_rhat(out.position)
    return az.from_dict(
        out.position, log_likelihood=out.logdensity_grad, sample_stats=r_hat
    )


def to_dict(az_dataset):
    return {
        k: jnp.array(v["data"]) for k, v in az_dataset.to_dict()["data_vars"].items()
    }


def asdict(d):
    return to_dict(d.posterior)


az.data.inference_data.InferenceData.dict = property(asdict)

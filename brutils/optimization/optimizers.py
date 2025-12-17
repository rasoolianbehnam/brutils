import jax
import time

jnp = jax.numpy
random = jax.random


def differential_evolution(f, population, k_max, p=0.5, w=1):
    m, n, *_ = population.shape
    p_ = jnp.empty_like(population)
    idx = jnp.arange(m)
    ii = jnp.arange(n)
    W = jnp.ones(m) / (m - 1)
    key = jax.random.PRNGKey(int(time.time() * 1e7))

    @jax.jit
    def update_population(key, population, k):
        x = population[k]
        weights = W.at[k].set(0)
        a, b, c = jax.random.choice(key, idx, shape=(3,), p=weights, replace=False)
        z = population[a] + w * (population[b] - population[c])
        cond = (jax.random.randint(key, minval=0, maxval=n, shape=(n,)) == ii) | (
            jax.random.uniform(key, n) < p
        )
        out = jnp.where(cond, z, x)
        return out

    update_population = jax.jit(jax.vmap(update_population, in_axes=(0, None, 0)))
    f_fn = jax.jit(jax.vmap(f))

    @jax.jit
    def body_fn(population, key):
        key = jax.random.split(key, m)
        p_ = update_population(key, population, idx)
        population = jnp.where((f_fn(population) < f_fn(p_))[:, None], population, p_)
        return population, 0

    population, _ = jax.lax.scan(body_fn, population, jax.random.split(key, k_max))
    return population

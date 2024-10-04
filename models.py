import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dx
from jaxtyping import Float, Array, Int, PRNGKeyArray

class CNNEmulator(eqx.Module):
    layers: list

    def __init__(self, key: PRNGKeyArray, hidden_dim: Int = 4):
        self.layers = eqx.nn.Sequential([
            eqx.nn.Conv2d(2, hidden_dim, 3, 1, 1, key=key),
            eqx.nn.Lambda(jnp.tanh),
            eqx.nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, key=key),
            eqx.nn.Lambda(jnp.tanh),
            eqx.nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, key=key),
            eqx.nn.Lambda(jnp.tanh),
            eqx.nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, key=key),
            eqx.nn.Lambda(jnp.tanh),
            eqx.nn.Conv2d(hidden_dim, 1, 3, 1, 1, key=key)
        ])

    def __call__(self, x: Float[Array, "2 n_res n_res"]) -> Float[Array, "1 n_res n_res"]:
        out = x
        for layer in self.layers:
            out = layer(out)

        return out

    def rollout(self, x: Float[Array, "2 n_res n_res"], n_step: Int) -> Float[Array, "n_step n_res n_res"]:
        result = [x]
        for i in range(n_step):
            x = jnp.concatenate([x[1:], self(x)], axis=0)
            result.append(x[1:])
        return jnp.stack(result)
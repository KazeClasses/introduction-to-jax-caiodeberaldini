import matplotlib.pyplot as plt
import diffrax as dx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree, Shaped, ArrayLike, Int
import equinox as eqx


class PendulumSimulation(eqx.Module):

    box_size: Float = 5.0
    ball_size: Float = 0.2
    image_size: Int = 64
    time: Float = 10.0
    dt: Float = 0.01
    save_interval: Int = 10

    @property
    def n_frames(self) -> Int:
        return int(self.time / (self.save_interval * self.dt))

    def __init__(
        self, box_size: Float = 5.0, ball_size: Float = 0.2, image_size: Int = 64
    ):
        self.box_size = box_size
        self.ball_size = ball_size
        self.image_size = image_size

    def ODE_system(
        self, t: Float, y: Float[Array, " 2"], args: PyTree
    ) -> Float[Array, " 2"]:
        angle, angular_velocity = y
        gravity, length = args
        return [angular_velocity, -gravity / length * jnp.sin(angle)]

    def simulate_pendulum(
        self,
        initial_angle: Float,
        initial_velocity: Float,
        gravity: Float,
        length: Float,
    ):
        # Fill in detail here
        term = dx.ODETerm(self.ODE_system)
        solver = dx.Dopri5()
        initial_state = [initial_angle, initial_velocity]
        args = [gravity, length]
        saveat = dx.SaveAt(ts=jnp.arange(0, self.time, self.save_interval * self.dt))
        sol = dx.diffeqsolve(
            term,
            solver,
            t0=0,
            t1=self.time,
            y0=initial_state,
            args=args,
            dt0=self.dt,
            saveat=saveat,
        )
        return sol

    def render_pendulum(
        self,
        angle: Float,
        angular_velocity: Float,
        length: Float,
    ) -> Float[Array, " n_res n_res"]:
        image = jnp.zeros((self.image_size, self.image_size)).reshape(-1)
        grid_x, grid_y = jnp.meshgrid(
            jnp.arange(self.image_size) - self.image_size // 2, 
            jnp.arange(self.image_size) - self.image_size // 2
        ) # Make x-y coordinate

        grid_x, grid_y = grid_x / (self.image_size * self.box_size), grid_y / (self.image_size * self.box_size)

        coordinates = jnp.stack([grid_x, grid_y], axis=-1).reshape(-1, 2) # Stack the coordination

        position = jnp.array([length * jnp.sin(angle), length * jnp.cos(angle)]) # Compute the position of the ball

        distance = jnp.linalg.norm(coordinates - position, axis=-1) # Compute distance from pixels to ball
        image = jnp.array(distance < length).astype(jnp.bool_) # Set pixel values correspondingly
        return image.reshape(self.image_size, self.image_size)

    def generate_dataset(
        self,
        n_sims: Int,
        gravity: Float,
        length: Float,
    ) -> tuple[
        Float[Array, " n_samples 2 n_res n_res"],
        Float[Array, " n_samples 1 n_res n_res"],
    ]:
        inputs = []
        outputs = []
        for i in range(n_sims):
            # Generate random initial conditions
            initial_angle = jax.random.uniform(jax.random.PRNGKey(i), minval=jnp.pi, maxval=jnp.pi) # Fill in detail here
            initial_velocity = jax.random.uniform(jax.random.PRNGKey(i * 1029423), minval=-10.0, maxval=10.0) # Fill in detail here
            solution = self.simulate_pendulum(
                initial_angle, initial_velocity, gravity, length
            )
            frames = jax.vmap(self.render_pendulum, in_axes=(0, 0, None))(solution.ys[0], solution.ys[1], length) # Render frames
            inputs.append(jnp.stack([frames[:-2], frames[1:-1]], axis=1))
            outputs.append(frames[2:])
        return jnp.stack(inputs).reshape(
            -1, 2, self.image_size, self.image_size
        ).astype(jnp.float32), jnp.stack(outputs).reshape(-1, 1, self.image_size, self.image_size).astype(jnp.float32)


if __name__ == "__main__":
    pendulum = PendulumSimulation(image_size=64)
    sol = pendulum.simulate_pendulum(0.0, 0.0, 9.8, 1.0)
    image = pendulum.render_pendulum(sol.ys[0][0], sol.ys[1][0], 1.0)
    dataset = pendulum.generate_dataset(5, 9.8, 1.0)
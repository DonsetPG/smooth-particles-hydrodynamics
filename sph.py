import matplotlib.pyplot as plt
import numpy as np
from numba import njit, typed
from scipy.spatial import cKDTree
from tqdm import tqdm


@njit
def spiky_kernel_pow2(dst, radius, scaling):
    if dst < radius:
        v = radius - dst
        return v * v * scaling
    return 0.0


@njit
def spiky_kernel_pow3(dst, radius, scaling):
    if dst < radius:
        v = radius - dst
        return v * v * v * scaling
    return 0.0


@njit
def derivative_spiky_pow2(dst, radius, scaling):
    if dst <= radius:
        v = radius - dst
        return -v * scaling
    return 0.0


@njit
def derivative_spiky_pow3(dst, radius, scaling):
    if dst <= radius:
        v = radius - dst
        return -v * v * scaling
    return 0.0


@njit
def smoothing_kernel_poly6(dst, radius, scaling):
    if dst < radius:
        v = radius * radius - dst * dst
        return v * v * v * scaling
    return 0.0


@njit
def vec_length(v):
    return (v[0] * v[0] + v[1] * v[1]) ** 0.5


@njit
def vec_normalize(v):
    l = vec_length(v)
    if l > 1e-8:
        return v / l
    else:
        return np.array([0.0, 1.0])


@njit
def compute_densities_numba(
    predicted_positions,
    neighbors,
    smoothing_radius,
    spiky_pow2_scaling,
    spiky_pow3_scaling,
    densities,
):
    num_particles = predicted_positions.shape[0]
    for i in range(num_particles):
        density = 0.0
        near_density = 0.0
        pos_i0 = predicted_positions[i, 0]
        pos_i1 = predicted_positions[i, 1]
        neigh_indices = neighbors[i]
        for j in range(neigh_indices.shape[0]):
            idx = neigh_indices[j]
            dx = predicted_positions[idx, 0] - pos_i0
            dy = predicted_positions[idx, 1] - pos_i1
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= smoothing_radius:
                density += spiky_kernel_pow2(dist, smoothing_radius, spiky_pow2_scaling)
                near_density += spiky_kernel_pow3(
                    dist, smoothing_radius, spiky_pow3_scaling
                )
        densities[i, 0] = density
        densities[i, 1] = near_density


@njit
def compute_pressure_force_numba(
    predicted_positions,
    densities,
    neighbors,
    smoothing_radius,
    spiky_pow2_deriv_scaling,
    spiky_pow3_deriv_scaling,
    target_density,
    pressure_multiplier,
    near_pressure_multiplier,
    delta_time,
    velocities,
    pressure_force_contrib,
):
    num_particles = predicted_positions.shape[0]
    density_floor = 0.1 * target_density

    for i in range(num_particles):
        pos_i0 = predicted_positions[i, 0]
        pos_i1 = predicted_positions[i, 1]
        density_i = densities[i, 0]
        density_near_i = densities[i, 1]

        density_eff = max(density_i, density_floor)

        # Calculate pressure terms for particle i
        pressure_i = (density_i - target_density) * pressure_multiplier
        near_pressure_i = near_pressure_multiplier * density_near_i

        # Accumulate total force on particle i from neighbors
        total_force_x = 0.0
        total_force_y = 0.0

        neigh_indices = neighbors[i]
        for j in range(neigh_indices.shape[0]):
            idx = neigh_indices[j]
            if idx == i:
                continue

            dx = predicted_positions[idx, 0] - pos_i0
            dy = predicted_positions[idx, 1] - pos_i1
            dist_sq = dx * dx + dy * dy

            # Check distance against smoothing radius squared for efficiency
            if dist_sq > smoothing_radius * smoothing_radius or dist_sq < 1e-12:
                continue

            dist = dist_sq**0.5

            dir_x = dx / dist
            dir_y = dy / dist

            # Get neighbor properties
            neigh_density = densities[idx, 0]
            neigh_density_near = densities[idx, 1]

            neigh_density_eff = max(neigh_density, density_floor)
            neigh_near_density_eff = max(neigh_density_near, density_floor)

            # Calculate pressure terms for neighbor j
            pressure_j = (neigh_density - target_density) * pressure_multiplier
            near_pressure_j = near_pressure_multiplier * neigh_density_near

            # Calculate shared pressure terms (symmetric form)
            shared_pressure = (pressure_i + pressure_j) * 0.5
            shared_near_pressure = (near_pressure_i + near_pressure_j) * 0.5

            # Standard Pressure Force Magnitude Contribution
            # F_press = - sum_j m * (P_i + P_j)/(2 * rho_j) * Grad(W_ij)
            pressure_force_mag = (
                shared_pressure
                * derivative_spiky_pow2(
                    dist, smoothing_radius, spiky_pow2_deriv_scaling
                )
                / neigh_density_eff
            )

            # Near Pressure Force Magnitude Contribution
            near_pressure_force_mag = (
                shared_near_pressure
                * derivative_spiky_pow3(
                    dist, smoothing_radius, spiky_pow3_deriv_scaling
                )
                / neigh_near_density_eff
            )

            # Total magnitude of repulsive force from this neighbor
            total_combined_magnitude = pressure_force_mag + near_pressure_force_mag

            total_force_x += dir_x * total_combined_magnitude
            total_force_y += dir_y * total_combined_magnitude

        # Convert total force to acceleration: a = F / rho_i
        acceleration_x = total_force_x / density_eff
        acceleration_y = total_force_y / density_eff

        # Record acceleration contribution for debugging node 0
        if i == 0:
            pressure_force_contrib[0] = acceleration_x
            pressure_force_contrib[1] = acceleration_y

        # Update velocity using acceleration
        velocities[i, 0] += acceleration_x * delta_time
        velocities[i, 1] += acceleration_y * delta_time


@njit
def compute_viscosity_numba(
    predicted_positions,
    velocities,
    neighbors,
    smoothing_radius,
    poly6_scaling,
    viscosity_strength,
    delta_time,
    viscosity_force_contrib,
):
    num_particles = predicted_positions.shape[0]
    for i in range(num_particles):
        pos_i0 = predicted_positions[i, 0]
        pos_i1 = predicted_positions[i, 1]
        vel_i0 = velocities[i, 0]
        vel_i1 = velocities[i, 1]
        visc_force_x = 0.0
        visc_force_y = 0.0
        neigh_indices = neighbors[i]
        for j in range(neigh_indices.shape[0]):
            idx = neigh_indices[j]
            if idx == i:
                continue
            dx = predicted_positions[idx, 0] - pos_i0
            dy = predicted_positions[idx, 1] - pos_i1
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > smoothing_radius:
                continue
            visc = smoothing_kernel_poly6(dist, smoothing_radius, poly6_scaling)
            visc_force_x += (velocities[idx, 0] - vel_i0) * visc
            visc_force_y += (velocities[idx, 1] - vel_i1) * visc
        # For node 0, record the viscosity force contribution
        if i == 0:
            viscosity_force_contrib[0] = visc_force_x * viscosity_strength
            viscosity_force_contrib[1] = visc_force_y * viscosity_strength
        velocities[i, 0] += visc_force_x * viscosity_strength * delta_time
        velocities[i, 1] += visc_force_y * viscosity_strength * delta_time


class FluidSim2D:
    def __init__(
        self,
        num_particles: int = 100,
        spawn_area_size: tuple = (10.0, 10.0),
        gravity: float = -9.81,
        delta_time: float = 1.0 / 60.0,
        collision_damping: float = 0.99,
        smoothing_radius: float = 2.0,
        target_density: float = 10.0,
        pressure_multiplier: float = 1.0,
        near_pressure_multiplier: float = 1.0,
        viscosity_strength: float = 0.01,
        bounds_size: tuple = (50, 50),
        interaction_input_point: tuple = (0, 0),
        interaction_input_strength: float = 0.0,
        interaction_input_radius: float = 5.0,
        poly6_scaling_factor: float = None,
        spiky_pow3_scaling_factor: float = None,
        spiky_pow2_scaling_factor: float = None,
        spiky_pow3_derivative_scaling_factor: float = None,
        spiky_pow2_derivative_scaling_factor: float = None,
    ):
        self.num_particles = num_particles
        self.gravity = gravity
        self.spawn_area_size = np.array(spawn_area_size, dtype=np.float64)
        self.delta_time = delta_time
        self.collision_damping = collision_damping
        self.smoothing_radius = smoothing_radius
        self.target_density = target_density
        self.pressure_multiplier = pressure_multiplier
        self.near_pressure_multiplier = near_pressure_multiplier
        self.viscosity_strength = viscosity_strength
        self.bounds_size = np.array(bounds_size, dtype=np.float64)
        self.interaction_input_point = np.array(
            interaction_input_point, dtype=np.float64
        )
        self.interaction_input_strength = interaction_input_strength
        self.interaction_input_radius = interaction_input_radius

        # Maximum allowed velocity to help prevent runaway speeds
        self.max_velocity = 15.0

        self.poly6_scaling_factor = (
            4.0 / (np.pi * (smoothing_radius**8))
            if poly6_scaling_factor is None
            else poly6_scaling_factor
        )
        self.spiky_pow3_scaling_factor = (
            10.0 / (np.pi * (smoothing_radius**5))
            if spiky_pow3_scaling_factor is None
            else spiky_pow3_scaling_factor
        )
        self.spiky_pow2_scaling_factor = (
            6.0 / (np.pi * (smoothing_radius**4))
            if spiky_pow2_scaling_factor is None
            else spiky_pow2_scaling_factor
        )
        self.spiky_pow3_derivative_scaling_factor = (
            30.0 / (np.pi * (smoothing_radius**5))
            if spiky_pow3_derivative_scaling_factor is None
            else spiky_pow3_derivative_scaling_factor
        )
        self.spiky_pow2_derivative_scaling_factor = (
            12.0 / (np.pi * (smoothing_radius**4))
            if spiky_pow2_derivative_scaling_factor is None
            else spiky_pow2_derivative_scaling_factor
        )

        # Allocate buffers
        self.positions = np.zeros((num_particles, 2), dtype=np.float64)
        self.predicted_positions = np.zeros((num_particles, 2), dtype=np.float64)
        self.velocities = np.zeros((num_particles, 2), dtype=np.float64)
        self.densities = np.zeros((num_particles, 2), dtype=np.float64)

        # Initialize positions and velocities
        aspect_ratio = (
            self.spawn_area_size[0] / self.spawn_area_size[1]
            if self.spawn_area_size[1] > 1e-6
            else 1.0
        )
        cols = int(np.ceil(np.sqrt(num_particles * aspect_ratio)))
        # Prevent cols from being zero if num_particles is zero
        if cols == 0:
            cols = 1
        rows = int(np.ceil(num_particles / float(cols)))
        if rows == 0:
            rows = 1

        # Calculate spacing needed to fit the grid in the spawn area
        dx = self.spawn_area_size[0] / cols
        dy = self.spawn_area_size[1] / rows

        start_x = -self.spawn_area_size[0] / 2.0
        start_y = -self.spawn_area_size[1] / 2.0

        # Place particles
        current_particle = 0
        for r in range(rows):
            for c in range(cols):
                if current_particle < num_particles:
                    # Position is center of the grid cell
                    self.positions[current_particle, 0] = start_x + c * dx + dx / 2.0
                    self.positions[current_particle, 1] = start_y + r * dy + dy / 2.0
                    current_particle += 1
                else:
                    break
            if current_particle >= num_particles:
                break

        initial_speed = 0.0
        self.velocities = np.zeros((num_particles, 2), dtype=np.float64)
        self.velocities[:, 0] = -initial_speed
        self.predicted_positions[:] = self.positions.copy()

    def calculate_total_kinetic_energy(self):
        """Calculates the total kinetic energy of the system."""
        speeds_sq = np.sum(self.velocities**2, axis=1)
        total_ke = 0.5 * np.sum(speeds_sq)
        return total_ke

    def calculate_total_potential_energy(self):
        """Calculates the total gravitational potential energy of the system."""
        g_magnitude = abs(self.gravity)
        heights = self.positions[:, 1] + self.bounds_size[1] / 2
        total_pe = g_magnitude * np.sum(heights)
        return total_pe

    def external_forces(self):
        gravity_accel = np.array([0.0, self.gravity], dtype=np.float64)
        self.velocities += gravity_accel * self.delta_time

        prediction_factor = 1 / 120.0
        self.predicted_positions = self.positions + self.velocities * prediction_factor
        # For node 0
        self.ext_force_contrib = gravity_accel * self.delta_time

    def handle_collisions(self):
        half_size = self.bounds_size * 0.5
        for i in range(self.num_particles):
            pos = self.positions[i]
            vel = self.velocities[i]
            edge_dst = half_size - np.abs(pos)
            if edge_dst[0] <= 0.0:
                pos[0] = half_size[0] * np.sign(pos[0])
                vel[0] *= -self.collision_damping
            if edge_dst[1] <= 0.0:
                pos[1] = half_size[1] * np.sign(pos[1])
                vel[1] *= -self.collision_damping
            self.positions[i] = pos
            self.velocities[i] = vel

    def update_positions(self):
        self.positions += self.velocities * self.delta_time
        self.handle_collisions()

    def step_simulation(self):
        # Apply external forces and predict new positions
        self.external_forces()

        # Build a KDTree from predicted positions and query neighbors within smoothing radius
        tree = cKDTree(self.predicted_positions)
        neighbor_lists = tree.query_ball_point(
            self.predicted_positions, r=self.smoothing_radius
        )
        # Convert neighbor_lists to a typed list for numba
        typed_neighbors = typed.List()
        for lst in neighbor_lists:
            typed_neighbors.append(np.array(lst, dtype=np.int64))

        compute_densities_numba(
            self.predicted_positions,
            typed_neighbors,
            self.smoothing_radius,
            self.spiky_pow2_scaling_factor,
            self.spiky_pow3_scaling_factor,
            self.densities,
        )

        # Prepare arrays to capture force contributions for node 0
        pressure_force_contrib = np.zeros(2, dtype=np.float64)
        viscosity_force_contrib = np.zeros(2, dtype=np.float64)

        compute_pressure_force_numba(
            self.predicted_positions,
            self.densities,
            typed_neighbors,
            self.smoothing_radius,
            self.spiky_pow2_derivative_scaling_factor,
            self.spiky_pow3_derivative_scaling_factor,
            self.target_density,
            self.pressure_multiplier,
            self.near_pressure_multiplier,
            self.delta_time,
            self.velocities,
            pressure_force_contrib,
        )

        compute_viscosity_numba(
            self.predicted_positions,
            self.velocities,
            typed_neighbors,
            self.smoothing_radius,
            self.poly6_scaling_factor,
            self.viscosity_strength,
            self.delta_time,
            viscosity_force_contrib,
        )

        # Clamp velocities
        speeds = np.linalg.norm(self.velocities, axis=1)
        mask = speeds > self.max_velocity
        if np.any(mask):
            self.velocities[mask] = (
                self.velocities[mask].T * (self.max_velocity / speeds[mask])
            ).T

        self.update_positions()

        # Return the force contributions for debugging
        return (
            self.ext_force_contrib / self.delta_time,
            pressure_force_contrib,
            viscosity_force_contrib,
        )


if __name__ == "__main__":
    sim = FluidSim2D(
        num_particles=5000,
        collision_damping=0.95,
        gravity=-10,
        smoothing_radius=0.3,
        target_density=40,
        pressure_multiplier=1000.0,
        near_pressure_multiplier=50.0,
        bounds_size=(40, 20),
        viscosity_strength=0.01,
        delta_time=0.002,
    )
    num_steps = 15000

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))
    ax_sim = axes[0]
    ax_energy = axes[1]

    scatter = ax_sim.scatter(
        sim.positions[:, 0],
        sim.positions[:, 1],
        s=1,
        c=np.linalg.norm(sim.velocities, axis=1),
        cmap="bwr",
        vmin=0,
        vmax=5,
    )
    node0_scatter = ax_sim.scatter(
        sim.positions[0, 0], sim.positions[0, 1], s=50, c="k", marker="*"
    )

    ax_sim.set_xlim(-20, 20)
    ax_sim.set_ylim(-10, 10)
    ax_sim.set_aspect("equal", adjustable="box")
    ax_sim.set_title("Particle Simulation")
    plt.colorbar(scatter, ax=ax_sim, label="Speed")
    debug_text = ax_sim.text(
        -9.5,
        9.5,
        "",
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    steps_history = []
    ke_history = []
    pe_history = []
    te_history = []
    (ke_line,) = ax_energy.plot([], [], "b-", label="Kinetic (KE)")
    (pe_line,) = ax_energy.plot([], [], "r-", label="Potential (PE)")
    (te_line,) = ax_energy.plot([], [], "g-", label="Total (TE)")
    ax_energy.set_xlabel("Step")
    ax_energy.set_ylabel("Energy")
    ax_energy.set_title("System Energy vs. Step")
    ax_energy.grid(True)
    ax_energy.legend()

    plt.tight_layout()

    for step in tqdm(range(num_steps)):
        ext_force, pressure_force, viscosity_force = sim.step_simulation()
        if step % 3 == 0:
            speeds = np.linalg.norm(sim.velocities, axis=1)
            scatter.set_offsets(sim.positions)
            scatter.set_array(speeds)
            if sim.num_particles > 0:
                node0_scatter.set_offsets(sim.positions[0].reshape(1, 2))
            ax_sim.set_title(f"Step {step}")
            if sim.num_particles > 0:
                debug_str = (
                    f"Node 0 Accel:\n"
                    f" Ext: ({ext_force[0]:.1f}, {ext_force[1]:.1f})\n"
                    f" Pres: ({pressure_force[0]:.1f}, {pressure_force[1]:.1f})\n"
                    f" Visc: ({viscosity_force[0]:.1f}, {viscosity_force[1]:.1f})"
                )
                debug_text.set_text(debug_str)
                global_accel = ext_force + pressure_force + viscosity_force
                node0_pos = sim.positions[0].reshape(1, 2)

            total_ke = sim.calculate_total_kinetic_energy()
            total_pe = sim.calculate_total_potential_energy()
            total_energy = total_ke + total_pe

            steps_history.append(step)
            ke_history.append(total_ke)
            pe_history.append(total_pe)
            te_history.append(total_energy)

            ke_line.set_data(steps_history, ke_history)
            pe_line.set_data(steps_history, pe_history)
            te_line.set_data(steps_history, te_history)

            ax_energy.relim()
            ax_energy.autoscale_view()

            fig.canvas.draw_idle()
            plt.pause(0.001)

    plt.ioff()
    print("Simulation Complete.")
    plt.show()

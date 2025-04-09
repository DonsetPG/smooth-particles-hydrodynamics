# Smooth Particles Hydrodynamics - SPH

This is a optimised python implementation of a Smooth Particles Hydrodynamics solver. 
This was built using numba for fast numpy operations, and a KDTree for efficient neighbours look-up. 

Below is a summary of the theory behind this implementation. This was inspired from the following sources: 

- [Coding Adventure: Simulating Fluids](https://www.youtube.com/watch?v=rSKMYc1CQHE&t=2402s)
- [Particle-Based Fluid Simulation for Interactive Applications](https://matthias-research.github.io/pages/publications/sca03.pdf)
- [Particle-based Viscoelastic Fluid Simulation](http://www.ligum.umontreal.ca/Clavet-2005-PVFS/pvfs.pdf)

# Results

## Regular Gravity

<img width="400" alt="gif" src="assets/classical.gif">

## Moving Gravity 

<img width="400" alt="gif" src="assets/movinggravity.gif">

## Dam Break

<img width="400" alt="gif" src="assets/dambreak.gif">

# SPH Fundamentals

In Smooth Particles Hydrodynamics (SPH), fluid properties are carried by discrete particles. Any field quantity $A$ at a position $\mathbf{r}$ is interpolated by summing contributions from neighboring particles $j$, weighted by a smoothing kernel $W$ with a characteristic smoothing radius $h$:

$$
A_S(\mathbf{r}) = \sum_{j} m_j \frac{A_j}{\rho_j} W(\mathbf{r} - \mathbf{r}_j, h)
$$

where $m_j$ is the mass (assumed constant and equal for all particles in the code), $\rho_j$ is the density, and $A_j$ is the quantity value at particle $j$'s position $\mathbf{r}_j$.

#### Density Calculation

The density $\rho_i$ for particle $i$ is computed by applying the SPH summation rule with $A=\rho$:

$$
\rho_i = \sum_{j} m_j W(\mathbf{r}_i - \mathbf{r}_j, h)
$$

We use the `` `spiky_kernel_pow2` `` as the density kernel.

#### Governing Equations

The motion of each particle $i$ is governed by Newton's second law, adapted from the Navier-Stokes momentum equation. The acceleration $\mathbf{a}_i$ is determined by the sum of forces $\mathbf{f}_i$ acting on the particle, divided by its density:

$$
\mathbf{a}_i = \frac{d\mathbf{v}_i}{dt} = \frac{\mathbf{f}_i}{\rho_i} = \frac{\mathbf{f}_i^{\text{pressure}} + \mathbf{f}_i^{\text{viscosity}} + \mathbf{f}_i^{\text{external}}}{\rho_i}
$$

### Force Computation

#### Pressure Force

Pressure $p$ is calculated from density using a modified ideal gas equation to maintain a target density $\rho_0$:

$$
p_i = k (\rho_i - \rho_0)
$$

where $k$ is a stiffness constant. The pressure force is computed symmetrically to ensure momentum conservation:

```math
\mathbf{f}_i^{\text{pressure}} = -\sum_{j} m_j \frac{p_i + p_j}{2\rho_j} \nabla W_{\text{spiky}}(\mathbf{r}_i - \mathbf{r}_j, h)
```

#### Near Pressure Anti-Clustering

To prevent unrealistic particle clustering, a second pressure term, derived from a *near-density* $\rho^{\text{near}}$, is introduced, following the "double density relaxation" concept. This near-density uses the sharper cubic kernel $W_{\text{spiky}}$, emphasizing very close neighbors:

$$
\rho_{i}^{\text{near}} = \sum_{j} (1 - r_{ij}/h)^3
$$

The corresponding near-pressure $P^{\text{near}}$ is designed to be purely repulsive by having a zero rest density:

$$
p_{i}^{\text{near}} = k^{\text{near}} \rho_{i}^{\text{near}}
$$

where $k^{\text{near}}$ is a separate stiffness parameter. This near-pressure adds to the standard pressure force, typically using the gradient of the same cubic kernel $W_{\text{spiky}}$:

```math
\mathbf{f}_i^{\text{nearpressure}} = -\sum_{j} m_j \frac{p_i^{\text{near}} + p_j^{\text{near}}}{2\rho_j} \nabla W_{\text{spiky}}(\mathbf{r}_i - \mathbf{r}_j, h)
```

The total pressure force acting on particle $i$ is the sum $\mathbf{f}_i^{\text{pressure}} + \mathbf{f}_i^{\text{nearpressure}}$. This approach ensures a more uniform particle distribution and contributes to emergent surface tension effects.

#### Viscosity Force

Viscosity models the internal friction of the fluid. The force is calculated based on velocity differences between particles:

```math
\mathbf{f}_i^{\text{viscosity}} = \mu \sum_{j} m_j \frac{\mathbf{v}_j - \mathbf{v}_i}{\rho_j} \nabla^2 W_{\text{viscosity}}(\mathbf{r}_i - \mathbf{r}_j, h)
```

where $\mu$ is the viscosity coefficient. The final force is divided by $\rho_i$ (implicitly, as it directly adds to velocity scaled by $\Delta t$).

#### External Forces

Gravity is applied as a constant downward acceleration in `` `external_forces` ``: $\mathbf{f}_i^{\text{external}} = \rho_i \mathbf{g}$.

### Smoothing Kernels

We employ several kernels with radius $h$:

* **Poly6 Kernel**: Used for viscosity
    ```math
    W_{\text{poly6}}(r,h) = \frac{315}{64\pi h^9} (h^2 - r^2)^3, \quad 0 \le r \le h
    ```
* **Spiky Kernel (pow3)**: Used for near-density calculation.
    ```math
    W_{\text{spiky}}(r,h) = \frac{15}{\pi h^6} (h - r)^3, \quad 0 \le r \le h
    ```
    Its gradient is used for near-pressure force.
* **Spiky Kernel (pow2)**: Used for standard density calculation. Not explicitly in the paper, but analogous.
    ```math
    W_{\text{spiky\_p2}}(r,h) = \frac{6}{\pi h^4} (h - r)^2, \quad 0 \le r \le h
    ```
    Its gradient is used for standard pressure force.

with $r = ||\mathbf{r}_i - \mathbf{r}_j||$.

### Numerical Integration

Particle positions and velocities are updated over time steps $\Delta t$ using an explicit Euler integration scheme. Finally, collision handling with boundaries is applied after the position update using symmetric collisions with a damping factor.

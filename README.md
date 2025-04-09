This is a optimised python implementation of a Smooth Particles Hydrodynamics solver. 
This was built using numba for fast numpy operations, and a KDTree for efficient neighbours look-up. 

Below is a summary of the theory behind this implementation. This was inspired from the following sources: 

- [Coding Adventure: Simulating Fluids](https://www.youtube.com/watch?v=rSKMYc1CQHE&t=2402s)
- [Particle-Based Fluid Simulation for Interactive Applications](https://matthias-research.github.io/pages/publications/sca03.pdf)
- [Particle-based Viscoelastic Fluid Simulation](http://www.ligum.umontreal.ca/Clavet-2005-PVFS/pvfs.pdf)
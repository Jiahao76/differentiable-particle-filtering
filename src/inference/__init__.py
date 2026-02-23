"""
Bayesian inference algorithms for parameter estimation.

Methods:
    - HMC: Hamiltonian Monte Carlo (Neal 2011)
    - PMMH: Particle Marginal Metropolis-Hastings (Andrieu et al. 2010)
    - ParticleGibbs: Particle Gibbs sampler for SSL models (Zheng 2017)
"""

from .hmc import HMC
from .pmmh import PMMH, random_walk_proposal
from .particle_gibbs import ParticleGibbs
from .utils import compute_ess

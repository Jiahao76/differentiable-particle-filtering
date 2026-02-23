"""
Filtering algorithms for state-space models.

Classical filters:
    - KalmanFilter: Linear Gaussian optimal filter
    - ExtendedKalmanFilter: Linearization-based nonlinear filter
    - UnscentedKalmanFilter: Sigma-point nonlinear filter

Particle filters:
    - ParticleFilter: Standard SIR particle filter
    - DifferentiableParticleFilter: PF with soft/OT/Gumbel resampling

Particle flow filters (Li & Coates 2017):
    - EDHFlowFilter: Exact Daum-Huang flow
    - LEDHFlowFilter: Local EDH flow
    - PFPF_EDH / PFPF_LEDH: Particle flow particle filters
    - PFPF_LEDH_Enhanced / PFPF_EDH_Enhanced: PF-PF with optimal homotopy (Dai 2022)
    - InvertibleFlowParticleFilter: Invertible flow PF

Stochastic particle flow (Dai 2022):
    - StochasticParticleFlowFilter: Stochastic flow with homotopy
    - HomotopyOptimizer / RobustHomotopyOptimizer: TPBVP solvers

Differentiable flow:
    - DifferentiablePFPF: LEDH flow + OT resampling for HMC

Neural OT resampling (Bonus 2):
    - OTResamplingNetwork: mGradNet for OT resampling
    - FourierOTOperator: FNO for OT resampling
    - DeepONetOT: DeepONet for OT resampling
    - neural_ot_resample: Convenience function
"""

# Classical filters
from .kalman_filter import KalmanFilter
from .ekf import ExtendedKalmanFilter
from .ukf import UnscentedKalmanFilter

# Standard particle filter
from .particle_filter import StandardParticleFilter as ParticleFilter

# Differentiable particle filter (Corenflos 2021)
from .differentiable_particle_filter import DifferentiableParticleFilter

# Deterministic particle flow (Li & Coates 2017)
from .edh_flow import EDHFlowFilter
from .ledh_flow import LEDHFlowFilter
from .pfpf_edh import PFPF_EDH
from .pfpf_ledh import PFPF_LEDH
from .pfpf_enhanced import PFPF_LEDH_Enhanced, PFPF_EDH_Enhanced
from .flow_pf import InvertibleFlowParticleFilter

# Stochastic particle flow (Dai 2022)
from .particle_flow_filters import TFStochasticParticleFlowFilter as StochasticParticleFlowFilter
from .homotopy_optimizer import HomotopyOptimizer
from .homotopy_optimizer_robust import RobustHomotopyOptimizer

# Neural OT resampling (Bonus 2: Chaudhari 2025, Jha 2025)
from .neural_ot_resampling import (
    OTResamplingNetwork,
    FourierOTOperator,
    DeepONetOT,
    neural_ot_resample,
    compute_statistics
)

# Differentiable flow (Bonus 1)
from .differentiable_pfpf import DifferentiablePFPF

# Gradient quality diagnostics
from .gradient_diagnostics import (
    estimate_gradient_variance,
    gradient_agreement,
    estimate_gradient_snr,
    estimate_condition_number,
    estimate_lipschitz,
)

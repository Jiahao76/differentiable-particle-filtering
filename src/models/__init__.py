"""
State-space models for filtering experiments.

Models:
    - LinearGaussianSSM: Standard LGSSM for Kalman filter benchmarks
    - StochasticVolatilityModel: SV model (Doucet 2009, Example 4)
    - NonlinearSSM: Highly nonlinear SSM (Andrieu et al. 2010)
    - BearingOnlyTrackingModel: 2D bearing-only tracking (Dai 2022, Section 4)
    - StateSpaceLSTM: Neural state-space model (Zheng 2017)
"""

from .base_model import StateSpaceModel
from .lgssm import LinearGaussianSSM
from .sv_model import StochasticVolatilityModel
from .nonlinear_ssm import NonlinearSSM
from .bearing_only_tracking import BearingOnlyTrackingModel
from .state_space_lstm import StateSpaceLSTM

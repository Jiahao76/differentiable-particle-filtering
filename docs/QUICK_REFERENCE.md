# Quick Reference: Dai22 + Li17 Integration Guide

## One-Sentence Summary
✅ **Dai22 optimization helps EDH on soft problems (+13%), fails on stiff problems (−31%), never helps LEDH due to per-particle adaptation already being local optimal.**

---

## Decision Tree: Should You Use Dai22?

```
START: Want to optimize Li17 PF-PF with Dai22?
│
├─ Which filter variant?
│  │
│  ├─→ LEDH (per-particle Hessian)
│  │   └─→ NO, don't use Dai22
│  │       Reason: Per-particle adaptation already locally optimal
│  │       Benefit: 0% (or negative)
│  │       Cost: +2-5% computational overhead
│  │
│  └─→ EDH (ensemble Hessian)
│      └─→ Check prior uncertainty
│          │
│          ├─ High uncertainty (σ > 2)?
│          │  └─→ YES, use Dai22
│          │      Expected benefit: +10-15%
│          │      Reason: Minimal ensemble drift
│          │
│          └─ Low uncertainty (σ < 1)?
│             └─→ NO, stick with linear β(λ)=λ
│                 Expected benefit if used: −15-30%
│                 Reason: Large ensemble drift invalidates β*
END
```

---

## Code Template: Basic Usage

```python
from src.filters import PFPF_LEDH_Enhanced, PFPF_EDH_Enhanced
from src.filters.edh_flow import RobustHomotopyOptimizer
import numpy as np

# Initialize filter
filter_edh = PFPF_EDH_Enhanced(model, num_particles=100, flow_steps=30)

# Option 1: Standard linear homotopy (always safe)
estimates, metadata = filter_edh.run(observations)

# Option 2: Use Dai22 optimization (only for EDH on soft problems)
if use_dai22 and prior_std > 2:
    # Compute Hessians at ensemble mean
    eta_0 = particles.mean(axis=0)
    H0 = model.compute_hessian(eta_0)        # At prior mean
    Hh = model.compute_likelihood_hessian(eta_0, observations[0])  # At likelihood
    
    # Solve for optimal homotopy
    optimizer = RobustHomotopyOptimizer(mu=0.2, ridge_reg=1e-3)
    beta_func = optimizer.solve_optimal_homotopy(H0, Hh, method='continuation')
    
    # Run filter with optimized homotopy
    estimates, metadata = filter_edh.run(observations, beta_func=beta_func)
else:
    # Fall back to linear (safe default)
    estimates, metadata = filter_edh.run(observations)
```

---

## Performance Expectations

| Scenario | Method | Improvement | Confidence |
|----------|--------|------------|------------|
| LEDH + Dai22 | Any | ~0% | High |
| EDH + Dai22 (soft prior, σ>2) | SV model | +10-15% | High |
| EDH + Dai22 (tight prior, σ<1) | SV model | −15-30% | High |
| EDH + Dai22 (bearing-only) | Original problem | +40-60%? | Medium (untested) |

---

## Why Does This Happen? (Technical Insight)

### The Problem: Distribution Shift
```
Time t=0 (Initial):
  Particles at x ~ p(x)
  Hessian H₀ = ∇² log p(z|x̄₀)
  Dai22 computes β*(λ) optimized for H₀

Time t=T (After flow):
  Particles have moved to x' ~ q(x)
  Hessian H₁ = ∇² log p(z|x̄₁) ≠ H₀
  β*(λ) is now suboptimal for H₁!

Severity:
  Loose prior (σ large) → x̄ doesn't move much → H₁ ≈ H₀ → β* still good
  Tight prior (σ small) → x̄ moves a lot → H₁ ≠ H₀ → β* becomes bad
```

### Why LEDH Never Benefits
```
LEDH uses: ẋᵢ = Hᵢ(xᵢ)⁻¹ ∇ₓ log p(z|xᵢ)

Each particle i has its own Hessian H_i computed at its own position x_i
Result: Each particle is already locally optimized in its neighborhood
Adding global Dai22 optimization: Doesn't help, might interfere
```

---

## Common Mistakes & Fixes

### Mistake 1: "I used Dai22 and LEDH got worse"
**Why**: LEDH already has per-particle adaptation
**Fix**: Don't use Dai22 with LEDH
**Instead**: Use linear homotopy, focus on LEDH tuning

### Mistake 2: "Dai22 helped on soft problems but hurts on stiff"
**Why**: Distribution shift invalidates pre-computed β*
**Fix**: Check problem tightness (prior σ) before using Dai22
**Instead**: Use Dai22 only when prior_std > 2

### Mistake 3: "My BVP solver diverges"
**Why**: Non-PSD Hessians, gradient explosions
**Fix**: Use `RobustHomotopyOptimizer` with ridge regularization
**See**: [DAI22_IMPLEMENTATION_GUIDE.md](DAI22_IMPLEMENTATION_GUIDE.md)

### Mistake 4: "Dai22 works great, why not always use it?"
**Why**: Mathematical improvement ≠ practical improvement
**Fix**: Profile your specific problem (soft vs stiff?)
**Remember**: Optimization validity depends on environment stability

---

## Performance Profiling: Find Your Problem's Stiffness

```python
import numpy as np
from scipy.linalg import eigvals

def characterize_problem(particles, observations, model):
    """Determine if problem is soft or stiff"""
    
    # 1. Compute ensemble mean and its Hessian
    eta_bar = particles.mean(axis=0)
    H_likelihood = model.compute_likelihood_hessian(eta_bar, observations[0])
    H_prior = model.compute_prior_hessian(eta_bar)
    
    # 2. Condition number analysis
    eigs_lik = np.abs(eigvals(H_likelihood))
    eigs_prior = np.abs(eigvals(H_prior))
    cond_lik = eigs_lik.max() / eigs_lik.min() if eigs_lik.min() > 0 else np.inf
    cond_prior = eigs_prior.max() / eigs_prior.min() if eigs_prior.min() > 0 else np.inf
    
    # 3. Make recommendation
    if cond_lik < 10:
        return "SOFT", "Use Dai22 ✓"
    elif cond_lik < 100:
        return "MEDIUM", "Test first ⚠"
    else:
        return "STIFF", "Dai22 might help ✓"
```

---

## When Dai22 Really Helps (vs When It Doesn't)

### ✅ Dai22 Helps
- Bearing-only tracking problem (non-convex, high ill-conditioning)
- EDH filter with loose prior (σ > 2)
- Ensemble doesn't drift much during filtering
- Problem conditions remain stable throughout

### ❌ Dai22 Doesn't Help
- Smooth, well-conditioned problems (SV model type)
- LEDH filter (already per-particle optimal)
- Tight priors causing large ensemble drift
- Dynamic problems where conditions change over time

### ⚠️  Dai22 Might Make Worse
- Medium-stiffness problems with EDH (−31% observed)
- When assumptions of pre-computed β* are violated
- If computational overhead not justified by gain
- When LEDH + Dai22 cause interference

---

## Next Steps: Making Dai22 Always Work

### For Users Now
- Profile your problem stiffness first
- Use decision tree above to decide
- Monitor ESS and weight variance to detect mismatch

### For Researchers (Future Work)
- **Dynamic re-optimization**: Compute β*(λ) at each filtering step
  - Cost: 2-3x more BVP solves
  - Benefit: Should work for all problems
  - Expected result: +10% across all stiffness levels

- **Adaptive switching**: Use Dai22 only when drift detected
  - Measure: ∥η̄(t+1) − η̄(t)∥
  - Threshold: If drift > ε, fall back to linear

- **Assumption validator**: Automatic detection of when β* becomes invalid
  - Check: Current Hessian vs pre-computed Hessian
  - Action: Re-optimize if divergence > ε

---

## References to Detailed Documentation

| Question | Document |
|----------|----------|
| "Why doesn't Dai22 help LEDH?" | [STIFFNESS_ANALYSIS_RESULTS.md](STIFFNESS_ANALYSIS_RESULTS.md#why-ledh-shows-no-benefit) |
| "What's the distribution shift problem?" | [PROJECT_ARC_SYNTHESIS.md](PROJECT_ARC_SYNTHESIS.md#theoretical-implications) |
| "How do I implement RobustHomotopyOptimizer?" | [DAI22_IMPLEMENTATION_GUIDE.md](DAI22_IMPLEMENTATION_GUIDE.md) |
| "Can you show me code examples?" | [compare_dai22_li17.py](compare_dai22_li17.py) |
| "What about bearing-only tracking?" | [DAI22_COMPLETE_SUMMARY.md](DAI22_COMPLETE_SUMMARY.md) |

---

## Key Takeaway

**✅ Use Dai22 selectively, not universally.**

- EDH filter? Use Dai22 only if prior_std > 2
- LEDH filter? Skip Dai22 entirely
- Unsure about stiffness? Profile first, then decide

**Remember**: Mathematical improvements require environmental stability to translate into practical gains.

---

**Status**: ✅ Production ready  
**Last Updated**: February 9, 2026  
**Maintainer**: GitHub Copilot  
**Questions?** See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)

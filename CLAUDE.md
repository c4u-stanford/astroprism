# astroprism

Probabilistic forward modeling framework for multi-wavelength astronomical imaging.
Built on NIFTy8.re (JAX) for Bayesian inference with full posterior uncertainty.

## Package structure

```
src/astroprism/
  models/
    gp.py           # SpatialGP, MixtureGP — signal priors
    forward.py      # ForwardModel — combines GP + response + noise
    response.py     # InstrumentResponse — PSF convolution + reprojection
    noise.py        # NoiseModel
    likelihood.py   # LikelihoodModel, build_likelihood()
  operators/
    convolution.py  # Convolver
    reprojection.py # Reprojector
  inference/
    vi.py           # VariationalInference (supports scheduling)
  io/
    dataset.py      # BaseDataset, SingleInstrumentDataset, MultiInstrumentDataset
    instrument_loaders.py
    preprocessing.py
```

## NIFTy8.re essentials

```python
import nifty8.re as jft
```

### jft.Model

Base class for all models. Subclass and implement `__call__`:

```python
class MyModel(jft.Model):
    def __init__(self, submodel):
        self.submodel = submodel
        domain = submodel.domain
        init = submodel.init
        super().__init__(domain=domain, init=init)

    def __call__(self, x: dict):
        return self.submodel(x)
```

- `domain`: tree of `ShapeWithDtype` describing input parameter space
- `init`: callable `(key) -> dict` returning initial parameter values
- Composed models merge `domain` and `init` with `|`
- All operations must be JAX-differentiable (no Python control flow on traced values)
- Mark sub-models as dynamic fields to avoid JIT recompilation issues:
  ```python
  submodel: Any = dataclasses.field(metadata=dict(static=False))
  ```

### jft.optimize_kl

The VI engine. Key parameters:

```python
samples, state = jft.optimize_kl(
    likelihood,                          # jft.Model (negative log-likelihood)
    jft.Vector(init_params),             # initial position
    n_total_iterations=20,
    n_samples=5,
    key=jax_prng_key,
    odir="output/run_001",               # checkpoint directory
    resume=True,                         # resume from checkpoint
    sample_mode="linear_resample",       # see modes below
    constants=lambda i: ("bg",) if i < 5 else (),       # frozen params
    point_estimates=lambda i: ("bg",) if i < 10 else (), # MAP params
    draw_linear_kwargs=...,
    nonlinearly_update_kwargs=...,
    kl_kwargs=...,
)
```

**sample_mode options** (cheapest → most accurate):
- `"linear_resample"` — fast, Gaussian approximation, good for early iters
- `"nonlinear_resample"` — full GeoVI, best quality (default in NIFTy)
- `"nonlinear_update"` — continuous updates, no resampling

**constants vs point_estimates:**
- `constants`: param is completely frozen — not touched by the optimiser
- `point_estimates`: param is MAP-estimated — optimised but not sampled over
- Both: `callable(iteration) -> tuple[str]` of param names
- Returning `()` means no params frozen/MAP for that iteration

**Solver kwargs structure:**
```python
draw_linear_kwargs = dict(
    cg_name="linear_solver",
    cg_kwargs=dict(absdelta=1e-3, maxiter=1000)
)
nonlinearly_update_kwargs = dict(
    minimize_kwargs=dict(name="nonlinear_solver", xtol=1e-3,
                         cg_kwargs=dict(name=None), maxiter=5)
)
kl_kwargs = dict(
    minimize=jft.optimize._newton_cg,
    minimize_kwargs=dict(name="kl_minimizer", xtol=1e-8,
                         absdelta=1e-2, cg_kwargs=dict(name=None), maxiter=100)
)
```

### Likelihoods

```python
# Fixed noise
jft.Gaussian(data, noise_std_inv=1.0/sigma)

# Variable noise — model must return (mean, inv_std) tuple
jft.VariableCovarianceGaussian(data)

# Compose likelihood with forward model
likelihood = jft.VariableCovarianceGaussian(data).amend(forward_model)
```

### GP / Correlated field priors

```python
cfm = jft.CorrelatedFieldMaker("signal")
cfm.set_amplitude_total_offset(mean, std)   # zero mode / mean level
cfm.add_fluctuations(
    shape, distances,
    fluctuations=jft.LogNormalPrior(0, 1),
    loglog_slope=jft.NormalPrior(-3, 1),
    flexibility=jft.LogNormalPrior(0.5, 0.1),
)
signal = cfm.finalize()
```

Available priors: `jft.NormalPrior`, `jft.LogNormalPrior`, `jft.UniformPrior`,
`jft.LaplacePrior`, `jft.InvGammaPrior`

### Accessing results

```python
posterior_mean = samples.pos       # expansion point (mean)
posterior_samples = samples.samples  # list of sample dicts
```

## Important gotchas

- **`jax_enable_x64 = True` must be set before any JAX/NIFTy import** — put it at the top of every script/notebook
- **Init params must be float64** — `{k: jnp.array(v, dtype=jnp.float64) for k, v in init.items()}`
- `jft.optimize_kl` takes `n_total_iterations` not `n_iterations` — cumulative, so resuming continues from where it left off toward the same total
- Use `jit=False` when debugging — errors are much clearer
- `mirror_samples=True` (default) doubles effective sample count cheaply

## Forward model pattern

```
GP params (x)
    ↓ gp_model(x)
signal s                        ← sky flux map
    ↓ response_model(x, s)      ← reprojection + PSF convolution
mean_data r
    ↓ noise_model(x, r)
(mean_data, inv_std)            ← input to VariableCovarianceGaussian
```

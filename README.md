# SHERPA

**S**pectroscopic **H**igh-resolution **E**mission & **R**etrieval **P**ipeline for **A**tmospheres.

Atmospheric retrievals of high-resolution spectra with [petitRADTRANS](https://petitradtrans.readthedocs.io/) v3, [pyfastchem](https://github.com/exoclime/FastChem), and [MultiNest](https://johannesbuchner.github.io/MultiNest/).

Requires Python ≥ 3.10.

## Install

```bash
pip install -e .
```

## Quick start

1. Set paths to external data (not bundled with SHERPA):

```bash
export PRT_INPUT_DATA_PATH=/path/to/petitRADTRANS/input_data
export FASTCHEM_INPUT_PATH=/path/to/FastChem/input
```

2. Place reduced iSHELL FITS data in `examples/data/gj3820/`.

3. Build the forward model and run a prior check:

```bash
python examples/gj3820_model.py
```

This creates the Radtrans object, saves `retrievals/<run>/radtrans.pkl`, and writes prior-check plots.

4. Run the retrieval (optionally with MPI):

```bash
python examples/gj3820_retrieval.py
mpiexec -np 8 python examples/gj3820_retrieval.py
```

Outputs are written to `retrievals/`. After the retrieval finishes, final diagnostic plots with the `_final` suffix are written automatically.

## Package

| Module | Purpose |
|--------|---------|
| `sherpa.data` | Load and prepare iSHELL M-band spectra (`IshellMband`) |
| `sherpa.pressure_temperature` | RCE pressure–temperature profiles |
| `sherpa.chemistry` | Equilibrium chemistry via `FastChemistry` |
| `sherpa.retrieval` | Forward model and MultiNest driver (`Retrieval`) |

A retrieval is configured with spectral data, `radtrans_kwargs`, `PT_kwargs`, `chem_kwargs`, free and constant parameters, then run with `Retrieval.run()`. See `examples/gj3820_model.py` and `examples/gj3820_retrieval.py` for a full setup.

Species names and petitRADTRANS linelists are mapped in `sherpa/data/chemistry/species_info.txt`. Chemistry returns volume mixing ratios; convert to mass fractions with `FastChemistry.vmrs_to_mass_fractions(vmrs)`.

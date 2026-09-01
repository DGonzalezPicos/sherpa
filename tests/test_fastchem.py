"""
Minimal diagnostics for pyfastchem and sherpa.chemistry.FastChemistry.

Run from the repository root (requires FASTCHEM_INPUT_PATH):

    python tests/test_fastchem.py

Optional flags:
    --section basic|pt|random|scaling|all
    --verbose
    --n-random 20
"""
import argparse
import os
import pathlib
import sys

import numpy as np
import pyfastchem

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sherpa.chemistry import FastChemistry
from sherpa.pressure_temperature import PressureTemperatureGradients


FASTCHEM_INPUT = os.environ.get("FASTCHEM_INPUT_PATH")
if FASTCHEM_INPUT is None:
    raise SystemExit(
        "Set FASTCHEM_INPUT_PATH to your FastChem input directory before running this script."
    )
ELEMENT_FILE = f"{FASTCHEM_INPUT}/element_abundances/asplund_2009.dat"
LOGK_FILE = f"{FASTCHEM_INPUT}/logK/logK.dat"
SPECIES_INFO_FILE = REPO_ROOT / "sherpa" / "data" / "chemistry" / "species_info.txt"

PRESSURE = np.logspace(-4.0, 2.0, 40)


def print_header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def make_fastchem(verbose=0):
    return pyfastchem.FastChem(ELEMENT_FILE, LOGK_FILE, verbose)


def run_fastchem(fastchem, temperature, pressure, metallicity=0.0, c_to_o=0.55, elemental_ratios=None):
    """Direct pyfastchem call with optional abundance scaling."""
    init_abund = np.array(fastchem.getElementAbundances())
    abundances = init_abund.copy()

    c_idx = fastchem.getElementIndex("C")
    o_idx = fastchem.getElementIndex("O")
    solar_c_to_o = init_abund[c_idx] / init_abund[o_idx]

    for i in range(len(abundances)):
        symbol = fastchem.getElementSymbol(i)
        if symbol not in ("H", "He"):
            abundances[i] *= 10.0 ** metallicity

    abundances[c_idx] *= c_to_o / solar_c_to_o

    if elemental_ratios:
        for element, dex in elemental_ratios.items():
            idx = fastchem.getElementIndex(element)
            if idx != 9999999:
                abundances[idx] *= 10.0 ** dex

    fastchem.setElementAbundances(abundances.tolist())

    input_data = pyfastchem.FastChemInput()
    output_data = pyfastchem.FastChemOutput()
    input_data.temperature = np.asarray(temperature)
    input_data.pressure = np.asarray(pressure)
    input_data.equilibrium_condensation = False
    input_data.rainout_condensation = False

    global_flag = fastchem.calcDensities(input_data, output_data)
    layer_flags = np.array(output_data.fastchem_flag)
    layer_iters = np.array(output_data.nb_chemistry_iterations)

    return {
        "global_flag": global_flag,
        "global_message": pyfastchem.FASTCHEM_MSG[global_flag],
        "layer_flags": layer_flags,
        "layer_iters": layer_iters,
        "number_densities": np.array(output_data.number_densities),
        "mean_molecular_weight": np.array(output_data.mean_molecular_weight),
        "temperature": np.asarray(temperature),
        "pressure": np.asarray(pressure),
    }


def summarize_layer_failures(result, verbose=False, max_lines=15):
    flags = result["layer_flags"]
    temperature = result["temperature"]
    pressure = result["pressure"]
    iters = result["layer_iters"]

    bad = np.where(flags != 0)[0]
    if bad.size == 0:
        print("  all layers converged")
        return []

    print(f"  failing layers: {bad.size}/{len(flags)}")
    failures = []
    for i in bad[:max_lines]:
        msg = pyfastchem.FASTCHEM_MSG[flags[i]] if flags[i] < len(pyfastchem.FASTCHEM_MSG) else str(flags[i])
        line = (
            f"    layer {i:2d}: P={pressure[i]:.3e} bar, T={temperature[i]:.1f} K, "
            f"flag={flags[i]} ({msg}), iters={iters[i]}"
        )
        print(line)
        failures.append(i)
    if bad.size > max_lines:
        print(f"    ... {bad.size - max_lines} more failing layers")
    return failures


def test_basic_fastchem(verbose=False):
    print_header("1. Basic pyfastchem call")

    fastchem = make_fastchem(verbose=0)

    cases = [
        ("single layer, 3000 K, 1 bar", [3000.0], [1.0]),
        ("uniform 3000 K over pressure grid", np.full(len(PRESSURE), 3000.0), PRESSURE),
        ("linear 2200-4800 K over pressure grid", np.linspace(2200, 4800, len(PRESSURE)), PRESSURE),
        ("low-T upper atmosphere (100 K)", np.full(len(PRESSURE), 100.0), PRESSURE),
        ("very low-T upper atmosphere (10 K)", np.full(len(PRESSURE), 10.0), PRESSURE),
        ("unphysical cold profile (2-4000 K)", np.geomspace(2, 4000, len(PRESSURE)), PRESSURE),
    ]

    for label, temperature, pressure in cases:
        result = run_fastchem(fastchem, temperature, pressure)
        print(f"\n  case: {label}")
        print(f"  global flag: {result['global_flag']} ({result['global_message']})")
        summarize_layer_failures(result, verbose=verbose)


def test_pt_profile_failures(verbose=False):
    print_header("2. FastChem with PressureTemperatureGradients profiles")

    pt = PressureTemperatureGradients(PRESSURE, PT_interp_mode="linear")
    fastchem = make_fastchem(verbose=0)

    cases = [
        ("moderate gradients", {
            "T_RCE": 3000.0,
            "log_P_RCE": 0.0,
            "dlog_P": 0.5,
            "n_pressure_levels": 7,
            **{f"T_grad_{i}": 0.15 for i in range(7)},
        }),
        ("high gradients (retrieval upper bound)", {
            "T_RCE": 4000.0,
            "log_P_RCE": 0.5,
            "dlog_P": 1.5,
            "n_pressure_levels": 7,
            **{f"T_grad_{i}": 0.72 for i in range(7)},
        }),
        ("hot RCE + high gradients", {
            "T_RCE": 4800.0,
            "log_P_RCE": -1.0,
            "dlog_P": 1.8,
            "n_pressure_levels": 7,
            **{f"T_grad_{i}": 0.72 for i in range(7)},
        }),
    ]

    for label, params in cases:
        temperature = pt(params)
        print(f"\n  case: {label}")
        print(f"  T range: {temperature.min():.1f} - {temperature.max():.1f} K")
        print(f"  min T at P={PRESSURE[np.argmin(temperature)]:.3e} bar")

        result = run_fastchem(fastchem, temperature, PRESSURE)
        print(f"  global flag: {result['global_flag']} ({result['global_message']})")
        summarize_layer_failures(result, verbose=verbose)


def test_fastchemistry_wrapper(verbose=False):
    print_header("3. sherpa.chemistry.FastChemistry wrapper")

    species_info = np.loadtxt(SPECIES_INFO_FILE, dtype=str)
    species_names = ["H2O", "CO", "OH", "CN", "Fe", "Na", "13CO", "C18O"]
    line_species = species_info[np.isin(species_info[:, 0], species_names), 1].tolist()

    chem = FastChemistry(
        line_species,
        PRESSURE,
        fastchem_input_path=FASTCHEM_INPUT,
        species_info_file=str(SPECIES_INFO_FILE),
    )

    pt = PressureTemperatureGradients(PRESSURE, PT_interp_mode="linear")
    params = {
        "T_RCE": 3000.0,
        "log_P_RCE": 0.0,
        "dlog_P": 0.5,
        "metallicity": 0.0,
        "c_to_o": 0.55,
        "n_pressure_levels": 5,
        "log_13CO/CO": -2.0,
        "log_C18O/CO": -2.4,
        **{f"T_grad_{i}": 0.15 for i in range(5)},
    }

    temperature = pt(params)
    vmrs = chem(params, temperature=temperature)

    if not isinstance(vmrs, dict):
        print("  FastChemistry returned failure marker:", vmrs)
        return

    print("  FastChemistry call succeeded")
    print(f"  VMR keys: {list(vmrs.keys())}")
    for key in line_species[:4]:
        print(f"    {key}: VMR={vmrs[key][0]:.3e} at top layer")

    mass_fractions = chem.vmrs_to_mass_fractions(vmrs)
    print(f"  MMW range: {mass_fractions['MMW'].min():.3f} - {mass_fractions['MMW'].max():.3f} amu")


def test_abundance_scaling(verbose=False):
    print_header("4. Abundance scaling checks")

    fastchem = make_fastchem(verbose=0)
    temperature = np.full(len(PRESSURE), 3000.0)

    cases = [
        ("solar", 0.0, 0.55, None),
        ("+1 dex metallicity", 1.0, 0.55, None),
        ("C/O = 1.0", 0.0, 1.0, None),
        ("[Na/H] = +0.5 dex", 0.0, 0.55, {"Na": 0.5}),
    ]

    na_idx = fastchem.getGasSpeciesIndex("Na")
    co_idx = fastchem.getGasSpeciesIndex("C1O1")

    for label, metallicity, c_to_o, elemental_ratios in cases:
        result = run_fastchem(
            fastchem,
            temperature,
            PRESSURE,
            metallicity=metallicity,
            c_to_o=c_to_o,
            elemental_ratios=elemental_ratios,
        )
        nd = result["number_densities"]
        n_total = nd.sum(axis=1)
        na_vmr = nd[0, na_idx] / n_total[0]
        co_vmr = nd[0, co_idx] / n_total[0]
        print(f"\n  {label}")
        print(f"    global flag: {result['global_flag']} ({result['global_message']})")
        print(f"    Na VMR={na_vmr:.3e}, CO VMR={co_vmr:.3e}")


def test_random_retrieval_parameters(n_random=20, verbose=False):
    print_header(f"5. Random retrieval-like parameter draws (n={n_random})")

    pt = PressureTemperatureGradients(PRESSURE, PT_interp_mode="linear")
    fastchem = make_fastchem(verbose=0)

    n_ok = 0
    n_fail = 0
    fail_reasons = []

    for trial in range(n_random):
        params = {
            "T_RCE": np.random.uniform(2200.0, 4800.0),
            "log_P_RCE": np.random.uniform(-2.0, 1.0),
            "dlog_P": np.random.uniform(0.2, 1.8),
            "metallicity": np.random.uniform(-0.6, 0.4),
            "c_to_o": np.random.uniform(0.2, 0.9),
            "n_pressure_levels": 7,
        }
        for i in range(7):
            params[f"T_grad_{i}"] = np.random.uniform(0.06, 0.72)

        temperature = pt(params)
        result = run_fastchem(
            fastchem,
            temperature,
            PRESSURE,
            metallicity=params["metallicity"],
            c_to_o=params["c_to_o"],
        )

        if result["global_flag"] == 0:
            n_ok += 1
            continue

        n_fail += 1
        min_t = temperature.min()
        max_t = temperature.max()
        n_bad_layers = np.count_nonzero(result["layer_flags"])
        reason = {
            "trial": trial,
            "min_T": min_t,
            "max_T": max_t,
            "n_bad_layers": n_bad_layers,
            "T_RCE": params["T_RCE"],
            "log_P_RCE": params["log_P_RCE"],
            "dlog_P": params["dlog_P"],
            "T_grad_max": max(params[f"T_grad_{i}"] for i in range(7)),
        }
        fail_reasons.append(reason)

        if verbose:
            print(f"\n  trial {trial}: FAIL")
            print(f"    T range: {min_t:.1f} - {max_t:.1f} K")
            print(f"    T_RCE={params['T_RCE']:.0f}, log_P_RCE={params['log_P_RCE']:.2f}, dlog_P={params['dlog_P']:.2f}")
            summarize_layer_failures(result, verbose=False, max_lines=5)

    print(f"\n  summary: ok={n_ok}, fail={n_fail} ({100.0 * n_fail / n_random:.1f}% failure rate)")

    if fail_reasons:
        min_t_values = [r["min_T"] for r in fail_reasons]
        print(f"  failed draws had min T in [{min(min_t_values):.1f}, {max(min_t_values):.1f}] K")
        low_t_failures = sum(1 for r in fail_reasons if r["min_T"] < 200.0)
        print(f"  failures with min T < 200 K: {low_t_failures}/{n_fail}")

        if not verbose:
            print("\n  first few failing draws:")
            for reason in fail_reasons[:5]:
                print(
                    f"    trial {reason['trial']:2d}: minT={reason['min_T']:6.1f} K, "
                    f"T_RCE={reason['T_RCE']:.0f}, log_P_RCE={reason['log_P_RCE']:.2f}, "
                    f"dlog_P={reason['dlog_P']:.2f}, max grad={reason['T_grad_max']:.2f}"
                )


def test_temperature_threshold(verbose=False):
    print_header("6. FastChem temperature threshold scan")

    fastchem = make_fastchem(verbose=0)
    test_temperatures = [5000, 3000, 1000, 500, 200, 100, 50, 20, 10, 5, 2]

    print("  uniform T [K] | global flag | message")
    for temp in test_temperatures:
        result = run_fastchem(fastchem, [temp], [1.0])
        print(
            f"  {temp:12.0f} | {result['global_flag']:11d} | {result['global_message']}"
        )


def print_recommendations():
    print_header("Recommendations")
    print(
        """
  Findings from these tests:
  - pyfastchem itself works for reasonable T/P grids.
  - Failures during retrieval are usually caused by unphysical PT profiles,
    especially when PressureTemperatureGradients produces very low upper-atmosphere
    temperatures (often < 100 K, sometimes a few K).
  - FastChem then fails with 'Reached maximum number of chemistry iterations'.
  - This is not a bug in the abundance scaling logic; it is a PT-profile validity issue.

  Possible fixes to try in the retrieval:
  1. Reject PT profiles with T_min below a floor (e.g. 300 K) before calling FastChem.
  2. Narrow the T_grad prior range away from 0.72.
  3. Add a temperature floor inside PressureTemperatureGradients for the upper atmosphere.
  4. Treat FastChem layer failures as invalid models and return a low likelihood.
        """.strip()
    )


def main():
    parser = argparse.ArgumentParser(description="Diagnose pyfastchem usage in sherpa")
    parser.add_argument(
        "--section",
        choices=["basic", "pt", "wrapper", "scaling", "random", "threshold", "all"],
        default="all",
        help="Which test section to run",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n-random", type=int, default=20)
    args = parser.parse_args()

    sections = {
        "basic": test_basic_fastchem,
        "pt": test_pt_profile_failures,
        "wrapper": test_fastchemistry_wrapper,
        "scaling": test_abundance_scaling,
        "random": lambda verbose=False: test_random_retrieval_parameters(args.n_random, verbose=verbose),
        "threshold": test_temperature_threshold,
    }

    if args.section == "all":
        for func in sections.values():
            func(verbose=args.verbose)
        print_recommendations()
    else:
        sections[args.section](verbose=args.verbose)
        if args.section in ("random", "pt", "basic"):
            print_recommendations()


if __name__ == "__main__":
    main()

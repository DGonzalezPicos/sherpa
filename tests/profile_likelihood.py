"""
Profile the likelihood evaluation pipeline for sherpa retrievals.

Mirrors the production call path in ``Retrieval.log_likelihood``:
  PT → FastChem → mass fractions → one ``calculate_flux`` per order
  → spline marginalization → chi-squared

Each section is timed once per evaluation; ``calculate_flux`` is not
called redundantly in separate benchmarks.

Run from the repository root (requires PRT_INPUT_DATA_PATH and FASTCHEM_INPUT_PATH):

    python tests/profile_likelihood.py

Optional flags:
    --n-repeat 5          Number of timed repetitions (after warmup)
    --warmup 1            Warmup iterations before timing
    --order 0             Echelle order index for detailed get_flux breakdown
    --cprofile            Run cProfile on a full log_likelihood call
    --build-radtrans      Build Radtrans objects instead of loading the pickle
"""
import argparse
import cProfile
import io
import pathlib
import pstats
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "examples"))

from petitRADTRANS import physical_constants as cst

from gj3820_config import build_gj3820_setup
from sherpa.retrieval import Retrieval, compute_chi2, load_radtrans


def print_header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def prior_center_cube(free_params):
    return np.full(len(free_params), 0.5, dtype=float)


def set_params_from_cube(retrieval, cube):
    retrieval.prior_transform(cube.copy())
    return retrieval.prior_transform.params.copy()


def build_retrieval(load_pickle=True):
    setup = build_gj3820_setup()
    atmospheres = None
    if load_pickle:
        pickle_path = setup["radtrans_pickle"]
        if not pickle_path.is_file():
            raise FileNotFoundError(
                f"Radtrans pickle not found: {pickle_path}\n"
                "Run examples/gj3820_model.py first, or pass --build-radtrans."
            )
        atmospheres = load_radtrans(pickle_path)
        n_rt_layers = len(atmospheres[0].pressures)
        if len(setup["pressure"]) != n_rt_layers:
            print(
                f"WARNING: config uses {len(setup['pressure'])} pressure layers but the "
                f"Radtrans pickle has {n_rt_layers}; adopting the pickle grid for profiling."
            )
            pressure = np.asarray(atmospheres[0].pressures, dtype=float)
            setup["pressure"] = pressure
            setup["PT_kwargs"]["pressure"] = pressure
            setup["constant_params"]["pressure"] = pressure

    ret = Retrieval(
        setup["data"],
        setup["radtrans_kwargs"],
        setup["PT_kwargs"].copy(),
        setup["chem_kwargs"],
        setup["free_params"],
        setup["constant_params"],
        save_path=setup["save_path"],
        atmospheres=atmospheres,
    )
    return ret, setup


class TimerAccumulator:
    def __init__(self):
        self.records = []

    def add(self, label, mean_s, std_s=None, extra=""):
        self.records.append(
            {"label": label, "mean_s": mean_s, "std_s": std_s or 0.0, "extra": extra}
        )

    def print_table(self, title="Timing summary"):
        print_header(title)
        total = sum(r["mean_s"] for r in self.records)
        rows = sorted(self.records, key=lambda r: r["mean_s"], reverse=True)
        print(f"{'Section':<42} {'mean [s]':>10} {'std [s]':>10} {'fraction':>10}")
        print("-" * 76)
        for row in rows:
            frac = 100.0 * row["mean_s"] / total if total > 0 else 0.0
            std_str = f"{row['std_s']:.4f}" if row["std_s"] else "   -"
            extra = f"  {row['extra']}" if row["extra"] else ""
            print(
                f"{row['label']:<42} {row['mean_s']:10.4f} {std_str:>10} {frac:9.1f}%{extra}"
            )
        print("-" * 76)
        print(f"{'TOTAL':<42} {total:10.4f}")
        if rows:
            print(f"\nSlowest step: {rows[0]['label']} ({rows[0]['mean_s']:.4f} s)")
        return total


def evaluate_likelihood_timed(retrieval, params):
    """
    Single likelihood evaluation with per-section timers.

    Matches ``Retrieval.log_likelihood``: one ``calculate_flux`` call per order.
    """
    sections = {}

    t0 = time.perf_counter()
    temperature = retrieval.PT(params)
    sections["PT profile"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    vmrs = retrieval.chem(params, temperature=temperature)
    sections["FastChem"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mass_fractions = retrieval.chem.vmrs_to_mass_fractions(vmrs)
    sections["VMR → mass fractions"] = time.perf_counter() - t0

    log_g = params.get("log_g")
    radius = params.get("radius", 1.0)
    distance = params.get("distance")
    vsini = params.get("vsini", 0.0)
    epsilon_limb = params.get("epsilon_limb", 0.0)
    rv = params.get("rv", 0.0)

    raw_spectrum = []
    rt_per_order = []
    post_per_order = []

    for i, wave_i in enumerate(retrieval.wave):
        atmosphere = retrieval.atmospheres[i]
        atmosphere.use_jit_compilation = True
        atmosphere.use_fast_functions = True

        t0 = time.perf_counter()
        freq, flux_nu, _ = atmosphere.calculate_flux(
            temperatures=temperature,
            mass_fractions=mass_fractions,
            mean_molar_masses=mass_fractions["MMW"],
            reference_gravity=10.0 ** log_g,
            frequencies_to_wavelengths=False,
            return_contribution=False,
            fast_opacity_interpolation=True,
        )
        rt_per_order.append(time.perf_counter() - t0)

        wave_cm = cst.c / freq
        wave_um = wave_cm * 1e4
        flux = flux_nu * cst.c / wave_cm ** 2
        flux *= 1e-7
        if radius is not None and radius > 0.0:
            flux *= ((radius * cst.r_jup) / (distance * cst.pc)) ** 2

        t0 = time.perf_counter()
        if vsini > 0.0:
            wave_um, flux = retrieval.rotational_broadening(wave_um, flux, vsini, epsilon_limb)
        flux_broad = retrieval.instrumental_broadening(
            wave_um, flux, retrieval.instrumental_resolution
        )
        wave_um_rv = wave_um * (1 + rv / 2.998e5)
        spec_i = retrieval.rebin(wave_i, wave_um_rv, flux_broad)
        post_per_order.append(time.perf_counter() - t0)
        raw_spectrum.append(spec_i)

    sections["calculate_flux (all orders)"] = sum(rt_per_order)
    sections["Broadening + rebin (all orders)"] = sum(post_per_order)

    t0 = time.perf_counter()
    spectrum = retrieval.apply_spline_to_spectrum(params, raw_spectrum)
    sections["Spline marginalization (all orders)"] = time.perf_counter() - t0

    b = params.get("b", 0.0)
    t0 = time.perf_counter()
    total_lnL = 0.0
    for i in range(len(retrieval.flux)):
        wave_i = retrieval.wave[i]
        flux_i = retrieval.flux[i]
        err_i = retrieval.err[i]
        model_i = spectrum[i]

        finite = (
            np.isfinite(wave_i)
            & np.isfinite(flux_i)
            & np.isfinite(err_i)
            & (err_i > 0)
            & np.isfinite(model_i)
        )
        n_finite = int(np.count_nonzero(finite))
        if n_finite == 0:
            continue

        var_eff = 0.5 * err_i ** 2 * (1 + 10 ** (2 * b))
        inv_var = np.zeros_like(var_eff)
        inv_var[finite] = 1.0 / var_eff[finite]
        logdet = np.sum(np.log(var_eff[finite]))

        d_finite = np.nan_to_num(flux_i, nan=0.0)[finite]
        model_finite = model_i[finite]
        r = d_finite - model_finite
        chi2 = compute_chi2(r, inv_var[finite])
        lnL = -0.5 * (n_finite * retrieval.log2_pi + logdet + chi2)
        if not np.isfinite(lnL):
            total_lnL = -1e90
            break
        total_lnL += lnL

    sections["Chi-squared (all orders)"] = time.perf_counter() - t0

    return sections, rt_per_order, post_per_order, total_lnL


def profile_likelihood_evaluation(retrieval, params, n_repeat=3, warmup=1):
    """Average section timings over repeated single-pass likelihood evaluations."""
    for _ in range(warmup):
        evaluate_likelihood_timed(retrieval, params)

    accumulated = {}
    stds = {}
    rt_orders = []
    post_orders = []
    lnL = None

    for _ in range(n_repeat):
        sections, rt_per_order, post_per_order, lnL = evaluate_likelihood_timed(
            retrieval, params
        )
        for label, dt in sections.items():
            accumulated.setdefault(label, []).append(dt)
        rt_orders.append(rt_per_order)
        post_orders.append(post_per_order)

    acc = TimerAccumulator()
    for label, times in accumulated.items():
        acc.add(label, float(np.mean(times)), float(np.std(times)))

    mean_rt = np.mean(rt_orders, axis=0)
    mean_post = np.mean(post_orders, axis=0)
    return acc, mean_rt, mean_post, lnL


def profile_get_flux_order(retrieval, params, order_i):
    """One-shot sub-step breakdown for a single order (diagnostic only)."""
    atmosphere = retrieval.atmospheres[order_i]
    wave_data = retrieval.wave[order_i]
    temperature = retrieval.PT(params)
    vmrs = retrieval.chem(params, temperature=temperature)
    mass_fractions = retrieval.chem.vmrs_to_mass_fractions(vmrs)

    log_g = params.get("log_g")
    radius = params.get("radius", 1.0)
    distance = params.get("distance")
    vsini = params.get("vsini", 0.0)
    epsilon_limb = params.get("epsilon_limb", 0.0)
    rv = params.get("rv", 0.0)

    atmosphere.use_jit_compilation = True
    atmosphere.use_fast_functions = True

    timings = {}

    t0 = time.perf_counter()
    freq, flux_nu, _ = atmosphere.calculate_flux(
        temperatures=temperature,
        mass_fractions=mass_fractions,
        mean_molar_masses=mass_fractions["MMW"],
        reference_gravity=10.0 ** log_g,
        frequencies_to_wavelengths=False,
        return_contribution=False,
        fast_opacity_interpolation=True,
    )
    timings["calculate_flux"] = time.perf_counter() - t0

    wave_cm = cst.c / freq
    wave_um = wave_cm * 1e4
    flux = flux_nu * cst.c / wave_cm ** 2
    flux *= 1e-7
    if radius is not None and radius > 0.0:
        flux *= ((radius * cst.r_jup) / (distance * cst.pc)) ** 2

    t0 = time.perf_counter()
    if vsini > 0.0:
        wave_um, flux = retrieval.rotational_broadening(wave_um, flux, vsini, epsilon_limb)
    timings["rotational_broadening"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    flux_broad = retrieval.instrumental_broadening(
        wave_um, flux, retrieval.instrumental_resolution
    )
    timings["instrumental_broadening"] = time.perf_counter() - t0

    wave_um_rv = wave_um * (1 + rv / 2.998e5)

    t0 = time.perf_counter()
    retrieval.rebin(wave_data, wave_um_rv, flux_broad)
    timings["rebin"] = time.perf_counter() - t0

    return timings


def verify_single_rt_pass(retrieval, params):
    """Confirm log_likelihood invokes calculate_flux exactly once per order."""
    call_counts = []

    for atmosphere in retrieval.atmospheres:
        original = atmosphere.calculate_flux

        def counted(*args, _orig=original, _counts=call_counts, **kwargs):
            _counts.append(1)
            return _orig(*args, **kwargs)

        atmosphere.calculate_flux = counted

    cube = prior_center_cube(retrieval.prior_transform.free_params)
    set_params_from_cube(retrieval, cube)
    retrieval.log_likelihood(cube)

    n_orders = len(retrieval.atmospheres)
    n_calls = len(call_counts)
    print_header("calculate_flux call count (single log_likelihood)")
    print(f"  Orders: {n_orders}")
    print(f"  calculate_flux calls: {n_calls}")
    if n_calls == n_orders:
        print("  OK: exactly one RT call per order, no redundant evaluation.")
    else:
        print(f"  WARNING: expected {n_orders} calls, got {n_calls}.")


def print_efficiency_recommendations(retrieval, setup, component_total, lnl_time):
    print_header("Efficiency improvement opportunities")

    n_orders = len(retrieval.wave)
    n_layers = len(retrieval.pressure)
    lbl_sampling = setup["radtrans_kwargs"].get("line_by_line_opacity_sampling", 1)
    n_spline = retrieval.n_spline

    items = [
        (
            "Radiative transfer dominates",
            "petitRADTRANS ``calculate_flux`` is the bottleneck (~65–85% of each "
            "likelihood call). All other steps are comparatively cheap.",
        ),
        (
            "Per-order Radtrans objects (already enabled)",
            "Narrow ``wavelength_boundaries`` per echelle order avoids building "
            "opacity on the full M-band grid. Rebuild ``radtrans.pkl`` after "
            "changing orders.",
        ),
        (
            "Line-by-line sampling",
            f"Current ``line_by_line_opacity_sampling={lbl_sampling}``. "
            "Increase to 5–10 for exploratory runs; use 1–3 for production.",
        ),
        (
            "Pressure grid",
            f"Current {n_layers} layers. Reducing to 24–30 layers speeds RT with "
            "minor impact on continuum; keep pickle and config layer counts matched.",
        ),
        (
            "Opacity interpolation",
            "``fast_opacity_interpolation=True`` is already set in ``get_flux``. "
            "Keep ``return_contribution=False`` during retrievals.",
        ),
        (
            "JIT / fast functions",
            "``use_jit_compilation`` and ``use_fast_functions`` are enabled on "
            "each atmosphere. Warm up with one likelihood call before timing.",
        ),
        (
            "Spline marginalization",
            f"``n_spline={n_spline}`` adds ~10–15% overhead. Consider ``n_spline=0`` "
            "when continuum is already normalized and well modelled.",
        ),
        (
            "FastChem caching",
            "FastChem (~5%) is already fast. Caching by (T, metallicity, C/O) hash "
            "only helps if many likelihood calls share identical thermochemistry — "
            "unlikely during nested sampling.",
        ),
        (
            "Parameter-subspace caching",
            "When only ``rv``/``vsini`` vary, RT outputs could be cached and re-broadened "
            "without re-running ``calculate_flux``. Not implemented; useful for "
            "grid searches over kinematics.",
        ),
        (
            "Rebinning",
            "``spectres`` is used when available; ``np.interp`` fallback is faster but "
            "less accurate at sharp lines. Precompute valid-pixel indices per order "
            "to avoid repeated ``isfinite`` masks in the likelihood loop.",
        ),
        (
            "Parallelism",
            f"{n_orders} independent orders could be evaluated in parallel (multiprocessing "
            "or joblib) since each order has its own Radtrans object. Not yet implemented.",
        ),
        (
            "Single-pass likelihood (already enabled)",
            "``log_likelihood`` calls ``compute_spectrum`` once (one ``calculate_flux`` "
            "per order), then ``apply_spline_to_spectrum``, then chi-squared. "
            "``get_spectrum`` is a convenience wrapper for plotting — not used in sampling.",
        ),
    ]

    for title, text in items:
        print(f"\n• {title}")
        print(f"  {text}")

    if lnl_time and component_total:
        delta = lnl_time - component_total
        print(
            f"\nMeasured component sum: {component_total:.3f} s  "
            f"vs log_likelihood: {lnl_time:.3f} s  (Δ={delta:+.3f} s)"
        )


def run_cprofile(retrieval, params):
    cube = prior_center_cube(retrieval.prior_transform.free_params)
    set_params_from_cube(retrieval, cube)

    profiler = cProfile.Profile()
    profiler.enable()
    retrieval.log_likelihood(cube)
    profiler.disable()

    print_header("cProfile: log_likelihood (top 40 by cumulative time)")
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(40)
    print(stream.getvalue())


def parse_args():
    parser = argparse.ArgumentParser(description="Profile sherpa likelihood evaluation speed")
    parser.add_argument("--n-repeat", type=int, default=3, help="Timed repetitions per section")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations before timing")
    parser.add_argument(
        "--order",
        type=int,
        default=0,
        help="Echelle order index for detailed get_flux breakdown",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Run cProfile on a full log_likelihood evaluation",
    )
    parser.add_argument(
        "--build-radtrans",
        action="store_true",
        help="Build Radtrans objects instead of loading the pickle (slow)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print_header("Building retrieval")
    t0 = time.perf_counter()
    retrieval, setup = build_retrieval(load_pickle=not args.build_radtrans)
    print(f"Retrieval ready in {time.perf_counter() - t0:.1f} s")
    print(f"  Orders: {len(retrieval.wave)}")
    print(f"  Free parameters: {len(retrieval.parameter_names)}")
    print(f"  Radtrans objects: {len(retrieval.atmospheres)}")

    cube = prior_center_cube(setup["free_params"])
    params = set_params_from_cube(retrieval, cube)
    print("\nParameter vector: prior centres (cube=0.5)")

    verify_single_rt_pass(retrieval, params)

    acc, mean_rt, mean_post, lnL = profile_likelihood_evaluation(
        retrieval, params, n_repeat=args.n_repeat, warmup=args.warmup
    )
    component_total = acc.print_table("Single likelihood evaluation (one RT call per order)")
    print(f"Reference lnL: {lnL:.2f}")

    print_header("calculate_flux per order")
    for i, (rt_s, post_s) in enumerate(zip(mean_rt, mean_post)):
        n_pix = len(retrieval.wave[i])
        ms_per_pix = 1e3 * rt_s / max(n_pix, 1)
        print(
            f"  order {i:2d}: RT {rt_s:7.4f} s  post {post_s:7.4f} s  "
            f"({ms_per_pix:.3f} ms/pix RT, {n_pix} pix)"
        )

    order_i = args.order
    if order_i < 0 or order_i >= len(retrieval.wave):
        raise ValueError(f"Order index {order_i} out of range [0, {len(retrieval.wave) - 1}]")

    flux_timings = profile_get_flux_order(retrieval, params, order_i)
    print_header(f"get_flux sub-steps (order {order_i}, single diagnostic run)")
    subtotal = sum(flux_timings.values())
    for label, dt in sorted(flux_timings.items(), key=lambda x: -x[1]):
        frac = 100.0 * dt / subtotal if subtotal > 0 else 0.0
        print(f"  {label:<28} {dt:8.4f} s  ({frac:5.1f}%)")
    print(f"  {'subtotal':<28} {subtotal:8.4f} s")

    # Reference timing via production entry point (should match component sum)
    lnl_times = []
    for _ in range(args.warmup):
        cube = prior_center_cube(setup["free_params"])
        set_params_from_cube(retrieval, cube)
        retrieval.log_likelihood(cube)
    for _ in range(args.n_repeat):
        cube = prior_center_cube(setup["free_params"])
        set_params_from_cube(retrieval, cube)
        t0 = time.perf_counter()
        retrieval.log_likelihood(cube)
        lnl_times.append(time.perf_counter() - t0)
    lnl_mean = float(np.mean(lnl_times))
    print_header("Production log_likelihood reference")
    print(f"  mean: {lnl_mean:.4f} s  (std: {np.std(lnl_times):.4f} s)")

    print_efficiency_recommendations(retrieval, setup, component_total, lnl_mean)

    if args.cprofile:
        run_cprofile(retrieval, params)


if __name__ == "__main__":
    main()

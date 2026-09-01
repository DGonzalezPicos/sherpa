"""
Run the GJ 3820 M-band retrieval using a precomputed Radtrans object.

Run the model setup first:

    python examples/gj3820_model.py

Then launch the retrieval on one CPU:

    python examples/gj3820_retrieval.py

Or with MPI:

    mpiexec -np 42 python examples/gj3820_retrieval.py
"""
import argparse
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sherpa.retrieval import Retrieval, is_main_process, load_radtrans

if not is_main_process():
    warnings.filterwarnings("ignore", category=FutureWarning)

from gj3820_config import build_gj3820_setup

from sherpa.evaluation import evaluate_posterior


def parse_args():
    parser = argparse.ArgumentParser(description="Run the GJ 3820 M-band retrieval")
    parser.add_argument("--nlive", type=int, default=200, help="Number of live points")
    parser.add_argument("--resume", action="store_true", help="Resume an existing MultiNest run", default=False)
    parser.add_argument("--live-plot", action="store_true", help="Write live diagnostic plots during sampling", default=True)
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip final posterior evaluation", default=False)
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip the retrieval step", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    quiet = not is_main_process()
    setup = build_gj3820_setup(use_data_cache=True, quiet=quiet)

    if not setup["radtrans_pickle"].is_file():
        if is_main_process():
            print(f"Missing Radtrans pickle: {setup['radtrans_pickle']}")
            print("Run examples/gj3820_model.py first.")
        sys.exit(1)

    atmospheres = load_radtrans(setup["radtrans_pickle"])

    ret = Retrieval(
        setup["data"],
        setup["radtrans_kwargs"],
        setup["PT_kwargs"],
        setup["chem_kwargs"],
        setup["free_params"],
        setup["constant_params"],
        save_path=setup["save_path"],
        atmospheres=atmospheres,
    )

    if is_main_process():
        print(f"Free parameters: {ret.parameter_names}")
        print(f"Number of dimensions: {ret.ndim}")
        print(f"Output directory: {setup['save_path']}")
        print(f"Using Radtrans pickle: {setup['radtrans_pickle']}")

    if not args.skip_retrieval:
        ret.run(
            nlive=args.nlive,
            resume=args.resume,
            live_plot=args.live_plot,
            sampling_efficiency=0.10,  # lower to <0.05 for production runs
            evidence_tolerance=1.0,   # lower to <0.5 for production runs
            n_iter_before_update=200,
        )

    if is_main_process() and not args.skip_evaluation:
        print("Running final posterior evaluation...")
        evaluate_posterior(ret, run_path=setup["save_path"], suffix="final")


if __name__ == "__main__":
    main()

"""
Build the forward model for GJ 3820 and run a prior check.

This script:
1. Creates the petitRADTRANS Radtrans object (slow step).
2. Saves it to a pickle file for fast reuse in the retrieval script.
3. Evaluates the model at the edges and center of the prior distributions.
4. Writes prior-check diagnostic plots (spectrum and PT/VMR).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gj3820_config import build_gj3820_setup

from sherpa.retrieval import Retrieval, save_radtrans


def main():
    setup = build_gj3820_setup()
    setup["data_obj"].summary()

    ret = Retrieval(
        setup["data"],
        setup["radtrans_kwargs"],
        setup["PT_kwargs"],
        setup["chem_kwargs"],
        setup["free_params"],
        setup["constant_params"],
        save_path=setup["save_path"],
    )

    save_radtrans(ret.atmospheres, setup["radtrans_pickle"])

    print(f"Free parameters: {ret.parameter_names}")
    print(f"Number of dimensions: {ret.ndim}")
    print(f"Output directory: {setup['save_path']}")

    ret.prior_check(n=3, random=False, save_dir=setup["save_path"] / "plots")


if __name__ == "__main__":
    main()

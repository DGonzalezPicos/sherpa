"""Shared configuration for the GJ 3820 M-band example scripts."""
import os
import pathlib

import numpy as np

from petitRADTRANS.config import petitradtrans_config_parser

from sherpa.data import IshellMband, load_retrieval_data_cache, save_retrieval_data_cache


def configure_prt():
    prt_input_path = os.environ.get("PRT_INPUT_DATA_PATH")
    if prt_input_path is not None:
        petitradtrans_config_parser.set_input_data_path(prt_input_path)


def _is_main_process():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        return True


def _mpi_barrier():
    try:
        from mpi4py import MPI
        MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass


def _load_processed_data(target, data_dir, file_path, quiet):
    data_obj = IshellMband(target_name=target, file_path=file_path)
    orders = np.arange(2, 11)
    data_obj.select_orders(orders)

    telluric_template = data_dir / "telluric_template.fits"
    data_obj.apply_telluric_mask(
        telluric_template,
        threshold=0.50,
        grow_mask=5,
        quiet=quiet,
    )

    data_obj.remove_empty_orders(n_minimum_pixels=100)
    data_obj.normalize_orders()
    return data_obj


def build_gj3820_setup(use_data_cache=False, quiet=False):
    """Return dictionaries and paths used by the model and retrieval scripts."""
    configure_prt()

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    target = "gj3820"
    data_dir = pathlib.Path(__file__).resolve().parent / "data" / target
    file_path = data_dir / f"{target}_m1_tellcor_merged.fits"

    line_by_line_opacity_sampling = 2
    run_name = f"{target}_lbl{line_by_line_opacity_sampling}_0"
    save_path = repo_root / "retrievals" / run_name
    save_path.mkdir(parents=True, exist_ok=True)
    data_cache_path = save_path / "data_cache.pkl"

    data_obj = None
    cached_data = load_retrieval_data_cache(data_cache_path) if use_data_cache else None
    if use_data_cache and cached_data is None:
        if _is_main_process():
            data_obj = _load_processed_data(target, data_dir, file_path, quiet=quiet)
            save_retrieval_data_cache(
                data_cache_path,
                data_obj.wave,
                data_obj.flux,
                data_obj.err,
            )
        _mpi_barrier()
        cached_data = load_retrieval_data_cache(data_cache_path)
        if cached_data is None:
            raise FileNotFoundError(f"Processed data cache not found: {data_cache_path}")

    if cached_data is not None:
        wave_obs = cached_data["wave"]
        flux_obs = cached_data["flux"]
        err_obs = cached_data["err"]
    else:
        data_obj = _load_processed_data(target, data_dir, file_path, quiet=quiet)
        wave_obs = data_obj.wave
        flux_obs = data_obj.flux
        err_obs = data_obj.err
        save_retrieval_data_cache(data_cache_path, wave_obs, flux_obs, err_obs)

    wave_range = [np.nanmin(wave_obs[0]), np.nanmax(wave_obs[-1])]

    n_atmosphere_layers = 30
    pressure = np.logspace(-5.0, 2.0, n_atmosphere_layers)

    molecules = ["H2O", "CO", "OH", "CN"]
    atoms = ["Fe", "Na"]
    isotopes = ["13CO", "C18O"]
    isotopes_dict = {
        "CO": ["13CO", "C18O", "C17O"],
        "H2O": ["H2O_181", "H2O_171"],
    }
    isotopes_dict_rev = {v: k for k, values in isotopes_dict.items() for v in values}
    species_of_interest = molecules + atoms + isotopes

    species_info_file = repo_root / "sherpa" / "data" / "chemistry" / "species_info.txt"
    species_info = np.loadtxt(species_info_file, dtype=str)
    for species in species_of_interest:
        assert species in species_info[:, 0], f"Species {species} not found in species_info.txt"
    species_info = species_info[np.isin(species_info[:, 0], species_of_interest)]
    line_species = list(species_info[:, 1])

    chem_kwargs = {
        "species_info_file": str(species_info_file),
        "quiet": quiet,
    }

    PT_kwargs = {
        "pressure": pressure,
        "PT_mode": "RCE",
        "PT_interp_mode": "linear",
    }

    radtrans_kwargs = {
        "line_species": line_species,
        "rayleigh_species": ["H2", "He"],
        "gas_continuum_contributors": ["H2-H2", "H2-He", "H-"],
        "line_opacity_mode": "lbl",
        "line_by_line_opacity_sampling": line_by_line_opacity_sampling,
    }

    parallax_mas = 100.0
    distance_pc = 1e3 / parallax_mas

    rv_max = 50.0
    n_pressure_levels = 5

    constant_params = {
        "distance": distance_pc,
        "wave_range": wave_range,
        "pressure": pressure,
        "grating": None,
        "instrumental_resolution": 70e3,
        "n_spline": 0,
        "normalize_flux": True,
        "return_contribution": False,
        "n_pressure_levels": n_pressure_levels,
    }

    free_params = {
        "rv": (-rv_max, rv_max),
        "vsini": (1.0, 20.0),
        "b": (-1.0, 2.0),
        "log_g": (4.0, 6.0),
        "T_RCE": (2200.0, 4800.0),
        "log_P_RCE": (-2.0, 1.0),
        "dlog_P": (0.2, 1.8),
        "metallicity": (-0.60, 0.40),
        "c_to_o": (0.2, 0.9),
    }

    for i in range(n_pressure_levels):
        free_params[f"T_grad_{i}"] = (0.02, 0.72)

    for species in species_of_interest:
        if species in isotopes:
            main = isotopes_dict_rev[species]
            # log(isotope/main): secondary species VMR = main VMR * 10**log(isotope/main)
            free_params[f"log_{species}/{main}"] = (-3.6, 0.0)

    data = {"wave": wave_obs, "flux": flux_obs, "err": err_obs}

    return {
        "target": target,
        "data_obj": data_obj,
        "data": data,
        "pressure": pressure,
        "radtrans_kwargs": radtrans_kwargs,
        "PT_kwargs": PT_kwargs,
        "chem_kwargs": chem_kwargs,
        "free_params": free_params,
        "constant_params": constant_params,
        "save_path": save_path,
        "run_name": run_name,
        "radtrans_pickle": save_path / "radtrans.pkl",
        "data_cache": data_cache_path,
    }

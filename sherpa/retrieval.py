import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import nnls
import pymultinest
from PyAstronomy import pyasl

from petitRADTRANS.radtrans import Radtrans
from petitRADTRANS import physical_constants as cst

from .pressure_temperature import PressureTemperature, PressureTemperatureGradients
from .chemistry import FastChemistry
from .data import normalize_order_by_median

try:
    from spectres import spectres
    HAS_SPECTRES = True
except ImportError:
    HAS_SPECTRES = False


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


def is_main_process():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        return True


def save_radtrans(atmospheres, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(atmospheres, list):
        atmospheres = [atmospheres]
    with open(path, "wb") as f:
        pickle.dump(atmospheres, f)
    if is_main_process():
        print(f"Saved {len(atmospheres)} Radtrans object(s) to {path}")


def load_radtrans(path):
    with open(path, "rb") as f:
        atmospheres = pickle.load(f)
    if not isinstance(atmospheres, list):
        atmospheres = [atmospheres]
    if is_main_process():
        print(f"Loaded {len(atmospheres)} Radtrans object(s) from {path}")
    return atmospheres


def multinest_output_basename(save_path=None):
    """
    Build a PyMultiNest ``outputfiles_basename`` that writes into ``pmn/``.

    PyMultiNest concatenates ``basename + filename`` with no path separator.
    The basename must therefore end with ``/`` so outputs land in
    ``pmn/post_equal_weights.dat`` instead of ``pmnpost_equal_weights.dat``.
    """
    if save_path is None:
        return "pmn_"
    pmn_dir = pathlib.Path(save_path) / "pmn"
    pmn_dir.mkdir(parents=True, exist_ok=True)
    return pmn_dir.as_posix() + "/"


class SplineModel:
    """Simple spline model adapted from the BREADS project."""

    def __init__(self, N_knots=10, spline_degree=3):
        self.N_knots = N_knots
        self.spline_degree = spline_degree
        if self.N_knots <= self.spline_degree:
            self.spline_degree = self.N_knots - 1

    def get_spline_model(self):
        assert hasattr(self, "x_knots")
        assert hasattr(self, "x_samples")
        self.x_knots = np.atleast_2d(self.x_knots)
        M_list = []
        for nodes in self.x_knots:
            M = np.zeros((np.size(self.x_samples), np.size(nodes)))
            mn, mx = np.min(nodes), np.max(nodes)
            inbounds = np.where((mn < self.x_samples) & (self.x_samples < mx))
            _x = self.x_samples[inbounds]
            for chunk in range(np.size(nodes)):
                tmp_y_vec = np.zeros(np.size(nodes))
                tmp_y_vec[chunk] = 1
                spl = InterpolatedUnivariateSpline(nodes, tmp_y_vec, k=self.spline_degree, ext=0)
                M[inbounds[0], chunk] = spl(_x)
            M_list.append(M)
        return np.concatenate(M_list, axis=1)

    def __call__(self, spec):
        self.x_samples = np.arange(spec.size)
        self.x_knots = np.linspace(-1, self.x_samples.size + 1, self.N_knots)
        self.spline_spec = self.get_spline_model().T * spec
        return self.spline_spec


def make_spline(spectrum, n_spline=10):
    return SplineModel(N_knots=n_spline, spline_degree=3)(spectrum)


def apply_spline_correction(retrieval, wave, flux, err, model, b=0.0):
    """Fit spline amplitudes and return the marginalized model spectrum."""
    if retrieval.n_spline <= 3:
        return model

    finite = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & np.isfinite(err)
        & (err > 0)
        & np.isfinite(model)
    )
    if not np.any(finite):
        return model

    var_eff = 0.5 * err ** 2 * (1 + 10 ** (2 * b))
    inv_var = np.zeros_like(var_eff)
    inv_var[finite] = 1.0 / var_eff[finite]

    model_finite = model[finite]
    flux_finite = flux[finite]

    # Rescale the model to the data level before building the spline basis.
    # This is equivalent for the fit but avoids catastrophic cancellation when
    # the petitRADTRANS flux and normalized data differ by many orders of magnitude.
    amp_scale = 1.0
    positive = np.abs(model_finite) > 0
    if np.any(positive):
        model_median = float(np.nanmedian(np.abs(model_finite[positive])))
        flux_median = float(np.nanmedian(flux_finite))
        if model_median > 0 and flux_median > 0:
            amp_scale = flux_median / model_median
    model_scaled = model_finite * amp_scale

    M = make_spline(model_scaled, n_spline=retrieval.n_spline)
    phi = solve_linear(flux_finite, inv_var[finite], M, use_nnls=True)
    corrected = model.copy()
    corrected[finite] = phi @ M
    return corrected


def solve_linear(d, C_inv, M, use_nnls=True):
    d = np.asarray(d, dtype=float)
    C_inv = np.asarray(C_inv, dtype=float)
    M = np.nan_to_num(M, nan=0.0)

    invalid_C = ~np.isfinite(C_inv) | (C_inv < 0)
    if np.any(invalid_C):
        C_inv = np.where(invalid_C, 0.0, C_inv)

    if C_inv.ndim == 1:
        weighted_M = M * C_inv[np.newaxis, :]
        lhs = weighted_M @ M.T
        rhs = weighted_M @ d
    elif C_inv.ndim == 2:
        lhs = M.T @ C_inv @ M
        rhs = M.T @ C_inv @ d
    else:
        raise ValueError("C_inv must be 1D or 2D")

    if np.all(lhs == 0) or np.all(rhs == 0):
        return np.zeros(lhs.shape[0], dtype=float)

    if use_nnls:
        try:
            return nnls(lhs, rhs)[0]
        except Exception:
            return np.zeros(lhs.shape[0], dtype=float)
    else:
        return np.linalg.solve(lhs, rhs)


def compute_chi2(r, C_inv):
    if C_inv.ndim == 1:
        return np.sum(r ** 2 * C_inv)
    else:
        return r.T @ C_inv @ r


class Parameters:
    """Prior transform and parameter bookkeeping."""

    def __init__(self, free_params, constant_params=None):
        if constant_params is None:
            constant_params = {}
        self.free_params = free_params
        self.constant_params = constant_params
        self.params = self.constant_params.copy()

    def __call__(self, cube, ndim=None, nparams=None):
        cube = self.transform_cube(cube)
        self.get_derived_params()
        return cube

    def transform_cube(self, cube):
        for i, (k, v) in enumerate(self.free_params.items()):
            v_min, v_max = v
            cube[i] = v_min + (v_max - v_min) * cube[i]
            self.params[k] = cube[i]
        return cube

    def get_derived_params(self):
        alias = {
            "metallicity": "z",
            "c_to_o": "c_o",
        }
        for k, v in alias.items():
            if k in self.params:
                self.params[v] = self.params[k]

        params_copy = self.params.copy()
        for k in params_copy:
            if k.startswith("log_"):
                self.params[k[4:]] = 10 ** self.params[k]
        return self.params


class Retrieval:
    """Minimal MultiNest retrieval driver."""

    log2_pi = np.log(2 * np.pi)

    def __init__(self, data, radtrans_kwargs, PT_kwargs, chem_kwargs,
                 free_params, constant_params, save_path=None, gaussian_params=None,
                 atmosphere=None, atmospheres=None):
        if gaussian_params is None:
            gaussian_params = []
        self.gaussian_params = gaussian_params

        self.wave = self._at_least_2d(data.get("wave"))
        self.flux = self._at_least_2d(data.get("flux"))
        self.err = self._at_least_2d(data.get("err"))

        self.parameter_names = list(free_params.keys())
        self.ndim = len(self.parameter_names)
        self.save_path = pathlib.Path(save_path) if save_path is not None else None

        if self.save_path is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)

        self.constant_params = constant_params
        self.instrumental_resolution = constant_params.get("instrumental_resolution", None)
        self.n_spline = constant_params.get("n_spline", 0)
        self.normalize_flux = constant_params.get("normalize_flux", False)

        rv_max = 50.0
        for key, value in free_params.items():
            if key.startswith("rv") and isinstance(value, (list, tuple)) and len(value) >= 2:
                rv_max = max(rv_max, abs(value[0]), abs(value[1]))

        wave_median = [np.nanmedian(wave_i) for wave_i in self.wave]
        wave_pad = [wave_median_i * (rv_max / 2.998e5) for wave_median_i in wave_median]
        self.wave_ranges = self.set_wave_ranges(self.wave, wave_pad)

        assert "pressure" in PT_kwargs
        pressure = PT_kwargs.pop("pressure")
        self.pressure = pressure

        PT_mode = PT_kwargs.pop("PT_mode")
        assert PT_mode == "RCE", "Only RCE mode is supported in this minimal package"
        self.PT = PressureTemperatureGradients(pressure, **PT_kwargs)

        self.radtrans_kwargs = radtrans_kwargs.copy()

        if atmospheres is None and atmosphere is not None:
            atmospheres = atmosphere if isinstance(atmosphere, list) else [atmosphere]

        if atmospheres is None:
            self.atmospheres = self.set_atmospheres(pressure, self.wave_ranges, self.radtrans_kwargs)
        else:
            self.atmospheres = atmospheres if isinstance(atmospheres, list) else [atmospheres]
            if len(self.atmospheres) != len(self.wave):
                raise ValueError(
                    f"Number of atmospheres ({len(self.atmospheres)}) must match "
                    f"number of data orders ({len(self.wave)})"
                )

        line_species = np.unique(np.concatenate([atm.line_species for atm in self.atmospheres])).tolist()
        self.chem = self._set_chemistry(line_species, pressure, chem_kwargs)

        self.prior_transform = Parameters(free_params, constant_params)
        self.return_spectrum = False

    @property
    def atmosphere(self):
        if len(self.atmospheres) != 1:
            raise ValueError(
                f"Multiple atmospheres present ({len(self.atmospheres)}). Use .atmospheres[i] instead."
            )
        return self.atmospheres[0]

    @property
    def wave_range(self):
        return self.wave_ranges

    def _at_least_2d(self, x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                return [x]
            elif x.ndim == 2:
                return [x[i] for i in range(x.shape[0])]
            else:
                raise ValueError("Unsupported array dimension")
        elif isinstance(x, list):
            if len(x) == 0:
                return None
            if isinstance(x[0], (list, np.ndarray)):
                return x
            return [np.array(x)]
        return [x]

    def set_wave_ranges(self, wave_list, wave_pad):
        wave_ranges = []
        for i, wave_i in enumerate(wave_list):
            if wave_i is not None and len(wave_i) > 0:
                wave_min = np.nanmin(wave_i) - wave_pad[i]
                wave_max = np.nanmax(wave_i) + wave_pad[i]
                wave_ranges.append([wave_min, wave_max])
            else:
                wave_ranges.append([1.0, 10.0])
        self.check_wave_ranges(wave_ranges)
        return wave_ranges

    def check_wave_ranges(self, wave_ranges):
        for i, wave_range in enumerate(wave_ranges):
            data_min = np.nanmin(self.wave[i])
            data_max = np.nanmax(self.wave[i])
            assert wave_range[0] <= data_min, (
                f"wave_range {i} lower bound {wave_range[0]:.4f} exceeds data minimum {data_min:.4f}"
            )
            assert wave_range[1] >= data_max, (
                f"wave_range {i} upper bound {wave_range[1]:.4f} is below data maximum {data_max:.4f}"
            )

    def set_atmospheres(self, pressure, wave_ranges, radtrans_kwargs):
        atmospheres = []
        for i, wave_range_i in enumerate(wave_ranges):
            radtrans_kwargs_copy = radtrans_kwargs.copy()
            radtrans_kwargs_copy["wavelength_boundaries"] = wave_range_i
            atmosphere = Radtrans(pressures=pressure, **radtrans_kwargs_copy)
            atmospheres.append(atmosphere)
        return atmospheres

    def _set_chemistry(self, line_species, pressure, chem_kwargs):
        return FastChemistry(line_species, pressure, **chem_kwargs)

    def log_likelihood(self, cube, ndim=None, nparams=None):
        params = self.prior_transform.params.copy()

        try:
            raw_spectrum, _ = self.compute_spectrum(params)
            spectrum = self.apply_spline_to_spectrum(params, raw_spectrum)
        except Exception as e:
            if is_main_process():
                print(f"Model evaluation failed: {e}")
            return -1e90

        b = params.get("b", 0.0)
        total_lnL = 0.0

        for i in range(len(self.flux)):
            wave_i = self.wave[i]
            flux_i = self.flux[i]
            err_i = self.err[i]
            model_i = spectrum[i]

            finite = np.isfinite(wave_i) & np.isfinite(flux_i) & np.isfinite(err_i) & (err_i > 0) & np.isfinite(model_i)
            n_finite = int(np.count_nonzero(finite))
            if n_finite == 0:
                continue

            var_eff = 0.5 * err_i ** 2 * (1 + 10 ** (2 * b))
            inv_var = np.zeros_like(var_eff)
            inv_var[finite] = 1.0 / var_eff[finite]
            logdet = np.sum(np.log(var_eff[finite]))

            d = np.nan_to_num(flux_i, nan=0.0)
            d_finite = d[finite]
            model_finite = model_i[finite]

            r = d_finite - model_finite
            chi2 = compute_chi2(r, inv_var[finite])
            N = n_finite
            lnL = -0.5 * (N * self.log2_pi + logdet + chi2)

            if not np.isfinite(lnL):
                return -1e90
            total_lnL += lnL

        return total_lnL if np.isfinite(total_lnL) else -1e90

    def compute_spectrum(self, params):
        """Compute the raw petitRADTRANS spectrum for each data order."""
        distance = params.get("distance")
        log_g = params.get("log_g")
        radius = params.get("radius", 1.0)
        vsini = params.get("vsini", 0.0)
        epsilon_limb = params.get("epsilon_limb", 0.0)
        rv = params.get("rv", 0.0)
        return_contribution = params.get("return_contribution", False)

        temperature = self.PT(params)
        vmrs = self.chem(params, temperature=temperature)

        if not isinstance(vmrs, dict):
            raise ValueError("Invalid chemistry output")

        mass_fractions = self.chem.vmrs_to_mass_fractions(vmrs)

        spectrum = []
        additional_outputs = []
        for i, wave_i in enumerate(self.wave):
            spec_i, outputs_i, _ = self.get_flux(
                self.atmospheres[i],
                temperature,
                mass_fractions,
                log_g,
                radius,
                distance,
                wave_data=wave_i,
                vsini=vsini,
                epsilon_limb=epsilon_limb,
                rv=rv,
                return_contribution=return_contribution,
            )
            if self.normalize_flux:
                spec_i, _ = normalize_order_by_median(spec_i)
            spectrum.append(spec_i)
            additional_outputs.append(outputs_i if outputs_i is not None else {})

        return spectrum, additional_outputs

    def apply_spline_to_spectrum(self, params, spectrum):
        """Apply the same spline marginalization used in the likelihood."""
        if self.n_spline <= 3:
            return spectrum

        b = params.get("b", 0.0)
        return [
            apply_spline_correction(self, self.wave[i], self.flux[i], self.err[i], spec_i, b=b)
            for i, spec_i in enumerate(spectrum)
        ]

    def get_spectrum(self, params):
        """Return the marginalized model spectrum used in the likelihood."""
        spectrum, _ = self.compute_spectrum(params)
        return self.apply_spline_to_spectrum(params, spectrum)

    @staticmethod
    def _standardize_emission_contribution(emission_contribution, n_pressure_layers):
        """Return emission contribution with shape (n_pressure, n_wavelength)."""
        ec = np.asarray(emission_contribution, dtype=float)
        if ec.ndim != 2:
            raise ValueError(
                f"emission_contribution must be 2D, got shape {ec.shape}"
            )
        if ec.shape[0] == n_pressure_layers:
            return ec
        if ec.shape[1] == n_pressure_layers:
            return ec.T
        raise ValueError(
            f"emission_contribution shape {ec.shape} is incompatible with "
            f"{n_pressure_layers} pressure layers"
        )

    @staticmethod
    def _emission_contribution_wavelengths(atmosphere, n_wave_ec, wave_um):
        """Wavelength grid [um] for the emission-contribution axis."""
        wave_um = np.asarray(wave_um, dtype=float)
        if n_wave_ec == wave_um.size:
            return wave_um

        edges = getattr(atmosphere, "frequency_bins_edges", None)
        if edges is not None:
            edges = np.asarray(edges, dtype=float)
            wave_edges_um = (cst.c / edges) * 1e4
            if n_wave_ec == wave_edges_um.size - 1:
                return 0.5 * (wave_edges_um[:-1] + wave_edges_um[1:])
            if n_wave_ec == wave_edges_um.size:
                return wave_edges_um

        raise ValueError(
            f"Cannot match emission_contribution wavelength axis (size {n_wave_ec}) "
            f"to the radiative-transfer grid (size {wave_um.size})"
        )

    def _broaden_and_rebin_to_data(
        self,
        wave_um,
        spectrum,
        wave_data,
        vsini=0.0,
        epsilon_limb=0.0,
        rv=0.0,
    ):
        """
        Apply the same rotation, instrumental broadening, RV shift, and rebin
        used for the model flux in ``get_flux``.
        """
        wave_um = np.asarray(wave_um, dtype=float)
        spectrum = np.asarray(spectrum, dtype=float)
        if vsini > 0.0:
            wave_um, spectrum = self.rotational_broadening(
                wave_um, spectrum, vsini, epsilon_limb
            )
        spectrum_broad = self.instrumental_broadening(
            wave_um, spectrum, self.instrumental_resolution
        )
        wave_um_rv = wave_um * (1.0 + rv / 2.998e5)
        return self.rebin(wave_data, wave_um_rv, spectrum_broad)

    def _process_emission_contribution(
        self,
        atmosphere,
        emission_contribution,
        wave_um,
        wave_data,
        vsini=0.0,
        epsilon_limb=0.0,
        rv=0.0,
    ):
        """Broaden and rebin layer-wise emission contribution onto the data grid."""
        ec = self._standardize_emission_contribution(emission_contribution, len(self.pressure))
        wave_ec = self._emission_contribution_wavelengths(atmosphere, ec.shape[1], wave_um)
        wave_um = np.asarray(wave_um, dtype=float)
        wave_data = np.asarray(wave_data, dtype=float)
        ec_data = np.zeros((ec.shape[0], wave_data.size), dtype=float)

        same_grid = (
            wave_ec.size == wave_um.size
            and np.allclose(wave_ec, wave_um, rtol=1e-6, atol=1e-8, equal_nan=True)
        )
        for layer_idx in range(ec.shape[0]):
            if same_grid:
                layer_um = np.asarray(ec[layer_idx], dtype=float)
            else:
                layer_um = self.rebin(wave_um, wave_ec, ec[layer_idx])
            ec_data[layer_idx] = self._broaden_and_rebin_to_data(
                wave_um,
                layer_um,
                wave_data,
                vsini=vsini,
                epsilon_limb=epsilon_limb,
                rv=rv,
            )

        valid_wave = np.isfinite(wave_data)
        ec_data[:, ~valid_wave] = np.nan
        ec_data[~np.isfinite(ec_data)] = np.nan

        return ec_data

    def get_flux(self, atmosphere, temperature, mass_fractions, log_g, radius, distance,
                 wave_data, vsini=0.0, epsilon_limb=0.0, rv=0.0, return_contribution=False):
        atmosphere.use_jit_compilation = True
        atmosphere.use_fast_functions = True

        freq, flux_nu, additional_outputs = atmosphere.calculate_flux(
            temperatures=temperature,
            mass_fractions=mass_fractions,
            mean_molar_masses=mass_fractions["MMW"],
            reference_gravity=10.0 ** log_g,
            frequencies_to_wavelengths=False,
            return_contribution=return_contribution,
            fast_opacity_interpolation=not return_contribution,
        )

        wave_cm = cst.c / freq
        wave_um = wave_cm * 1e4
        flux = flux_nu * cst.c / wave_cm ** 2
        flux *= 1e-7

        if radius is not None and radius > 0.0:
            flux *= ((radius * cst.r_jup) / (distance * cst.pc)) ** 2

        if return_contribution and additional_outputs is not None:
            if "emission_contribution" in additional_outputs:
                additional_outputs["emission_contribution"] = self._process_emission_contribution(
                    atmosphere,
                    additional_outputs["emission_contribution"],
                    wave_um,
                    wave_data,
                    vsini=vsini,
                    epsilon_limb=epsilon_limb,
                    rv=rv,
                )

        spec = self._broaden_and_rebin_to_data(
            wave_um,
            flux,
            wave_data,
            vsini=vsini,
            epsilon_limb=epsilon_limb,
            rv=rv,
        )
        return spec, additional_outputs, wave_um

    def rebin(self, new_wave, old_wave, old_flux):
        assert old_flux.ndim == 1
        assert old_wave.ndim == 1
        assert new_wave.ndim == 1
        if old_wave.size != old_flux.size:
            raise ValueError(
                f"Wavelength and flux size mismatch in rebin: "
                f"{old_wave.size} vs {old_flux.size}"
            )
        if old_wave[-1] < old_wave[0]:
            old_wave = old_wave[::-1]
            old_flux = old_flux[::-1]
        if HAS_SPECTRES:
            finite = np.isfinite(new_wave)
            new_wave_clean = np.where(finite, new_wave, np.nanmin(old_wave))
            new_wave_clean = np.clip(new_wave_clean, np.nanmin(old_wave), np.nanmax(old_wave))
            resampled = spectres(new_wave_clean, old_wave, old_flux, verbose=False)
            resampled = np.where(finite, resampled, np.nan)
            return resampled
        return np.interp(new_wave, old_wave, old_flux)

    def instrumental_broadening(self, wave_um, flux, instrumental_resolution=None):
        if instrumental_resolution is None:
            return flux
        sigma = (1.0 / instrumental_resolution) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        spacing = np.nanmean(2.0 * np.diff(wave_um) / (wave_um[1:] + wave_um[:-1]))
        sigma /= spacing
        return gaussian_filter1d(flux, sigma)

    def rotational_broadening(self, wave, flux, vsini, epsilon_limb=0.0):
        assert vsini > 0.0
        assert 0.0 <= epsilon_limb <= 1.0
        wave_even = np.linspace(wave.min(), wave.max(), wave.size)
        flux_even = np.interp(wave_even, wave, flux)
        if vsini > 0.1:
            flux_broad = pyasl.fastRotBroad(wave_even, flux_even, epsilon=epsilon_limb, vsini=vsini)
        else:
            flux_broad = flux_even
        return wave, np.interp(wave, wave_even, flux_broad)

    def run(self, nlive=100, resume=False, live_plot=False, **kwargs):
        self.return_spectrum = False
        outputfiles_basename = multinest_output_basename(self.save_path)

        sampling_efficiency = kwargs.get("sampling_efficiency", 0.05)
        const_efficiency_mode = kwargs.get("const_efficiency_mode", True)
        evidence_tolerance = kwargs.get("evidence_tolerance", 0.5)
        n_iter_before_update = kwargs.get("n_iter_before_update", 400)
        dump_callback = kwargs.pop("dump_callback", None)
        if live_plot and dump_callback is None:
            from .evaluation import make_multinest_callback

            dump_callback = make_multinest_callback(self)

        if dump_callback is not None and is_main_process():
            print(
                f"MultiNest live callback enabled; diagnostic plots every "
                f"{n_iter_before_update} iterations -> "
                f"{pathlib.Path(self.save_path or '.') / 'plots' / 'live'}"
            )

        pymultinest.run(
            LogLikelihood=self.log_likelihood,
            Prior=self.prior_transform,
            n_dims=self.ndim,
            outputfiles_basename=outputfiles_basename,
            resume=resume,
            verbose=is_main_process(),
            const_efficiency_mode=const_efficiency_mode,
            sampling_efficiency=sampling_efficiency,
            n_live_points=nlive,
            evidence_tolerance=evidence_tolerance,
            n_iter_before_update=n_iter_before_update,
            dump_callback=dump_callback,
        )

    def prior_check(self, n=3, random=False, save_dir=None):
        """Evaluate the model at prior edges and center, and save diagnostic plots."""
        from .evaluation import run_prior_check

        if save_dir is None and self.save_path is not None:
            save_dir = self.save_path / "plots"
        return run_prior_check(self, n=n, random=random, save_dir=save_dir)

    def plot_spectrum(self, params=None, fig_path=None, show=False):
        if params is None:
            params = self.prior_transform.params.copy()
        self.return_spectrum = True
        spectrum = self.get_spectrum(params)
        self.return_spectrum = False

        n = len(self.wave)
        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)
        for i, ax in enumerate(axes.ravel()):
            finite = np.isfinite(self.flux[i]) & np.isfinite(spectrum[i])
            ax.plot(self.wave[i], np.where(finite, self.flux[i], np.nan), "k", lw=0.7, label="data")
            ax.plot(self.wave[i], np.where(finite, spectrum[i], np.nan), "r", lw=0.7, label="model")
            ax.set_xlabel("Wavelength [um]")
            ax.set_ylabel("Flux")
            ax.legend()
        plt.tight_layout()
        if fig_path is not None:
            fig.savefig(fig_path)
        if show:
            plt.show()
        return fig

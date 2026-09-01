"""Posterior evaluation and live MultiNest diagnostic plots."""

import pathlib
import time

import corner
import matplotlib.pyplot as plt
import numpy as np


def _plot_name(filename, suffix=None):
    if suffix is None:
        return filename
    stem, ext = filename.rsplit(".", 1)
    return f"{stem}_{suffix}.{ext}"


def _species_plot_label(species_name: str) -> str:
    """Use the chemical formula part of a petitRADTRANS line-species name."""
    return str(species_name).split("__", 1)[0]


def _mask_model_to_data(model, flux, err=None):
    """Set model values to NaN wherever the data are invalid."""
    model = np.asarray(model, dtype=float).copy()
    invalid = ~np.isfinite(flux)
    if err is not None:
        invalid |= ~np.isfinite(err) | (err <= 0)
    model[invalid] = np.nan
    return model


def _order_spectrum_axes(n_orders, figsize_per_order=2.6):
    """Create stacked spec / residual / spacer axes following ramon conventions."""
    fig, axes = plt.subplots(
        n_orders * 3,
        1,
        figsize=(12, figsize_per_order * n_orders),
        squeeze=False,
        gridspec_kw={"height_ratios": [3, 1, 0.4] * n_orders, "hspace": 0.08},
    )
    axes_flat = axes.ravel()
    axes_spec = axes_flat[::3]
    axes_residuals = axes_flat[1::3]
    axes_empty = axes_flat[2::3]
    for ax in axes_empty:
        ax.set_visible(False)
        ax.axis("off")
    return fig, axes_spec, axes_residuals


def _physical_posterior_samples(posterior, prior_transform=None):
    """
    Extract physical parameter samples from a PyMultiNest dump_callback posterior array.

    The posterior rows are ``[params..., logL, aux]``. If the parameter block looks
    like a unit cube, apply ``prior_transform`` row by row.
    """
    posterior = np.asarray(posterior)
    if posterior.ndim == 1:
        posterior = posterior[np.newaxis, :]

    samples = posterior[:, :-2]
    log_likelihood = posterior[:, -2]

    if prior_transform is not None and np.all(np.nanmedian(samples, axis=0) < 1.0):
        physical = np.zeros_like(samples)
        for i, cube in enumerate(samples):
            cube_copy = cube.copy()
            prior_transform(cube_copy)
            physical[i] = cube_copy
        samples = physical

    return samples, log_likelihood


def _multinest_post_equal_weights_path(run_path):
    """Return the equal-weighted posterior chain file for a MultiNest run."""
    run_path = pathlib.Path(run_path)
    candidates = [
        run_path / "pmn" / "post_equal_weights.dat",
        run_path / "pmnpost_equal_weights.dat",
    ]
    for chain_file in candidates:
        if chain_file.is_file():
            return chain_file
    raise FileNotFoundError(
        "MultiNest output not found. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def load_multinest_samples(run_path, prior_transform=None):
    """
    Load equal-weighted posterior samples from a MultiNest run.

    Returns
    -------
    samples : np.ndarray
        Shape (n_samples, n_params) in physical parameter space.
    log_likelihood : np.ndarray
        Log-likelihood values for each sample.
    """
    run_path = pathlib.Path(run_path)
    chain_file = _multinest_post_equal_weights_path(run_path)

    data = np.loadtxt(chain_file)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    samples = data[:, :-1]
    log_likelihood = data[:, -1]

    return samples, log_likelihood


def set_free_parameters(retrieval, params_array):
    """Update a retrieval object with a vector of free parameters."""
    for i, name in enumerate(retrieval.parameter_names):
        retrieval.prior_transform.params[name] = params_array[i]
    retrieval.prior_transform.get_derived_params()
    return retrieval.prior_transform.params.copy()


def evaluate_model(retrieval, params_array, return_contribution=False):
    """
    Evaluate the forward model for a parameter vector.

    Returns
    -------
    spectrum : list[np.ndarray]
        Spline-marginalized model spectrum used in the likelihood.
    additional_outputs : list[dict]
    params : dict
    """
    params = set_free_parameters(retrieval, params_array)
    params["return_contribution"] = return_contribution

    raw_spectrum, additional_outputs = retrieval.compute_spectrum(params)
    spectrum = retrieval.apply_spline_to_spectrum(params, raw_spectrum)

    return spectrum, additional_outputs, params


def median_parameters(samples):
    """Posterior medians for each parameter."""
    return np.median(samples, axis=0)


def bestfit_parameters(samples, log_likelihood=None):
    """Best-fit parameters from maximum likelihood or posterior median."""
    if log_likelihood is not None and len(log_likelihood) == len(samples):
        return samples[np.nanargmax(log_likelihood)]
    return median_parameters(samples)


def spectral_integration(wave, flux, array):
    """Flux-weighted average of `array` along the wavelength axis."""
    flux = np.nan_to_num(flux, nan=0.0)
    array = np.asarray(array)
    numerator = np.trapz(flux * array, wave, axis=-1)
    denominator = np.trapz(flux, wave, axis=-1)
    if np.ndim(denominator) == 0:
        if denominator == 0:
            return np.zeros_like(numerator)
        return numerator / denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, 0.0)


def compute_pt_vmr_envelopes(retrieval, samples, max_samples=200, quantiles=(0.16, 0.5, 0.84)):
    """
    Compute temperature and VMR quantile envelopes from posterior samples.
    """
    if len(samples) > max_samples:
        idx = np.linspace(0, len(samples) - 1, max_samples, dtype=int)
        samples = samples[idx]

    pressure = retrieval.pressure
    species = list(retrieval.chem.line_species)
    n_samples = len(samples)
    n_layers = len(pressure)

    temperature_samples = np.zeros((n_samples, n_layers))
    vmr_samples = np.zeros((n_samples, len(species), n_layers))

    for i, sample in enumerate(samples):
        params = set_free_parameters(retrieval, sample)
        temperature = retrieval.PT(params)
        vmrs = retrieval.chem(params, temperature=temperature)
        temperature_samples[i] = temperature
        for j, species_name in enumerate(species):
            vmr_samples[i, j] = vmrs.get(species_name, np.zeros(n_layers))

    q = np.quantile(temperature_samples, quantiles, axis=0)
    vmr_q = np.quantile(vmr_samples, quantiles, axis=0)
    return q, vmr_q, species


def plot_corner(samples, param_names, save_path=None, show=False, title=None):
    """Corner plot of posterior samples."""
    plot_samples = samples[-min(len(samples), 2000):]
    fig = corner.corner(
        plot_samples,
        labels=param_names,
        show_titles=True,
        quiet=True,
        bins=20,
        max_n_ticks=3,
        quantiles=[0.16, 0.84],
        title_quantiles=[0.16, 0.5, 0.84],
        title_fmt=".3f",
        color="black",
        linewidths=0.5,
        hist_kwargs={"color": "black", "alpha": 0.35, "fill": True, "lw": 0.5},
        fill_contours=False,
        plot_datapoints=True,
    )
    if title is not None:
        fig.suptitle(title, fontsize=12)
    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_bestfit_spectrum(
    retrieval,
    params_array,
    save_path=None,
    show=False,
    title_prefix="Best-fit spectrum",
    spectrum=None,
):
    """Plot data, model, and residuals in a single stacked column per order."""
    if spectrum is None:
        spectrum, _, _ = evaluate_model(retrieval, params_array, return_contribution=False)

    n_orders = len(retrieval.wave)
    fig, axes_spec, axes_residuals = _order_spectrum_axes(n_orders)

    for i in range(n_orders):
        wave = np.asarray(retrieval.wave[i], dtype=float)
        flux = np.asarray(retrieval.flux[i], dtype=float)
        err = np.asarray(retrieval.err[i], dtype=float)
        model = _mask_model_to_data(spectrum[i], flux, err=err)

        finite = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(model)
        if not np.any(finite):
            continue

        wave_plot = wave
        flux_plot = np.where(finite, flux, np.nan)
        err_plot = np.where(finite, err, np.nan)
        model_plot = np.where(finite, model, np.nan)
        residuals = flux_plot - model_plot

        wave_min, wave_max = np.nanmin(wave_plot), np.nanmax(wave_plot)
        pad = 0.01 * (wave_max - wave_min)
        xlim = (wave_min - pad, wave_max + pad)

        ax_spec = axes_spec[i]
        ax_res = axes_residuals[i]
        ax_res.sharex(ax_spec)

        # ax_spec.plot(wave_plot, flux_plot, "k.", ms=1.5, label="data")
        # ax_spec.fill_between(
        #     wave_plot,
        #     flux_plot - err_plot,
        #     flux_plot + err_plot,
        #     color="k",
        #     alpha=0.15,
        #     lw=0,
        # )
        ax_spec.errorbar(wave_plot, flux_plot, yerr=err_plot, marker=".", ms=0.4, label="data",
                         capsize=0.5, capthick=0.0, elinewidth=0.2, color='k', alpha=0.4, zorder=0)
        
        ax_spec.plot(wave_plot, model_plot, color="orangered", lw=0.8, label="model")
        ax_spec.set_ylabel("Flux")
        ax_spec.set_xlim(xlim)
        ax_spec.grid(True, alpha=0.3)
        if i == 0:
            ax_spec.legend(loc="upper right", fontsize=8)
            ax_spec.set_title(title_prefix)

        ax_res.axhline(0, color="orangered", lw=0.8)
        ax_res.plot(
            wave_plot,
            residuals,
            color="orangered",
            marker=".",
            ms=1.5,
            ls="none",
        )
        ax_res.fill_between(wave_plot, -err_plot, err_plot, color="k", alpha=0.15, lw=0)
        ax_res.set_ylabel("Residual")
        ax_res.set_xlim(xlim)
        ax_res.grid(True, alpha=0.3)
        if i == n_orders - 1:
            ax_res.set_xlabel("Wavelength [um]")
        else:
            ax_spec.tick_params(labelbottom=False)
            ax_res.tick_params(labelbottom=True)

    fig.subplots_adjust(hspace=0.08)
    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_pt_composition(
    retrieval,
    params_array=None,
    samples=None,
    integrated_ec=None,
    save_path=None,
    show=False,
    max_species=6,
    max_envelope_samples=200,
):
    """Two-panel PT profile and VMR plot."""
    pressure = retrieval.pressure

    if samples is not None and len(samples) > 1:
        temperature_envelopes, vmr_envelopes, species = compute_pt_vmr_envelopes(
            retrieval, samples, max_samples=max_envelope_samples
        )
        q_median = len(temperature_envelopes) // 2
        temperature = temperature_envelopes[q_median]
    else:
        if params_array is None:
            raise ValueError("Provide params_array or samples.")
        params = set_free_parameters(retrieval, params_array)
        temperature = retrieval.PT(params)
        vmrs = retrieval.chem(params, temperature=temperature)
        species = list(retrieval.chem.line_species)
        vmr_envelopes = None
        temperature_envelopes = None
        q_median = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    if temperature_envelopes is not None:
        for j in range(temperature_envelopes.shape[0]):
            axes[0].fill_betweenx(
                pressure,
                temperature_envelopes[j],
                temperature_envelopes[-j - 1],
                color="purple",
                alpha=0.2,
                lw=0,
            )
    axes[0].plot(temperature, pressure, color="purple", lw=1.5, label="T(P)")
    if integrated_ec is not None:
        ax_ec = axes[0].twiny()
        iec = np.asarray(integrated_ec)
        ax_ec.plot(iec, pressure, color="k", alpha=0.6, label="Emission contribution")
        ax_ec.fill_betweenx(pressure, iec, 0.0, color="k", alpha=0.15)
        ax_ec.set_xlim(0, 3.0 * np.nanmax(np.nan_to_num(iec, nan=0.0)))
        ax_ec.set_xticks([])
    axes[0].set(yscale="log", xlabel="Temperature [K]", ylabel="Pressure [bar]")
    axes[0].set_ylim(pressure.max(), pressure.min())
    axes[0].set_title("Pressure-temperature profile")
    axes[0].legend(loc="best", fontsize=8)

    if vmr_envelopes is not None:
        abundances = np.median(vmr_envelopes[q_median], axis=-1)
        order = np.argsort(abundances)[::-1]
        for plot_idx, species_idx in enumerate(order[:max_species]):
            species_name = species[species_idx]
            color = f"C{plot_idx}"
            for j in range(vmr_envelopes.shape[0]):
                axes[1].fill_betweenx(
                    pressure,
                    vmr_envelopes[j, species_idx],
                    vmr_envelopes[-j - 1, species_idx],
                    color=color,
                    alpha=0.2,
                    lw=0,
                )
            axes[1].plot(
                vmr_envelopes[q_median, species_idx],
                pressure,
                color=color,
                lw=1.2,
                label=_species_plot_label(species_name),
            )
    else:
        params = set_free_parameters(retrieval, params_array)
        vmrs = retrieval.chem(params, temperature=temperature)
        abundances = [np.median(vmrs.get(s, np.zeros_like(pressure))) for s in species]
        order = np.argsort(abundances)[::-1]
        for plot_idx, species_idx in enumerate(order[:max_species]):
            species_name = species[species_idx]
            axes[1].plot(
                vmrs[species_name],
                pressure,
                lw=1.2,
                label=_species_plot_label(species_name),
                color=f"C{plot_idx}",
            )

    axes[1].set(xscale="log", yscale="log", xlabel="Volume mixing ratio", ylabel="Pressure [bar]")
    axes[1].set_xlim(1e-12, 1e-1)
    axes[1].set_ylim(pressure.max(), pressure.min())
    axes[1].set_title("Atmospheric composition")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    plt.tight_layout()
    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def prepare_emission_contribution_for_plot(
    wave,
    pressure,
    emission_contribution,
    flux=None,
    ec_threshold=0.01,
    pressure_margin=0.08,
):
    """
    Mask invalid pixels and derive the pressure range spanned by the spectrum.

    Non-finite wavelength, flux, and emission-contribution values are set to
    NaN so downstream plotting does not interpolate across gaps.

    Returns
    -------
    wave_plot, ec_plot, pressure_limits, formation_pressure
    """
    wave = np.asarray(wave, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    ec = np.asarray(emission_contribution, dtype=float)

    wave = np.where(np.isfinite(wave), wave, np.nan)
    ec = np.where(np.isfinite(ec), ec, np.nan)

    if flux is not None:
        flux = np.asarray(flux, dtype=float)
        if flux.shape != wave.shape:
            raise ValueError(
                f"flux shape {flux.shape} must match wavelength shape {wave.shape}"
            )
        valid_data = np.isfinite(flux)
        ec[:, ~valid_data] = np.nan

    finite_wave = np.isfinite(wave)
    if np.any(finite_wave):
        sort = np.argsort(wave, kind="mergesort")
        sort = sort[finite_wave[sort]]
        wave = wave[sort]
        ec = ec[:, sort]
        finite_wave = np.isfinite(wave)

    ec_plot = ec.copy()
    ec_plot[:, ~finite_wave] = np.nan
    ec_plot[~np.isfinite(ec_plot)] = np.nan

    layer_max = np.nanmax(ec_plot, axis=1)
    peak = np.nanmax(layer_max)
    if not np.isfinite(peak) or peak <= 0:
        p_lo, p_hi = pressure.min(), pressure.max()
    else:
        active = layer_max >= ec_threshold * peak
        if not np.any(active):
            active = layer_max > 0
        p_active = pressure[active]
        log_p_lo = np.log10(p_active.min())
        log_p_hi = np.log10(p_active.max())
        log_span = max(log_p_hi - log_p_lo, 0.05)
        p_lo = 10 ** (log_p_lo - pressure_margin * log_span)
        p_hi = 10 ** (log_p_hi + pressure_margin * log_span)

    ec_sum = np.nansum(ec_plot, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        formation_pressure = np.where(
            ec_sum > 0,
            np.nansum(pressure[:, None] * ec_plot, axis=0) / ec_sum,
            np.nan,
        )

    return wave, ec_plot, (p_lo, p_hi), formation_pressure


def plot_emission_contribution(
    wave,
    pressure,
    emission_contribution,
    flux=None,
    save_path=None,
    show=False,
    title=None,
    ec_threshold=0.01,
):
    """Pressure-wavelength emission contribution map over the active pressure range."""
    if wave is None or emission_contribution is None:
        return None

    wave, ec, (p_lo, p_hi), p_form = prepare_emission_contribution_for_plot(
        wave,
        pressure,
        emission_contribution,
        flux=flux,
        ec_threshold=ec_threshold,
    )
    pressure = np.asarray(pressure)

    if ec.ndim != 2 or ec.shape != (len(pressure), len(wave)):
        raise ValueError(
            f"emission_contribution must have shape ({len(pressure)}, {len(wave)}), got {ec.shape}"
        )

    ec_peak = np.nanmax(ec)
    if not np.isfinite(ec_peak) or ec_peak <= 0:
        return None

    ec_threshold_value = ec_threshold * ec_peak
    ec_masked = np.ma.masked_where(
        ~np.isfinite(ec) | (ec < ec_threshold_value),
        ec,
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x, y = np.meshgrid(wave, pressure)
    mesh = ax.pcolormesh(x, y, ec_masked, shading="auto", cmap="bone_r")
    fig.colorbar(mesh, ax=ax, pad=0.01, label="Emission contribution")

    finite = np.isfinite(wave) & np.isfinite(p_form)
    if np.any(finite):
        ax.plot(
            wave[finite],
            p_form[finite],
            color="orangered",
            lw=1.2,
            label="Formation pressure",
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set(yscale="log", xlabel="Wavelength [um]", ylabel="Pressure [bar]")
    ax.set_ylim(p_hi, p_lo)
    ax.set_title(title or "Emission contribution function")

    if save_path is not None:
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def evaluate_posterior(
    retrieval,
    run_path=None,
    max_samples=2000,
    max_envelope_samples=200,
    show=False,
    suffix=None,
):
    """
    Load a finished MultiNest run and write standard diagnostic plots.
    """
    run_path = pathlib.Path(run_path or retrieval.save_path)
    plots_dir = run_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    samples, log_likelihood = load_multinest_samples(run_path, retrieval.prior_transform)
    if len(samples) > max_samples:
        samples = samples[-max_samples:]
        log_likelihood = log_likelihood[-max_samples:]

    params_best = bestfit_parameters(samples, log_likelihood)
    spectrum, additional_outputs, _ = evaluate_model(
        retrieval, params_best, return_contribution=True
    )

    plot_corner(
        samples,
        retrieval.parameter_names,
        save_path=plots_dir / _plot_name("corner.pdf", suffix),
        show=show,
        title="Posterior",
    )
    plot_bestfit_spectrum(
        retrieval,
        params_best,
        save_path=plots_dir / _plot_name("bestfit_spectrum.pdf", suffix),
        show=show,
        spectrum=spectrum,
    )

    integrated_ec = None
    wave_merged = []
    ec_merged = []
    flux_merged = []
    for i, wave_i in enumerate(retrieval.wave):
        ec_i = additional_outputs[i].get("emission_contribution")
        if ec_i is not None:
            wave_i = np.asarray(wave_i, dtype=float)
            ec_i = np.asarray(ec_i, dtype=float)
            flux_i = np.asarray(retrieval.flux[i], dtype=float)
            valid = np.isfinite(wave_i) & np.isfinite(flux_i)
            ec_i = np.where(np.isfinite(ec_i), ec_i, np.nan)
            ec_i[:, ~valid] = np.nan
            wave_merged.append(np.where(valid, wave_i, np.nan))
            ec_merged.append(ec_i)
            flux_merged.append(flux_i)

    if ec_merged:
        wave_all = np.concatenate(wave_merged)
        ec_all = np.hstack(ec_merged)
        flux_all = np.concatenate(flux_merged)
        ec_all[~np.isfinite(ec_all)] = np.nan
        plot_emission_contribution(
            wave_all,
            retrieval.pressure,
            ec_all,
            flux=flux_all,
            save_path=plots_dir / _plot_name("emission_contribution.pdf", suffix),
            show=show,
        )
        valid = np.isfinite(wave_all) & np.isfinite(flux_all)
        integrated_ec = spectral_integration(
            wave_all[valid], flux_all[valid], ec_all[:, valid]
        )

    plot_pt_composition(
        retrieval,
        samples=samples,
        integrated_ec=integrated_ec,
        save_path=plots_dir / _plot_name("pt_composition.pdf", suffix),
        show=show,
        max_species=6,
        max_envelope_samples=max_envelope_samples,
    )

    return {
        "samples": samples,
        "log_likelihood": log_likelihood,
        "bestfit_params": params_best,
        "spectrum": spectrum,
        "additional_outputs": additional_outputs,
    }


def run_prior_check(retrieval, n=3, random=False, save_dir=None):
    """
    Evaluate the model at the edges and center of the prior hypercube.

    Saves combined spectrum/residual and PT/VMR diagnostic plots.
    """
    if save_dir is None:
        save_dir = pathlib.Path("plots")
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if random:
        cube_fractions = np.random.rand(n)
    else:
        cube_fractions = np.linspace(0.0, 1.0, n)

    n_orders = len(retrieval.wave)
    fig_spec, axes_spec, axes_residuals = _order_spectrum_axes(n_orders, figsize_per_order=3.0)
    fig_pt, axes_pt = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for i in range(n_orders):
        if retrieval.wave[i] is not None and retrieval.flux[i] is not None:
            axes_spec[i].plot(
                retrieval.wave[i],
                retrieval.flux[i],
                color="k",
                lw=0.8,
                label="data",
            )

    timings = []
    for j, fraction in enumerate(cube_fractions):
        cube = np.ones(retrieval.ndim, dtype=float) * fraction
        retrieval.prior_transform(cube)
        params = retrieval.prior_transform.params.copy()
        param_subset = {k: params[k] for k in retrieval.parameter_names}
        print(f"prior check {j + 1}/{n}: fraction={fraction:.3f}, params={param_subset}")

        start = time.perf_counter()
        try:
            params_array = np.array([params[name] for name in retrieval.parameter_names])
            spectrum, _, _ = evaluate_model(retrieval, params_array)
            lnL = retrieval.log_likelihood(cube.copy())
        except Exception as exc:
            print(f"  model evaluation failed: {exc}")
            continue
        timings.append((time.perf_counter() - start) * 1000.0)

        color = f"C{j}"
        for i in range(n_orders):
            wave_i = np.asarray(retrieval.wave[i], dtype=float)
            flux_i = np.asarray(retrieval.flux[i], dtype=float)
            err_i = np.asarray(retrieval.err[i], dtype=float)
            model_i = _mask_model_to_data(spectrum[i], flux_i, err=err_i)
            finite = np.isfinite(flux_i) & np.isfinite(model_i)
            if not np.any(finite):
                continue
            axes_spec[i].plot(
                wave_i,
                np.where(finite, model_i, np.nan),
                color=color,
                lw=0.9,
                label=f"f={fraction:.2f}, lnL={lnL:.1f}",
            )
            axes_residuals[i].plot(
                wave_i,
                np.where(finite, flux_i - model_i, np.nan),
                color=color,
                lw=0.9,
            )

        temperature = retrieval.PT(params)
        vmrs = retrieval.chem(params, temperature=temperature)
        axes_pt[0].plot(temperature, retrieval.pressure, color=color, lw=1.2, label=f"f={fraction:.2f}")

        species = list(retrieval.chem.line_species)
        abundances = [np.median(vmrs.get(s, np.zeros_like(retrieval.pressure))) for s in species]
        order = np.argsort(abundances)[::-1][:6]
        for plot_idx, species_idx in enumerate(order):
            species_name = species[species_idx]
            axes_pt[1].plot(
                vmrs[species_name],
                retrieval.pressure,
                color=color,
                lw=1.0,
                alpha=0.8 if plot_idx > 0 else 1.0,
                label=_species_plot_label(species_name) if j == 0 else None,
            )

    if timings:
        print(
            f"prior check timing: mean={np.mean(timings):.1f} ms, "
            f"std={np.std(timings):.1f} ms over {len(timings)} calls"
        )

    axes_spec[0].set_ylabel("Flux")
    axes_spec[0].legend(loc="upper right", fontsize=8)
    axes_residuals[-1].set_xlabel("Wavelength [um]")
    axes_pt[0].set(yscale="log", xlabel="Temperature [K]", ylabel="Pressure [bar]")
    axes_pt[0].set_ylim(retrieval.pressure.max(), retrieval.pressure.min())
    axes_pt[0].legend(loc="best", fontsize=8)
    axes_pt[1].set(xscale="log", yscale="log", xlabel="Volume mixing ratio", ylabel="Pressure [bar]")
    axes_pt[1].set_xlim(1e-12, 1e-1)
    axes_pt[1].set_ylim(retrieval.pressure.max(), retrieval.pressure.min())
    axes_pt[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    random_label = "_random" if random else ""
    spec_path = save_dir / f"prior_check{random_label}.pdf"
    pt_path = save_dir / f"prior_check_pt{random_label}.pdf"
    fig_spec.subplots_adjust(hspace=0.08)
    fig_pt.tight_layout()
    fig_spec.savefig(spec_path, dpi=150, bbox_inches="tight")
    fig_pt.savefig(pt_path, dpi=150, bbox_inches="tight")
    print(f"Saved prior-check spectrum plot to {spec_path}")
    print(f"Saved prior-check PT/VMR plot to {pt_path}")
    plt.close(fig_spec)
    plt.close(fig_pt)

    return {
        "cube_fractions": cube_fractions,
        "timings_ms": timings,
        "spectrum_plot": spec_path,
        "pt_plot": pt_path,
    }


def make_multinest_callback(
    retrieval,
    max_posterior_samples=2000,
    save_dir=None,
):
    """
    Build a pymultinest dump_callback for live diagnostic plots.

    Produces, on each callback:
    - corner plot of current posterior samples
    - order-by-order best-fit spectrum
    - PT profile and VMR composition for the current best fit
    """
    save_dir = pathlib.Path(save_dir or retrieval.save_path or ".") / "plots" / "live"
    save_dir.mkdir(parents=True, exist_ok=True)
    state = {"call_count": 0}

    def _callback(
        n_samples,
        n_live,
        n_params,
        live_points,
        posterior,
        stats,
        max_ln_L,
        ln_Z,
        ln_Z_err,
        nullcontext,
    ):
        from .retrieval import is_main_process

        if not is_main_process():
            return
        if posterior is None or len(posterior) == 0:
            return

        plot_samples, log_l = _physical_posterior_samples(
            posterior, retrieval.prior_transform
        )
        if len(plot_samples) == 0:
            return

        state["call_count"] += 1
        tag = state["call_count"]

        best_params = plot_samples[np.nanargmax(log_l)]
        plot_samples = plot_samples[-min(len(plot_samples), max_posterior_samples) :]

        print(
            f"[live] samples={n_samples}, live={n_live}, max lnL={max_ln_L:.2f}, "
            f"logZ={ln_Z:.2f} +/- {ln_Z_err:.2f}"
        )

        plot_corner(
            plot_samples,
            retrieval.parameter_names,
            save_path=save_dir / f"corner_{tag:04d}.pdf",
            title=f"Posterior (n={n_samples})",
        )
        plot_bestfit_spectrum(
            retrieval,
            best_params,
            save_path=save_dir / f"bestfit_spectrum_{tag:04d}.pdf",
            title_prefix=f"Live best fit (n={n_samples})",
        )

        plot_pt_composition(
            retrieval,
            params_array=best_params,
            save_path=save_dir / f"pt_composition_{tag:04d}.pdf",
        )

    return _callback

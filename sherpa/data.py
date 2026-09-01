import pickle

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage import binary_dilation, gaussian_filter1d


def normalize_order_by_median(flux):
    """
    Normalize a single-order flux vector by its median.

    Uses the same scheme as ``Data.normalize_orders``.
    """
    flux = np.asarray(flux, dtype=float)
    scale = float(np.nanmedian(flux))
    if not (scale > 0 and np.isfinite(scale)):
        raise ValueError(f"Invalid flux median for normalization: {scale}")
    return flux / scale, scale


def save_retrieval_data_cache(path, wave, flux, err):
    """Persist processed spectroscopic data for fast MPI worker startup."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump({"wave": wave, "flux": flux, "err": err}, handle)


def load_retrieval_data_cache(path):
    """Load processed spectroscopic data written by ``save_retrieval_data_cache``."""
    path = Path(path)
    if not path.is_file():
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


class Data:
    """Minimal base class for spectroscopic data."""

    def __init__(self, target_name):
        self.target_name = target_name
        self.wave = None
        self.flux = None
        self.err = None
        self.wave_unit = "um"
        self.flux_unit = "erg/cm^2/s/nm"
        self.metadata = {}
        self._load_data()

    def _load_data(self):
        raise NotImplementedError

    def summary(self):
        if self.wave is None:
            print(f"Target: {self.target_name} - No data loaded")
            return
        n_orders = len(self.wave)
        wave_all = np.concatenate(self.wave)
        flux_all = np.concatenate(self.flux)
        print(f"Target: {self.target_name}")
        print(f"  Orders: {n_orders}")
        print(f"  Data points: {sum(len(w) for w in self.wave)}")
        print(f"  Wavelength range: {np.nanmin(wave_all):.3f} - {np.nanmax(wave_all):.3f} {self.wave_unit}")
        print(f"  Flux range: {np.nanmin(flux_all):.2e} - {np.nanmax(flux_all):.2e} {self.flux_unit}")
        print(f"  Median S/N: {self.get_median_snr():.1f}")

    def get_median_snr(self):
        snr = self.get_snr()
        if isinstance(snr, np.ndarray) and snr.dtype == object:
            values = np.concatenate([np.ravel(s) for s in snr if s is not None])
            return float(np.nanmedian(values))
        return float(np.nanmedian(snr))

    def get_snr(self):
        if self.flux is None or self.err is None:
            raise ValueError("No flux or error data loaded")
        if isinstance(self.flux, list):
            return np.array([
                np.divide(f, e, out=np.full_like(f, np.nan), where=e > 0)
                for f, e in zip(self.flux, self.err)
            ], dtype=object)
        return np.divide(self.flux, self.err, out=np.full_like(self.flux, np.nan), where=self.err > 0)

    def select_orders(self, orders):
        orders = np.atleast_1d(orders).astype(int)
        assert len(orders) > 0, "Orders list is empty"
        assert np.max(orders) < len(self.wave), f"Orders out of range"
        self.wave = [self.wave[i] for i in orders]
        self.flux = [self.flux[i] for i in orders]
        self.err = [self.err[i] for i in orders]
        if self.metadata.get("order_ids") is not None:
            self.metadata["order_ids"] = [self.metadata["order_ids"][i] for i in orders]
        if hasattr(self, "transmission") and self.transmission is not None:
            self.transmission = [self.transmission[i] for i in orders]
        return self

    def remove_empty_orders(self, n_minimum_pixels=100):
        keep = []
        for i, (wave_i, flux_i, err_i) in enumerate(zip(self.wave, self.flux, self.err)):
            mask = np.isfinite(wave_i) & np.isfinite(flux_i) & np.isfinite(err_i) & (err_i > 0)
            if int(np.count_nonzero(mask)) >= n_minimum_pixels:
                keep.append(i)
        assert len(keep) > 0, f"No orders remain with >= {n_minimum_pixels} valid pixels"
        self.wave = [self.wave[i] for i in keep]
        self.flux = [self.flux[i] for i in keep]
        self.err = [self.err[i] for i in keep]
        if hasattr(self, "transmission") and self.transmission is not None:
            self.transmission = [self.transmission[i] for i in keep]
        return self

    def normalize_orders(self, store_scales=True):
        scales = []
        for i, (flux_i, err_i) in enumerate(zip(self.flux, self.err)):
            flux_i, scale = normalize_order_by_median(flux_i)
            self.flux[i] = flux_i
            self.err[i] = err_i / scale
            scales.append(scale)
        if store_scales:
            self.metadata["normalization_scales"] = scales
        return self

    def apply_telluric_mask(self, transmission=None, threshold=0.60, grow_mask=3):
        """Set flux and error to NaN where telluric transmission is below threshold."""
        if transmission is None:
            print("No transmission array provided; skipping telluric mask")
            return self

        if isinstance(transmission, np.ndarray) and transmission.ndim == 1:
            idx = 0
            transmission = [transmission[idx:idx + len(self.wave[i])] for i in range(len(self.wave))]

        for i, (flux_i, err_i, tel_i) in enumerate(zip(self.flux, self.err, transmission)):
            deep = self.telluric_mask(tel_i, threshold=threshold, grow_mask=grow_mask)
            self.flux[i] = np.where(deep, np.nan, flux_i)
            self.err[i] = np.where(deep, np.nan, err_i)
        return self

    @staticmethod
    def telluric_mask(transmission, threshold=0.60, grow_mask=3):
        """True where telluric transmission is below ``threshold``."""
        deep = np.asarray(transmission, dtype=float) < threshold
        deep |= ~np.isfinite(transmission)
        if grow_mask > 0:
            struct = np.ones(2 * grow_mask + 1, dtype=bool)
            deep = binary_dilation(deep, structure=struct)
        return deep

    def plot(self, fig_path=None, normalize=True, show=False):
        n_orders = len(self.wave)
        order_ids = self.metadata.get("order_ids", list(range(n_orders)))
        fig, axes = plt.subplots(n_orders, 1, figsize=(11, 3.2 * n_orders), squeeze=False)
        axes_flat = axes.ravel()
        for i, (wave_i, flux_i, err_i) in enumerate(zip(self.wave, self.flux, self.err)):
            ax = axes_flat[i]
            f_plot = flux_i.copy()
            e_plot = err_i.copy()
            if normalize:
                finite = np.isfinite(f_plot) & (f_plot > 0)
                if np.any(finite):
                    scale = float(np.nanpercentile(f_plot[finite], 90.0))
                    f_plot = f_plot / scale
                    e_plot = e_plot / scale
                y_label = "Normalized flux"
            else:
                y_label = f"Flux [{self.flux_unit}]"
            ax.plot(wave_i, f_plot, color="0.15", lw=0.7)
            finite_err = np.isfinite(f_plot) & np.isfinite(e_plot)
            if np.any(finite_err):
                ax.fill_between(wave_i, f_plot - e_plot, f_plot + e_plot, color="0.15", alpha=0.15, linewidth=0)
            ax.set_ylabel(y_label)
            ax.set_xlabel("Wavelength [um]")
            ax.set_title(f"Order {order_ids[i]}")
        plt.tight_layout()
        if fig_path is not None:
            fig.savefig(fig_path)
            print(f"Saved plot to {fig_path}")
        if show:
            plt.show()
        return fig


class IshellMband(Data):
    """Load an iSHELL M-band merged spectrum from a Spextool FITS file."""

    ROW_WAVE, ROW_FLUX, ROW_ERR, ROW_FLAG = 0, 1, 2, 3

    def __init__(self, target_name="Ishell", file_path=None):
        if file_path is None:
            default = Path(__file__).resolve().parent.parent / "data" / target_name.lower()
            file_path = default / f"{target_name.lower()}_m1_tellcor_merged.fits"
        self.file_path = Path(file_path)
        self.header = None
        self.transmission = None
        super().__init__(target_name)

    @staticmethod
    def load_telluric_template(file_path):
        """
        Load a telluric transmission template saved by ``save_telluric_template``.

        FITS format: row 0 = wavelength [nm], row 1 = telluric transmission.
        """
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Telluric template not found: {file_path}")

        with fits.open(file_path) as hdul:
            header = hdul[0].header.copy()
            data = np.asarray(hdul[0].data, dtype=float)

        if data.ndim != 2 or data.shape[0] != 2:
            raise ValueError(f"Expected telluric template shape (2, n), got {data.shape}")

        metadata = {
            "file_path": str(file_path),
            "wlen_id": header.get("SKYWLENID", "M"),
            "observatory": header.get("SKYOBS", "paranal"),
            "airmass": float(header.get("SKYAIRMAS", 1.5)),
            "pwv": float(header.get("SKYPWV", 2.5)),
            "wres": float(header.get("SKYWRES", 500_000.0)),
            "resolution": float(header.get("RP", 70_000.0)),
        }
        if "TELTHRSH" in header:
            metadata["telluric_threshold"] = float(header["TELTHRSH"])
        if "TELGROW" in header:
            metadata["telluric_grow_mask"] = int(header["TELGROW"])

        return {
            "wavelength_nm": data[0],
            "transmission": data[1],
            "metadata": metadata,
        }

    @staticmethod
    def resample_telluric_template(template, wave_nm, new_resolution=None, kind="linear"):
        """Resample a native telluric template onto an observed wavelength grid [nm]."""
        wavelength = np.asarray(template["wavelength_nm"], dtype=float)
        transmission = np.asarray(template["transmission"], dtype=float)
        metadata = template.get("metadata", {})

        if new_resolution is not None:
            original_resolution = float(metadata.get("wres", 500_000.0))
            if new_resolution < original_resolution:
                mean_wavelength = float(np.mean(wavelength))
                target_fwhm = mean_wavelength / new_resolution
                original_fwhm = mean_wavelength / original_resolution
                target_sigma = target_fwhm / (2 * np.sqrt(2 * np.log(2)))
                original_sigma = original_fwhm / (2 * np.sqrt(2 * np.log(2)))
                if target_sigma > original_sigma:
                    additional_sigma = np.sqrt(target_sigma ** 2 - original_sigma ** 2)
                    pixel_spacing = float(np.mean(np.diff(wavelength)))
                    sigma_pixels = additional_sigma / pixel_spacing
                    transmission = gaussian_filter1d(transmission, sigma=sigma_pixels)

        interp_func = interp1d(
            wavelength, transmission, kind=kind, bounds_error=False, fill_value=1.0
        )
        return np.asarray(interp_func(np.asarray(wave_nm, dtype=float)), dtype=float)

    def apply_telluric_mask(self, telluric_template_path, threshold=0.60, grow_mask=3, quiet=False):
        """
        Mask deep telluric pixels using a precomputed SkyCalc template stored in FITS.

        Parameters
        ----------
        telluric_template_path : str or Path
            FITS file written by ``save_telluric_template`` (row 0: wavelength [nm],
            row 1: transmission).
        threshold : float
            Pixels with transmission below this value are masked.
        grow_mask : int
            Dilation radius [pixels] applied to the telluric mask.
        """
        template = self.load_telluric_template(telluric_template_path)
        resolution = float(self.metadata.get("resolution", 70_000.0))

        self.transmission = []
        for wave_i in self.wave:
            wave_nm = np.asarray(wave_i, dtype=float) * 1000.0
            tel_i = self.resample_telluric_template(
                template, wave_nm, new_resolution=resolution
            )
            self.transmission.append(tel_i)

        order_ids = self.metadata.get(
            "selected_order_ids",
            self.metadata.get("order_ids", list(range(len(self.wave)))),
        )
        n_masked_total = 0
        n_pix_total = 0
        for i, (flux_i, err_i, tel_i) in enumerate(zip(self.flux, self.err, self.transmission)):
            mask = self.telluric_mask(tel_i, threshold=threshold, grow_mask=grow_mask)
            self.flux[i] = np.where(mask, np.nan, flux_i)
            self.err[i] = np.where(mask, np.nan, err_i)
            n_masked = int(np.count_nonzero(mask))
            n_masked_total += n_masked
            n_pix_total += len(mask)
            if not quiet:
                print(
                    f"Order {order_ids[i]}: telluric mask {n_masked}/{len(mask)} pixels "
                    f"({100.0 * n_masked / max(len(mask), 1):.1f}%)"
                )

        self.metadata["telluric_template"] = str(Path(telluric_template_path))
        self.metadata["telluric_threshold"] = threshold
        self.metadata["telluric_grow_mask"] = grow_mask
        if not quiet:
            print(
                f"Telluric mask total: {n_masked_total}/{n_pix_total} pixels "
                f"({100.0 * n_masked_total / max(n_pix_total, 1):.1f}%)"
            )
        return self

    @staticmethod
    def split_merged_orders(wave, gap_factor=10.0):
        if wave.size < 2:
            return [(0, wave.size)]
        dw = np.diff(wave)
        med_dw = float(np.median(dw))
        if not np.isfinite(med_dw) or med_dw <= 0:
            raise ValueError("Cannot determine median wavelength spacing")
        gap_idx = np.where(dw > gap_factor * med_dw)[0]
        bounds = np.concatenate([[0], gap_idx + 1, [wave.size]])
        return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]

    def _load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"FITS file not found: {self.file_path}")

        with fits.open(self.file_path) as hdul:
            self.header = hdul[0].header.copy()
            data = np.asarray(hdul[0].data, dtype=float)

        if data.ndim != 2 or data.shape[0] != 4:
            raise ValueError(f"Expected shape (4, n), got {data.shape}")

        wave_1d = data[self.ROW_WAVE]
        flux_1d = data[self.ROW_FLUX]
        err_1d = data[self.ROW_ERR]
        flag_1d = data[self.ROW_FLAG]

        order_slices = self.split_merged_orders(wave_1d)
        order_ids = []
        for key in self.header:
            if key.startswith(("DISPO", "XROR")) and key[-3:].isdigit():
                order_ids.append(int(key[-3:]))
        order_ids = sorted(set(order_ids))
        if len(order_ids) != len(order_slices):
            order_ids = list(range(len(order_slices)))

        wave_list, flux_list, err_list = [], [], []
        for s, e in order_slices:
            wave_i = wave_1d[s:e].copy()
            flux_i = flux_1d[s:e].copy()
            err_i = err_1d[s:e].copy()
            flag_i = flag_1d[s:e].copy()

            bad = ~np.isfinite(wave_i) | ~np.isfinite(flux_i) | ~np.isfinite(err_i)
            bad |= flag_i != 0
            bad |= err_i <= 0
            flux_i[bad] = np.nan
            err_i[bad] = np.nan

            wave_list.append(wave_i)
            flux_list.append(flux_i)
            err_list.append(err_i)

        self.wave = wave_list
        self.flux = flux_list
        self.err = err_list
        self.wave_unit = self.header.get("XUNITS", "um")
        self.flux_unit = "erg/cm^2/s/nm"
        self.metadata = {
            "file_path": str(self.file_path),
            "data_type": "iSHELL M1",
            "instrument": self.header.get("INSTR", "iSHELL"),
            "order_ids": order_ids,
            "n_orders": len(order_ids),
            "resolution": float(self.header.get("RP", 70_000)),
        }

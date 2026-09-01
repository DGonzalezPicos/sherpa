import numpy as np
from scipy.interpolate import interp1d


class PressureTemperature:
    """Base pressure-temperature profile class."""

    def __init__(self, pressure, **kwargs):
        self.pressure = pressure
        self.temperature = None
        self.mode = kwargs.get("mode", None)

    def __call__(self, params):
        return np.zeros_like(self.pressure)

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        kwargs["lw"] = kwargs.get("lw", 2)
        kwargs["color"] = kwargs.get("color", "brown")
        ax.plot(self.temperature, self.pressure, **kwargs)
        ax.set_ylim(self.pressure.max(), self.pressure.min())
        ax.set(yscale="log", xlabel="Temperature [K]", ylabel="Pressure [bar]")
        return ax


class PressureTemperatureGradients(PressureTemperature):
    """
    Temperature profile for a radiative-convective equilibrium atmosphere with one
    convective region.
    """

    def __init__(self, pressure, PT_interp_mode="linear", **kwargs):
        super().__init__(pressure)
        self.flipped_ln_pressure = np.log(self.pressure)[::-1]
        self.log10_pressure = np.log10(self.pressure)
        self.PT_interp_mode = PT_interp_mode

    def __call__(self, params):
        assert "log_P_RCE" in params, "RCE profile requires log_P_RCE parameter"

        n_pressure_levels = params["n_pressure_levels"]
        T_grads = [params[f"T_grad_{i}"] for i in range(n_pressure_levels)]

        if "log_P_knots" in params:
            assert len(params["log_P_knots"]) == n_pressure_levels
            self.log_P_knots = params["log_P_knots"]
        else:
            if "dlog_P_1" in params:
                dlog_P = [params["dlog_P_1"], params["dlog_P_3"]]
            else:
                dlog_P = [params["dlog_P"], params["dlog_P"]]

            self.log_P_knots = np.ones(n_pressure_levels)
            self.log_P_knots[0] = self.log10_pressure.max()
            self.log_P_knots[-1] = self.log10_pressure.min()

            x = 1.0
            if n_pressure_levels == 7:
                self.log_P_knots[2] = min(params["log_P_RCE"] + dlog_P[0], self.log_P_knots[0] * 0.8)
                self.log_P_knots[4] = max(params["log_P_RCE"] - dlog_P[1], self.log_P_knots[-1] * 0.8)
                x = 2.0

            self.log_P_knots[1] = min(params["log_P_RCE"] + x * dlog_P[0], self.log10_pressure.max() * 0.9)
            self.log_P_knots[-2] = max(params["log_P_RCE"] - x * dlog_P[1], self.log10_pressure.min() * 0.9)
            self.log_P_knots[n_pressure_levels // 2] = params["log_P_RCE"]

        interp_func = interp1d(
            self.log_P_knots[::-1],
            T_grads[::-1],
            kind=self.PT_interp_mode,
        )
        dlnT_dlnP_array = interp_func(self.log10_pressure)[::-1]
        dlnT_dlnP_array[dlnT_dlnP_array < 0.0] = 0.0

        if "T_RCE" in params:
            ln_p_ref = np.log(10 ** params["log_P_RCE"])
            ref_index = int(np.argmin(np.abs(self.flipped_ln_pressure - ln_p_ref)))

            temperature_flip = np.empty_like(self.pressure, dtype=float)
            temperature_flip[ref_index] = float(params["T_RCE"])

            for i in range(ref_index, len(self.pressure) - 1):
                temperature_flip[i + 1] = temperature_flip[i] * np.exp(
                    (self.flipped_ln_pressure[i + 1] - self.flipped_ln_pressure[i]) * dlnT_dlnP_array[i]
                )

            for i in range(ref_index - 1, -1, -1):
                temperature_flip[i] = temperature_flip[i + 1] * np.exp(
                    (self.flipped_ln_pressure[i] - self.flipped_ln_pressure[i + 1]) * dlnT_dlnP_array[i]
                )

            self.temperature = temperature_flip[::-1]
        else:
            assert "T_0" in params, "RCE profile requires T_0 or T_RCE parameter"
            self.temperature = [params["T_0"]]
            for i in range(len(self.pressure) - 1):
                T_i1 = self.temperature[-1] * np.exp(
                    (self.flipped_ln_pressure[i + 1] - self.flipped_ln_pressure[i]) * dlnT_dlnP_array[i]
                )
                self.temperature.append(T_i1)
            self.temperature = np.array(self.temperature)[::-1]

        self.dlnT_dlnP_array = dlnT_dlnP_array[::-1]
        return self.temperature

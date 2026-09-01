import os
import contextlib
import io
import numpy as np
import pandas as pd
import pathlib
import pyfastchem


MODULE_DIR = pathlib.Path(__file__).parent.resolve()
SPECIES_INFO_DEFAULT = str(MODULE_DIR / "data" / "chemistry" / "species_info.txt")

# Mapping from internal species names to pyfastchem gas species names.
FASTCHEM_NAME_MAP = {
    "H2": "H2",
    "He": "He",
    "H": "H",
    "H-": "H1-",
    "e-": "e-",
    "H2O": "H2O1",
    "CO": "C1O1",
    "13CO": "C1O1",
    "C18O": "C1O1",
    "C17O": "C1O1",
    "OH": "H1O1",
    "CN": "C1N1",
    "SiO": "O1Si1",
    "HF": "F1H1",
    "12CH": "C1H1",
    "CH": "C1H1",
    "12CH4": "C1H4",
    "13CH4": "C1H4",
    "CH3D": "C1H4",
    "CO2": "C1O2",
    "13CO2": "C1O2",
    "H2S": "H2S1",
    "NH3": "H3N1",
    "15NH3": "H3N1",
    "PH3": "H3P1",
    "Fe": "Fe",
    "Na": "Na",
    "K": "K",
    "Ca": "Ca",
    "Mg": "Mg",
    "Al": "Al",
    "Ti": "Ti",
    "V": "V",
    "Cr": "Cr",
    "Mn": "Mn",
    "Ni": "Ni",
    "Si": "Si",
    "S": "S",
    "TiO": "O1Ti1",
    "VO": "O1V1",
    "FeH": "Fe1H1",
    "CrH": "Cr1H1",
    "MgH": "H1Mg1",
}

# Isotope relations: isotope -> (main species, default isotope/main ratio).
ISOTOPES = {
    "13CO": ("CO", 1.0 / 90.0),
    "C18O": ("CO", 1.0 / 500.0),
    "C17O": ("CO", 1.0 / 2700.0),
    "H2O_181": ("H2O", 1.0 / 500.0),
    "H2O_171": ("H2O", 1.0 / 2500.0),
    "13CH4": ("12CH4", 1.0 / 90.0),
    "CH3D": ("12CH4", 1.0 / 6400.0),
    "13CO2": ("CO2", 1.0 / 90.0),
    "15NH3": ("NH3", 1.0 / 270.0),
    "13CH": ("12CH", 1.0 / 90.0),
}


class FastChemistry:
    """Equilibrium chemistry using a direct pyfastchem call."""

    def __init__(self, line_species, pressure, fastchem_input_path=None, species_info_file=None,
                 temperature_range=(300.0, 6000.0), quiet=False):
        if fastchem_input_path is None:
            fastchem_input_path = os.environ.get("FASTCHEM_INPUT_PATH")
        if fastchem_input_path is None:
            raise ValueError(
                "FastChem input path not set. Pass fastchem_input_path or set the "
                "FASTCHEM_INPUT_PATH environment variable."
            )
        self.fastchem_input_path = pathlib.Path(fastchem_input_path)
        self.temperature_range = tuple(temperature_range)
        assert len(self.temperature_range) == 2, "temperature_range must be (T_min, T_max)"
        assert self.temperature_range[0] < self.temperature_range[1], "temperature_range must have T_min < T_max"

        self.pressure = np.asarray(pressure)
        self.n_atm_layers = len(self.pressure)
        self.quiet = quiet

        if species_info_file is None:
            species_info_file = SPECIES_INFO_DEFAULT
        self._load_species_info(species_info_file)

        if isinstance(line_species, dict):
            line_species = list(line_species.values())
        self.line_species = line_species

        self.species = [self.pRT_name_dict.get(line_species_i, line_species_i) for line_species_i in self.line_species]

        self._init_fastchem()

    def _load_species_info(self, file):
        cols = ["name", "pRT_name", "mass", "C", "O", "H"]
        self.species_info = pd.DataFrame(np.loadtxt(file, dtype=str), columns=cols)
        self.species_info["mass"] = self.species_info["mass"].astype(float)
        for col in ["C", "O", "H"]:
            self.species_info[col] = self.species_info[col].astype(int)

        self.pRT_name_dict = {row["pRT_name"]: row["name"] for _, row in self.species_info.iterrows()}
        self.name_dict = {row["name"]: row["pRT_name"] for _, row in self.species_info.iterrows()}

    def _init_fastchem(self):
        element_file = self.fastchem_input_path / "element_abundances" / "asplund_2009.dat"
        logK_file = self.fastchem_input_path / "logK" / "logK.dat"

        assert element_file.is_file(), f"FastChem element abundance file not found: {element_file}"
        assert logK_file.is_file(), f"FastChem logK file not found: {logK_file}"

        verbose = 0 if self.quiet else 1
        init_stdout = io.StringIO() if self.quiet else contextlib.nullcontext()
        with contextlib.redirect_stdout(init_stdout), contextlib.redirect_stderr(init_stdout):
            self.fastchem = pyfastchem.FastChem(str(element_file), str(logK_file), verbose)
        self.init_element_abundances = np.array(self.fastchem.getElementAbundances())
        self.element_indices = {symbol: self.fastchem.getElementIndex(symbol) for symbol in ["H", "He", "C", "O"]}
        self.solar_c_to_o = self.init_element_abundances[self.element_indices["C"]] / self.init_element_abundances[self.element_indices["O"]]

    def _set_element_abundances(self, metallicity, c_to_o, elemental_ratios=None):
        """
        Scale solar element abundances.

        - Heavy elements (everything except H and He) are scaled by 10^metallicity.
        - Carbon is further scaled so that the final C/O ratio equals c_to_o.
        - Any element in elemental_ratios is additionally scaled by 10^dex, where
          the dex value is given with respect to the solar abundance.
        """
        if elemental_ratios is None:
            elemental_ratios = {}

        abundances = self.init_element_abundances.copy()
        n_elements = len(abundances)

        for i in range(n_elements):
            symbol = self.fastchem.getElementSymbol(i)
            if symbol not in ("H", "He"):
                abundances[i] *= 10.0 ** metallicity

        c_idx = self.element_indices["C"]
        o_idx = self.element_indices["O"]
        abundances[c_idx] *= c_to_o / self.solar_c_to_o

        for element, dex in elemental_ratios.items():
            idx = self.fastchem.getElementIndex(element)
            if idx == 9999999:
                if not self.quiet:
                    print(f"[FastChemistry] WARNING: element {element} not found in FastChem, skipping")
                continue
            abundances[idx] *= 10.0 ** dex

        self.fastchem.setElementAbundances(abundances.tolist())

    def _get_isotope_ratio(self, params, isotope, main):
        """Return the isotope/main volume mixing ratio (linear scale)."""
        linear_key = f"{isotope}/{main}"
        log_key = f"log_{linear_key}"

        if linear_key in params:
            return params[linear_key]
        if log_key in params:
            return 10.0 ** params[log_key]

        return ISOTOPES.get(isotope, (None, 1.0e-6))[1]

    def vmrs_to_mass_fractions(self, vmrs):
        """Convert volume mixing ratios to petitRADTRANS mass fractions."""
        from petitRADTRANS.chemistry.utils import (
            volume_mixing_ratios2mass_fractions,
            compute_mean_molar_masses,
        )

        mass_fractions = volume_mixing_ratios2mass_fractions(vmrs, mean_molar_masses=None)
        mass_fractions["MMW"] = compute_mean_molar_masses(vmrs, mode="vmr")
        return mass_fractions

    def __call__(self, params, temperature=None):
        """Return a dictionary of volume mixing ratios for each layer."""
        if temperature is None:
            temperature = getattr(self, "temperature", None)
        assert temperature is not None, "Temperature profile must be provided."
        temperature = np.asarray(temperature)
        assert len(temperature) == len(self.pressure)
        temperature = np.clip(temperature, self.temperature_range[0], self.temperature_range[1])

        metallicity = params.get("metallicity", params.get("z", 0.0))
        c_to_o = params.get("c_to_o", params.get("c_o", 0.55))
        elemental_ratios = params.get("elemental_ratios", {})

        self._set_element_abundances(metallicity, c_to_o, elemental_ratios)

        input_data = pyfastchem.FastChemInput()
        output_data = pyfastchem.FastChemOutput()
        input_data.temperature = temperature
        input_data.pressure = self.pressure
        input_data.equilibrium_condensation = False
        input_data.rainout_condensation = False

        flag = self.fastchem.calcDensities(input_data, output_data)
        if flag != 0:
            return -np.inf

        number_densities = np.array(output_data.number_densities)
        n_total = np.sum(number_densities, axis=1)

        vmrs = {}

        # Line species from FastChem.
        for line_species_i, species_name in zip(self.line_species, self.species):
            if species_name in ISOTOPES:
                continue

            fc_name = FASTCHEM_NAME_MAP.get(species_name, species_name)
            idx = self.fastchem.getGasSpeciesIndex(fc_name)
            if idx == 9999999:
                if not self.quiet:
                    print(f"[FastChemistry] WARNING: {species_name} ({fc_name}) not found in FastChem, setting to 0")
                vmrs[line_species_i] = np.zeros(self.n_atm_layers)
                continue

            vmrs[line_species_i] = number_densities[:, idx] / n_total

        # Isotopes derived from main species and ratio parameters.
        for line_species_i, species_name in zip(self.line_species, self.species):
            if species_name not in ISOTOPES:
                continue

            main, _ = ISOTOPES[species_name]
            fc_name = FASTCHEM_NAME_MAP.get(main, main)
            idx = self.fastchem.getGasSpeciesIndex(fc_name)
            if idx == 9999999:
                if not self.quiet:
                    print(f"[FastChemistry] WARNING: main species {main} not found in FastChem, setting isotope {species_name} to 0")
                vmrs[line_species_i] = np.zeros(self.n_atm_layers)
                continue

            vmr_main = number_densities[:, idx] / n_total
            ratio = self._get_isotope_ratio(params, species_name, main)
            vmrs[line_species_i] = vmr_main * ratio

        # Background species required by petitRADTRANS.
        for key, fc_name in [("H2", "H2"), ("He", "He"), ("H", "H"), ("e-", "e-"), ("H-", "H1-")]:
            idx = self.fastchem.getGasSpeciesIndex(fc_name)
            if idx == 9999999:
                vmrs[key] = np.zeros(self.n_atm_layers)
                continue
            vmrs[key] = number_densities[:, idx] / n_total

        return vmrs

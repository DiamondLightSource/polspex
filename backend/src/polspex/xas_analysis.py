"""
Load experimental data from nexus files


"""

import os
import numpy as np
import h5py
import hdfmap
from tabulate import tabulate
from lmfit.models import LinearModel, QuadraticModel, ExponentialModel, StepModel

from .environment import get_scan_number, replace_scan_number
from .parameters import AVAILABLE_EDGES
from .plot_models import gen_line_data, gen_plot_props


class XASMetadata:
    cmd = '(cmd|user_command|scan_command)'
    date = 'start_time'
    pol = 'polarisation?("lh")'
    iddgap = 'iddgap'
    rowphase = 'idutrp if iddgap == 100 else iddtrp'
    endstation = 'instrument_name'
    temp = '(T_sample|itc3_device_sensor_temp|lakeshore336_cryostat|lakeshore336_sample?(300))'
    rot = '(scmth|xabs_theta|ddiff_theta?(0))'
    field = 'np.sqrt(field_x?(0)**2 + field_y?(0)**2 + field_z?(0)**2)'
    field_x = 'field_x?(0)'
    field_y = 'field_y?(0)'
    field_z = '(magnet_field|ips_demand_field|field_z?(0))'
    energy = '(fastEnergy|pgm_energy|energye|energyh|energy)'
    monitor = '(C2|ca62sr|mcs16|macr16|mcse16|macj316|mcsh16|macj216)'
    tey = '(C1|ca61sr|mcs17|macr17|mcse17|macj317|mcsh17|macj217)'
    tfy = '(C3|ca63sr|mcs18|macr18|mcse18|macj318|mcsh18|macaj218)'


def check_metadata_paths(hdf_obj: h5py.File, hdf_map: hdfmap.NexusMap) -> dict[str, str]:
    """Return a dictionary of paths for the metadata"""
    return {k: hdf_map.eval(hdf_obj, f"_{v}") for k, v in vars(XASMetadata).items() if isinstance(v, str)}


def gen_metadata_str(filename: str) -> str:
    """Return metadata string for Nexus file"""
    meta_str = (
        "{filename}\n" +
        "{" + XASMetadata.date + "}\n" + 
        "{" + XASMetadata.cmd + "}\n" + 
        "E = {np.mean(" + XASMetadata.energy + "):.2f} eV\n" + 
        "T = {" + XASMetadata.temp + ":.2f} K\n" + 
        "B = {np.sqrt(%s**2 + %s**2 + %s**2)} T\n" % (XASMetadata.field_x, XASMetadata.field_y, XASMetadata.field_z) + 
        "Pol = {" + XASMetadata.pol + "}" 
    )
    hdf_map = hdfmap.create_nexus_map(filename)
    try:
        meta = hdf_map.format_hdf(hdf_map.load_hdf(), meta_str)
    except Exception as e:
        meta = str(e)
    return meta


def average_energy_scans(*args: tuple[np.ndarray]):
    """Return the minimum range covered by all input arguments"""
    min_energy = np.max([np.min(en) for en in args])
    max_energy = np.min([np.max(en) for en in args])
    min_step = np.min([np.min(np.abs(np.diff(en))) for en in args])
    return np.arange(min_energy, max_energy + min_step, min_step)


def combine_energy_scans(energy, *args: tuple[np.ndarray, np.ndarray]):
    """Average energy scans, interpolating at given energy"""
    data = np.zeros([len(args), len(energy)])
    for n, (en, dat) in enumerate(args):
        data[n, :] = np.interp(energy, en, dat)
    return data.mean(axis=0)


def determine_element(energy: float) -> tuple[str, str]:
    """
    Determine the element from the energy axis of the spectra.

    Note that only elements with available parameters are considered.

    :param energy: energy value (eV) to determine the element
    :return: (Element symbol, edge) as  strings
    """
    elements = list(AVAILABLE_EDGES.keys())
    values = np.array(list(AVAILABLE_EDGES.values()))
    return elements[np.argmin(np.abs(energy - values))], 'L2,3'  # currently only L2,3 edges are available


def orbital_angular_momentum(energy: np.ndarray, linear_xas: np.ndarray, 
                             difference: np.ndarray, nholes: float) -> float:
    """
    Calculate the sum rule for the angular momentum of the spectra
    using the formula:
    L = -2 * nholes * int[spectra d energy] / sum(spectra)

    :param energy: Energy axis of the spectra
    :param linear_xas: linear polarisation XAS spectra, or (left + right) polarisation
    :param difference: difference XAS spectra (right - left) for both polarisations
    :param nholes: Number of holes in the system
    :return: Angular momentum of the spectra
    """
    if len(energy) != len(linear_xas) or len(energy) != len(difference):
        raise ValueError(f"Energy and spectra must have the same length: {len(energy)} != {len(linear_xas)}")
    if nholes <= 0:
        raise ValueError(f"Number of holes must be greater than 0: {nholes}")
    
    # total intensity
    tot = np.trapezoid(linear_xas, energy)
    
    # Calculate the sum rule for the angular momentum
    L = -2 * nholes * np.trapezoid(difference, energy) / tot
    return L


def spin_angular_momentum(energy: np.ndarray, average: np.ndarray, 
                          difference: np.ndarray, nholes: float, 
                          split_energy: int | None = None, dipole_term: float = 0) -> float:
    """
    Calculate the sum rule for the spin angular momentum of the spectra
    using the formula:
    S = -2 * nholes * int[spectra d energy] / sum(spectra)

    :param energy: Energy axis of the spectra
    :param average: average XAS spectra (left + right) for both polarisations
    :param difference: difference XAS spectra (right - left) for both polarisations
    :param nholes: Number of holes in the system
    :param split_energy: energy to split the spectra between L3 and L2 (or None to use the middle of the spectra)
    :param dipole_term: magnetic dopole term (T_z), defaults to 0 for effective spin
    :return: Spin angular momentum of the spectra
    """
    if len(energy) != len(average) or len(energy) != len(difference):
        raise ValueError(f"Energy and spectra must have the same length: {len(energy)} != {len(average)}")
    if nholes <= 0:
        raise ValueError(f"Number of holes must be greater than 0: {nholes}")
    if split_energy is None:
        split_energy = (energy[0] + energy[-1]) / 2
    
    # total intensity
    tot = np.trapezoid(average, energy)
    
    # Calculate the sum rule for the spin angular momentum
    split_index = np.argmin(np.abs(energy - split_energy))
    l3_energy = energy[split_index:]  # L3 edge at lower energy
    l3_difference = difference[split_index:]
    l3_integral = np.trapezoid(l3_difference, l3_energy)
    l2_energy = energy[:split_index]
    l2_difference = difference[:split_index]
    l2_integral = np.trapezoid(l2_difference, l2_energy)
    S_eff = (3 / 2) * nholes * (l3_integral - 2 * l2_integral) / tot
    S = S_eff - dipole_term
    return S


def magnetic_moment(orbital: float, spin: float) -> float:
    """
    Calculate the magnetic moment of the system using the formula:
    M = -g * (L + 2 * S)  WHERE DOES THIS COME FROM?

    :param orbital: Orbital angular momentum of the system
    :param spin: Spin angular momentum of the system
    :return: Magnetic moment of the system
    """
    print('magnetic moment is probably wrong!')
    g = 2.0  # Landé g-factor for free electron
    return -g * (orbital + 2 * spin)


def xmcd_table(energy: np.ndarray, average: np.ndarray, difference: np.ndarray,
                nholes: float, split_energy: int | None = None, dipole_term: float = 0) -> str:
    """
    Create a table with the sum rules for the spectra.

    :param energy: Energy axis of the spectra
    :param average: average XAS spectra (left + right) for both polarisations
    :param difference: difference XAS spectra (right - left) for both polarisations
    :param nholes: Number of holes in the system
    :param split_energy: energy to split the spectra between L3 and L2 (or None to use the middle of the spectra)
    :param dipole_term: magnetic dopole term (T_z), defaults to 0 for effective spin
    :return: table with the sum rules for the spectra
    """
    orbital = orbital_angular_momentum(energy, average, difference, nholes)
    spin = spin_angular_momentum(energy, average, difference, nholes, split_energy, dipole_term)
    table2 = [[r'sL$_z$', 'sS$_{eff}$'], [orbital, spin]]
    tfmt = 'github'
    table_string = '\n'.join([
        # "## Theoretical values (Quanty)",
        # tabulate(table1, headers='firstrow', tablefmt=tfmt),
        "### Sum rules :",
        tabulate(table2, headers='firstrow', tablefmt=tfmt),
        # "### Sum rules 0:",
        # tabulate(table3, headers='firstrow', tablefmt=tfmt),
        # "### Deviations:",
        # tabulate(table4, headers='firstrow', tablefmt=tfmt)
    ])
    return table_string


def signal_jump(energy, signal, ev_from_start=5., ev_from_end=None) -> float:
    """Return signal jump from start to end"""
    ev_from_end = ev_from_end or ev_from_start
    ini_signal = np.mean(signal[energy < np.min(energy) + ev_from_start])
    fnl_signal = np.mean(signal[energy > np.max(energy) - ev_from_end])
    return fnl_signal - ini_signal


def subtract_flat_background(energy, signal, ev_from_start=5.) -> tuple[np.ndarray, np.ndarray]:
    """Subtract flat background"""
    bkg = np.mean(signal[energy < np.min(energy) + ev_from_start])
    return np.subtract(signal, bkg), bkg * np.ones_like(signal)


def normalise_background(energy, signal, ev_from_start=5.) -> tuple[np.ndarray, np.ndarray]:
    """Normalise background to one"""
    bkg = np.mean(signal[energy < np.min(energy) + ev_from_start])
    return np.divide(signal, bkg), bkg * np.ones_like(signal)


def fit_linear_background(energy, signal, ev_from_start=5.) -> tuple[np.ndarray, np.ndarray]:
    """Use lmfit to determine sloping background"""
    model = LinearModel(prefix='bkg_')
    region = energy < np.min(energy) + ev_from_start
    en_region = energy[region]
    sig_region = signal[region]
    pars = model.guess(sig_region, x=en_region)
    fit_output = model.fit(sig_region, pars, x=en_region)
    bkg = fit_output.eval(x=energy)
    return signal - bkg, bkg


def fit_curve_background(energy, signal, ev_from_start=5.) -> tuple[np.ndarray, np.ndarray]:
    """Use lmfit to determine sloping background"""
    model = QuadraticModel(prefix='bkg_')
    # region = (energy < np.min(energy) + ev_from_start) + (energy > np.max(energy) - ev_from_start)
    region = energy < np.min(energy) + ev_from_start
    en_region = energy[region]
    sig_region = signal[region]
    pars = model.guess(sig_region, x=en_region)
    fit_output = model.fit(sig_region, pars, x=en_region)
    bkg = fit_output.eval(x=energy)
    return signal - bkg, bkg


def fit_exp_background(energy, signal, ev_from_start=5.) -> tuple[np.ndarray, np.ndarray]:
    """Use lmfit to determine sloping background"""
    model = ExponentialModel(prefix='bkg_')
    # region = (energy < np.min(energy) + ev_from_start) + (energy > np.max(energy) - ev_from_start)
    region = energy < np.min(energy) + ev_from_start
    en_region = energy[region]
    sig_region = signal[region]
    pars = model.guess(sig_region, x=en_region)
    fit_output = model.fit(sig_region, pars, x=en_region)
    # print('exp background\n:', fit_output.fit_report())
    bkg = fit_output.eval(x=energy)
    return signal - bkg, bkg


def fit_exp_step(energy, signal, ev_from_start=5., ev_from_end=5.) -> tuple[np.ndarray, np.ndarray]:  # good?
    """Use lmfit to determine sloping background"""
    model = ExponentialModel(prefix='bkg_') + StepModel(form='arctan', prefix='jmp_')  # form='linear'
    region = (energy < np.min(energy) + ev_from_start) + (energy > np.max(energy) - ev_from_start)
    en_region = energy[region]
    sig_region = signal[region]
    # pars = model.guess(sig_region, x=en_region)
    guess_jump = signal_jump(energy, signal, ev_from_start, ev_from_end)
    pars = model.make_params(
        bkg_amplitude=np.max(sig_region),
        bkg_decay=100.0,
        jmp_amplitude=dict(value=guess_jump, min=0),
        jmp_center=energy[np.argmax(signal)],  # np.mean(energy),
        jmp_sigma=dict(value=1, min=0),
    )
    fit_output = model.fit(sig_region, pars, x=en_region)
    bkg = fit_output.eval(x=energy)
    jump = fit_output.params['jmp_amplitude']
    # print('fit_exp_step:\n', fit_output.fit_report())
    # print(jump)
    return (signal - bkg) / jump, bkg / jump


def i06_norm(energy, signal) -> tuple[np.ndarray, np.ndarray]:
    """I06 norm and post_edge_norm option"""
    sig = 1.0 * signal
    sig /= sig[energy < energy[0] + 5].mean()  # nomalise by the average of a range of energy
    jump = sig[energy > energy[-1] - 5].mean() - sig[energy < energy[0] + 5].mean()

    print(jump)
    print(sig[energy < energy[0] + 5].mean())
    sig -= sig[energy < energy[0] + 5].mean()  # - 1
    jump2 = sig[energy > energy[-1] - 5].mean()
    print(jump2)
    sig /= jump2
    return sig, jump2 * np.ones_like(sig)


def fit_bkg_then_norm_to_peak(energy, signal, ev_from_start=5., ev_from_end=5.) -> tuple[np.ndarray, np.ndarray]:  # good?
    """Fit the background then normalise the post-edge to 1"""
    fit_signal, bkg = fit_exp_background(energy, signal, ev_from_start)
    peak = np.max(abs(fit_signal))
    return fit_signal / peak, bkg / peak


def fit_bkg_then_norm_to_jump(energy, signal, ev_from_start=5., ev_from_end=5.) -> tuple[np.ndarray, np.ndarray]:  # good?
    """Fit the background then normalise the post-edge to 1"""
    fit_signal, bkg = fit_exp_background(energy, signal, ev_from_start)
    jump = signal_jump(energy, fit_signal, ev_from_start, ev_from_end)
    return fit_signal / abs(jump), bkg / abs(jump)


BACKGROUND_FUNCTIONS = {
    # description: (en, sig, *args, **kwargs) -> spectra, bkg
    'flat': subtract_flat_background,
    'norm': normalise_background,
    'linear': fit_linear_background,
    'curve': fit_curve_background,
    'exp': fit_exp_background,
}


class XASMeasurement:
    def __init__(self, filename: str, nexus_map: hdfmap.NexusMap):
        self.filename = filename
        self.basename = os.path.basename(filename)
        self.scan_number = get_scan_number(filename)
        self.map = nexus_map

        with hdfmap.load_hdf(filename) as hdf:
            try:
                self.cmd = self.map.eval(hdf, XASMetadata.cmd)
                self.polarisation = self.map.eval(hdf, XASMetadata.pol)
                self.temperature = self.map.eval(hdf, XASMetadata.temp)
                self.field_x = self.map.eval(hdf, XASMetadata.field_x)
                self.field_y = self.map.eval(hdf, XASMetadata.field_y)
                self.field_z = self.map.eval(hdf, XASMetadata.field_z)
                self.energy = self.map.eval(hdf, XASMetadata.energy)
                self.mean_energy = np.mean(self.energy)
                default = np.ones_like(self.energy)
                self.monitor = self.map.eval(hdf, XASMetadata.monitor, default=default)
                self.tey = self.map.eval(hdf, XASMetadata.tey, default=default)
                self.tfy = self.map.eval(hdf, XASMetadata.tfy, default=default)
            except Exception as e:
                paths = check_metadata_paths(hdf, self.map)
                path_str = '\n '.join([f"{k}: {v}" for k, v in paths.items()])
                raise ValueError(f"Error loading {self.basename}\n    paths:\n{path_str}\n\nException:\n{e}")
            if len(self.energy) <= 1:
                en_path = self.map.eval(hdf, '_' + XASMetadata.energy)
                raise ValueError(f"Energy has the wrong shape: Energy [{self.energy.shape}]: {en_path}")
        
        self.tey = self.tey / self.monitor
        self.tfy = self.tfy / self.monitor
        self.tey_background = np.zeros_like(self.tey)
        self.tfy_background = np.zeros_like(self.tfy)
        self.process = 'raw'
        self.label = f"{self.scan_number}: {self.polarisation}"
    
    def __repr__(self):
        return f"XASMeasurement(#{self.scan_number}, '{self.polarisation}')"
    
    def determine_element(self) -> str:
        """Determine the element and edge from the energy axis of the spectra."""
        return determine_element(self.energy.mean())
    
    def remove_background(self, name='flat', *args, **kwargs):
        self.tey, self.tey_background = BACKGROUND_FUNCTIONS[name](self.energy, self.tey, *args, **kwargs)
        self.tfy, self.tfy_background = BACKGROUND_FUNCTIONS[name](self.energy, self.tfy, *args, **kwargs)
        self.process = f"{self.process} - {name}"

    def norm_to_peak(self):
        peak_tey =  np.max(abs(self.tey))
        peak_tfy =  np.max(abs(self.tfy))
        self.tey = self.tey / peak_tey
        self.tfy = self.tfy / peak_tfy
        self.tey_background = self.tey_background / peak_tey
        self.tfy_background = self.tfy_background / peak_tfy
        self.process = f"({self.process}) / peak"
        

    def norm_to_jump(self, ev_from_start=5., ev_from_end=None):
        jump_tey = signal_jump(self.energy, self.tey, ev_from_start, ev_from_end)
        jump_tfy = signal_jump(self.energy, self.tfy, ev_from_start, ev_from_end)
        self.tey = self.tey / jump_tey
        self.tfy = self.tfy / jump_tfy
        self.tey_background = self.tey_background / jump_tey
        self.tfy_background = self.tfy_background / jump_tfy
        self.process = f"({self.process}) / jump"
    
    def auto_norm(self, name='exp', ev_from_start=5., ev_from_end=None):
        """
        Automatically normalise the spectra using the given method.
        The method is chosen based on the energy range of the spectra.
        """
        # remove flat background
        self.remove_background('flat', ev_from_start=ev_from_start, ev_from_end=ev_from_end)
        # remove curved background
        self.remove_background(name, ev_from_start=ev_from_start, ev_from_end=ev_from_end)
        # normalise to jump
        self.norm_to_jump(ev_from_start=ev_from_start, ev_from_end=ev_from_end)
    
    def plot(self):
        return gen_line_data(self.energy, self.tey, label=self.label)
    
    def plot_background(self):
        return gen_line_data(self.energy, self.tey_background, fmt=':', label=f"{self.label} background", colour='black')


class PolarisationPair:
    def __init__(self, measurement1: XASMeasurement, measurement2: XASMeasurement):
        self.measurement1 = measurement1
        self.measurement2 = measurement2

        # calculate difference
        av_energy = average_energy_scans(measurement1.energy, measurement2.energy)
        interp_pc = combine_energy_scans(av_energy, (measurement1.energy, measurement1.tey))
        interp_nc = combine_energy_scans(av_energy, (measurement2.energy, measurement2.tey))

        self.energy = av_energy
        self.difference = interp_pc - interp_nc
        self.temperature = (measurement1.temperature + measurement2.temperature) / 2
        self.field_x = (measurement1.field_x + measurement2.field_x) / 2
        self.field_y = (measurement1.field_y + measurement2.field_y) / 2
        self.field_z = (measurement1.field_z + measurement2.field_z) / 2

        self.title = (
            f"#{measurement1.scan_number}[{measurement1.polarisation}] - "
            f"#{measurement2.scan_number}[{measurement2.polarisation}]\n"# + 
            # f"Field = {round(self.field, 3): .3g} T, Temp = {round(self.temperature, 3): .3g} K"
        )
    
    def __repr__(self):
        return f"PolarisationPair({self.measurement1}, {self.measurement2})"
    
    def plot(self):
        return (
            gen_line_data(self.energy, self.difference, label=f"{self.measurement1.polarisation} - {self.measurement2.polarisation}", colour='red')
        )
    
    def output(self):
        lines = (
            self.measurement1.plot(), 
            self.measurement1.plot_background(),
            self.measurement2.plot(), 
            self.measurement2.plot_background(),
            self.plot()
        )
        return gen_plot_props(
            self.title,
            'Energy (eV)',
            'Difference (a.u.)',
            (self.energy.min(), self.energy.max()),
            (self.difference.min(), max(self.measurement1.tey.max(), self.measurement2.tey.max(), self.difference.max())),
            *lines
        )


class PolarisationSet:
    def __init__(self, *measurementPairs: PolarisationPair):
        self.measurements = measurementPairs

        # calculate difference
        av_energy = average_energy_scans(*[pair.energy for pair in measurementPairs])
        interp_pc = combine_energy_scans(av_energy, *[(pair.measurement1.energy, pair.measurement1.tey) for pair in measurementPairs])
        interp_nc = combine_energy_scans(av_energy, *[(pair.measurement2.energy, pair.measurement2.tey) for pair in measurementPairs])

        self.energy = av_energy
        self.mean_energy = np.mean(av_energy)
        self.element, self.edge = determine_element(self.mean_energy)
        self.xas1 = interp_pc 
        self.xas2 = interp_nc
        self.difference = interp_pc - interp_nc
        self.temperature = sum([pair.temperature for pair in measurementPairs]) / len(measurementPairs)
        self.field_x = sum([pair.field_x for pair in measurementPairs]) / len(measurementPairs)
        self.field_y = sum([pair.field_y for pair in measurementPairs]) / len(measurementPairs)
        self.field_z = sum([pair.field_z for pair in measurementPairs]) / len(measurementPairs)
        self.field = np.sqrt(self.field_x**2 + self.field_y**2 + self.field_z**2)

        pol1 = self.measurements[0].measurement1.polarisation
        pol2 = self.measurements[0].measurement2.polarisation
        scans1 = ', '.join([str(pair.measurement1.scan_number) for pair in measurementPairs])
        scans2 = ', '.join([str(pair.measurement2.scan_number) for pair in measurementPairs])
        self.title = f"Average: [{pol1}]: #({scans1}), [{pol2}]: #({scans2})"

        self.pol1 = pol1 
        self.pol2 = pol2
    
    def __repr__(self):
        return f"PolarisationSet({len(self.measurements)} measurements, {self.element}{self.edge}, T={self.temperature:.1f} K, B={self.field:.1f} T)"
    
    def table(self):
        output = xmcd_table(
            energy=self.energy,
            average=self.xas1 + self.xas2,
            difference=self.difference,
            nholes=2,  # TODO: get from metadata
            split_energy=None,  # TODO: get from metadata
            dipole_term=0,  # TODO: get from metadata
        )
        return output
    
    def plot(self):
        return (
            gen_line_data(self.energy, self.xas1, label=f"XAS {self.pol1}", colour='blue'),
            gen_line_data(self.energy, self.xas2, label=f"XAS {self.pol2}", colour='green'),
            gen_line_data(self.energy, self.difference, label=f"{self.pol1} - {self.pol2}", colour='red'),
        )
    
    def output(self):
        return gen_plot_props(
            self.title,
            'Energy (eV)',
            'Difference (a.u.)',
            (self.energy.min(), self.energy.max()),
            (self.difference.min(), max(self.xas1.max(), self.xas2.max(), self.difference.max())),
            *self.plot()
        )
    

def load_xas_measurements(*filenames: str) -> list[XASMeasurement]:
    """
    Load XAS measurements from files
    """
    nexus_map = hdfmap.create_nexus_map(filenames[0])
    return [XASMeasurement(filename, nexus_map) for filename in filenames]


def find_matching_scans(filename: str, match_field: str = 'scan_command', 
                        search_scans_before: int = 10, search_scans_after: int | None = None) -> list[str]:
    """
    Find scans with scan numbers close to the current file with matching scan command

    :param filename: nexus file to start at (must include scan number in filename)
    :param match_field: nexus field to compare between scan files
    :param search_scans_before: number of scans before current scan to look for
    :param search_scans_after: number of scans after current scan to look for (None==before)
    :returns: list of scan files that exist and have matching field values
    """
    nexus_map = hdfmap.create_nexus_map(filename)
    field_value = nexus_map.eval(nexus_map.load_hdf(), match_field)
    scanno = get_scan_number(filename)
    if search_scans_after is None:
        search_scans_after = search_scans_before
    matching_files = []
    for scn in range(scanno - search_scans_before, scanno + search_scans_after):
        new_filename = replace_scan_number(filename, scn)
        if os.path.isfile(new_filename):
            new_field_value = nexus_map.eval(hdfmap.load_hdf(new_filename), match_field)
            if field_value == new_field_value:
                matching_files.append(new_filename)
    return matching_files


def find_similar_measurements(*filenames: str, energy_tol=1., temp_tol=1., field_tol=0.1) -> list[XASMeasurement]:
    """
    Find similar measurements based on energy, temperature and field.

    Each measurement is compared to the first one in the list, using energy, temperature and field tolerances.

    The polarisation is also checked to be similar (lh, lv or cl, cr).

    Scans with different or missing metadata are removed from the list.

    :param filenames: List of filenames to compare
    :param energy_tol: Tolerance for energy comparison (default: 1.0 eV)
    :param temp_tol: Tolerance for temperature comparison (default: 0.1 K)
    :param field_tol: Tolerance for field comparison (default: 0.1 T)
    :return: List of similar measurements
    """
    if len(filenames) == 1:
        filenames = find_matching_scans(filenames[0])
    measurements = load_xas_measurements(*filenames)
    mean_energy = measurements[0].mean_energy
    temperature = measurements[0].temperature
    field_x = measurements[0].field_x
    field_y = measurements[0].field_y
    field_z = measurements[0].field_z
    polarisation = measurements[0].polarisation
    if polarisation in ['lh', 'lv']:
        similar_pols = ['lh', 'lv']
    elif polarisation in ['cl', 'cr']:
        similar_pols = ['cl', 'cr']
    elif polarisation in ['nc', 'pc']:
        similar_pols = ['nc', 'pc']
    else:
        raise ValueError(f"Unknown polarisation: {polarisation}")
    similar = []
    for m in measurements:
        if (
            abs(m.mean_energy - mean_energy) < energy_tol and 
            abs(m.temperature - temperature) < temp_tol and
            abs(m.field_x - field_x) < field_tol and
            abs(m.field_y - field_y) < field_tol and
            abs(m.field_z - field_z) < field_tol and
            m.polarisation in similar_pols
        ):
            similar.append(m)
        else:
            print(f"Measurement {m} is not similar to {measurements[0]}")   
    return similar


def find_pairs(*filenames: str, background_type: str | None = None, check_similar=True) -> PolarisationSet:
    """
    returns pairs of xas measurements in paired polarisations (cl,cr or lh,lv etc)
    """
    if check_similar:
        measurements = find_similar_measurements(*filenames)
    else:
        measurements = load_xas_measurements(*filenames)
    if len(measurements) < 2:
        raise ValueError(f"Not enough measurements! {len(measurements)}")
    
    print('Measurements: ', measurements)
    polarisations = [m.polarisation for m in measurements]
    print('polarisations: ', polarisations)
    '''
    either, find matching pattern 'cl,cr,cr,cl', or find pairs of different polarisations
    '''
    pol_indexes = [
        [i for i, x in enumerate(polarisations) if x == pol]
        for pol in set(polarisations)
    ]
    if len(pol_indexes) < 2:
        raise ValueError(f"Not enough polarisations! {set(polarisations)}")
    if background_type:
        print('Removing background')
        for m in measurements:
            m.remove_background('flat')
            if background_type != 'flat':
                m.remove_background(background_type)
            # m.norm_to_jump(ev_from_start=5., ev_from_end=5.)
    print(f"Pairing Polarisations: {polarisations[pol_indexes[0][0]]} and {polarisations[pol_indexes[1][0]]}")
    pairs = [
        PolarisationPair(measurements[i1], measurements[i2])
        for i1, i2 in zip(pol_indexes[0], pol_indexes[1])
    ]
    return PolarisationSet(*pairs)


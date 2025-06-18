
import os
import subprocess
import numpy as np
from tabulate import tabulate

from .integrate import trapz, romb
from .plot_models import gen_line_data, gen_plot_props


def load_processed_spectra(ion: str, path: str, rawout: subprocess.CompletedProcess):
    """
    Load spectra from completed Quanty simulation

    Note that due to the simplicity of the file paths, the files are only correct at the end
    of the quanty simulation, as future simulations may overwrite them.

    The output dict has structure:
    {   
        # float values
        'E': energy [eV]
        'S2': 
        'L2': 
        'J2': 
        'S_k': 
        'L_k': 
        'J_k': 
        'T_k': 
        'LdotS': 
        'Seff': S_k + T_k
        'spectra': {
            # numpy arrays [energy, spectra]
            'iso': (cr + cl) / 2  isotropic (unpolarised) spectra
            'mcd': (cr - cl) circular dichroism
            'mld': (lv - lh) linear dichroism
            'cl': circular left
            'cr': circular right
            'lh': linear horizontal
            'lv': linear vertical
        }
    }

    """
    output_values = {}
    output_values['quanty'] = treat_output(rawout)

    label = ion + '_XAS'
    xz = np.loadtxt(os.path.join(path, label + '_iso.spec'), skiprows=5)  # == (xr + xl) / 2  isotropic (unpolarised) spectra
    mcd = np.loadtxt(os.path.join(path, label + '_cd.spec'), skiprows=5)  # == xr - xl  circular dichroism
    mld = np.loadtxt(os.path.join(path, label + '_ld.spec'), skiprows=5)  # == Gv - Gh, 'ld' linear dichroism
    xl = np.loadtxt(os.path.join(path, label + '_l.spec'), skiprows=5)  # circular left
    xr = np.loadtxt(os.path.join(path, label + '_r.spec'), skiprows=5)  # circular right
    lh = np.loadtxt(os.path.join(path, label + '_h.spec'), skiprows=5)  # linear horizontal
    lv = np.loadtxt(os.path.join(path, label + '_v.spec'), skiprows=5)  # linear vertical
    output_values['spectra'] = {
        'iso': xz,
        'mcd': mcd,
        'mld': mld,
        'cl': xl,
        'cr': xr,
        'lh': lh,
        'lv': lv
    }

    # sum rules
    # mcd2 = mcd = cr - cl
    # lz = -2 * nh * mcd2 / xas
    # lz0 = -2 * nh * mcd2 / xas0
    # szef = 3/2 * nh * mcd2[l3] - 2 * mcd2[l2] / xas
    # szef = 3/2 * nh * mcd2[l3] - 2 * mcd2[l2] / xas0
    l23_split = xl.shape()[0] // 2
    pos_l3 = xr[:l23_split, :]
    neg_l3 = xl[:l23_split, :]
    pos_l2 = xr[l23_split:, :]
    neg_l2 = xl[l23_split:, :]
    nh = 10 - 7
    lz, szef = calculate_sum_rules(pos_l2, neg_l2, pos_l3, neg_l3, nh, iso=xz)
    lz0, szef0 = calculate_sum_rules(pos_l2, neg_l2, pos_l3, neg_l3, nh)
    output_values['sum rules'] = {
        'lz': lz,
        'szef': szef,
        'lz0': lz0,
        'szef0': szef0
    }
    return output_values


def integrate_spectra(spectra: np.ndarray, use_trapz=False):
    """
    Integrate a spectra array
    """
    if use_trapz:
        tot = trapz(spectra[:, 2], spectra[:, 0])
    else:
        tot = romb(spectra[:, 2], dx=float(spectra[1, 0] - spectra[0, 0]))
    return tot


def calculate_sum_rules(pos_l2: np.ndarray, neg_l2: np.ndarray, 
                        pos_l3: np.ndarray, neg_l3: np.ndarray, 
                        nh: float = 0, iso: np.ndarray | None = None, use_trapz=False):
    """

    <Lz> = -2 * nh * integral(delta_l2 + delta_l3) / integral(mu0)
    <Sz> = 3/2 * nh * integral(delta_l3 - 2 * delta_l2) / integral(mu0)

    nh = number of holes (Co = 10 - 7 = 3)
    delta_l2 - difference in absorption spectrum at L2 between polarisations  mup - mun
    delta_l3 - difference in absorption spectrum at L3 between polarisations  mup - mun
    iso - linear polarisation == mu0, or None to use 3/2 (mup + mun)
    mup - absorption spectrum L23 for left (+) circularly polarized light.
    mun - absorption spectrum L23 for right (-) circularly polarized light.
    md - magnetic dichroism mun - mup
    mu0 - absorption spectrum for linearly polarized light, with polarization parallel to quantization axis.
    
    :returns: lz, szef - orbital and spin components of magnetic moment
    """

    if iso is None:
        tot = integrate_spectra(pos_l3 + neg_l3, use_trapz) + integrate_spectra(pos_l2 + neg_l3, use_trapz)
    else:
        tot = integrate_spectra(iso, use_trapz)
    delta_l3 = pos_l3 - neg_l3 # CANT ADD 2xn arrays! DEFINE SPECTRA
    delta_l2 = pos_l2 - neg_l2 
    l3 = integrate_spectra(delta_l3, use_trapz)
    l2 = integrate_spectra(delta_l2, use_trapz)

    lz = -2 * nh * (l3 + l2) / tot
    szef = 3 / 2 * nh * (l3 - 2 * l2) / tot
    return lz, szef


def process_results2(ion: str, path: str, Nelec: float, edge: float, Rawout: subprocess.CompletedProcess):
    """
    Analyse completed Quanty simulation
    """

    label = ion + '_XAS'
    nh = 10 - Nelec 
    xz = np.loadtxt(os.path.join(path, label + '_iso.spec'), skiprows=5)  # == (xr + xl) / 2
    mcd = np.loadtxt(os.path.join(path, label + '_cd.spec'), skiprows=5)  # == xr - xl
    xl = np.loadtxt(os.path.join(path, label + '_l.spec'), skiprows=5)
    xr = np.loadtxt(os.path.join(path, label + '_r.spec'), skiprows=5)

    



def process_results(ion: str, path: str, Nelec: float, edge: float, Rawout: subprocess.CompletedProcess):
    """
    Analyse completed Quanty simulation
    """

    label = ion + '_XAS'
    xz = np.loadtxt(os.path.join(path, label + '_iso.spec'), skiprows=5)  # == (xr + xl) / 2
    mcd = np.loadtxt(os.path.join(path, label + '_cd.spec'), skiprows=5)  # == xr - xl
    xl = np.loadtxt(os.path.join(path, label + '_l.spec'), skiprows=5)
    xr = np.loadtxt(os.path.join(path, label + '_r.spec'), skiprows=5)
    mcd2 = xr.copy()
    mcd2[:, 2] = xl[:, 2] - xr[:, 2]
    npts = np.shape(xz)[0]
    mcd2 = mcd
    # TOTAL spectra
    xas = xz.copy()
    xas[:, 2] = xz[:, 2] + xl[:, 2] + xr[:, 2]

    xas0 = xz.copy()
    xas0[:, 2] = (xl[:, 2] + xr[:, 2]) / 2 + xl[:, 2] + xr[:, 2]

    dx = xz.copy()
    dx[:, 2] = xl[:, 2] + xr[:, 2] - 2 * xz[:, 2]

    # xas = iso + cl + cr
    # xas0 = (cl + cr) / 2 + (cl + cr)  (no requirement for iso, more similar to experiment)
    # xas0 = 3/2 * (cl + cr)
    # dx = cl + cr - 2iso

    # deltaXas = dx / xas

    # sum rules
    # mcd2 = mcd = cr - cl
    # lz = -2 * nh * mcd2 / xas
    # lz0 = -2 * nh * mcd2 / xas0
    # szef = 3/2 * nh * mcd2[l3] - 2 * mcd2[l2] / xas
    # szef = 3/2 * nh * mcd2[l3] - 2 * mcd2[l2] / xas0


    # ### Integration using Trapezoidal rule

    nh = 10 - Nelec #  params['Nelec']

    use_trapz = False
    if use_trapz:
        tot = trapz(xas[:, 2], xas[:, 0])
        tot0 = trapz(xas0[:, 2], xas0[:, 0])
        dx0 = trapz(dx[:, 2], dx[:, 0])
    else:
        tot = romb(xas[:, 2], dx=float(xas[1, 0] - xas[0, 0]))
        tot0 = romb(xas0[:, 2], dx=float(xas0[1, 0] - xas0[0, 0]))
        dx0 = romb(dx[:, 2], dx=float(dx[1, 0] - dx[0, 0]))

    deltaXas = dx0 / tot

    if use_trapz:
        lz = -2 * nh * trapz(mcd2[:, 2], mcd2[:, 0]) / tot
        szef = 3 / 2 * nh * (
                trapz(mcd2[0:npts // 2, 2], mcd2[0:npts // 2, 0]) -
                2 * trapz(mcd2[npts // 2:, 2], mcd2[npts // 2:, 0])
        ) / tot
        lz0 = -2 * nh * trapz(mcd2[:, 2], mcd2[:, 0]) / tot0
        szef0 = 3 / 2 * nh * (
                trapz(mcd2[0:npts // 2, 2], mcd2[0:npts // 2, 0]) -
                2 * trapz(mcd2[npts // 2:, 2], mcd2[npts // 2:, 0])
        ) / tot0
    else:
        print(len(mcd2[npts // 2:, 2]), len(mcd2[0:npts // 2 + 1]))
        mydelta = mcd2[1, 0] - mcd2[0, 0]
        lz = -2 * nh * romb(mcd2[:, 2], float(mydelta)) / tot
        szef = 3 / 2 * nh * (
                romb(mcd2[0:npts // 2 + 1, 2], float(mydelta)) -
                2 * romb(mcd2[npts // 2:, 2], float(mydelta))
        ) / tot
        lz0 = -2 * nh * romb(mcd2[:, 2], float(mydelta)) / tot0
        szef0 = 3 / 2 * nh * (
                romb(mcd2[0:npts // 2 + 1, 2], float(mydelta)) -
                2 * romb(mcd2[npts // 2:, 2], float(mydelta))
        ) / tot0

    # Sum rules table
    outdic = treat_output(Rawout)
    Lz_t = outdic['L_k']
    Sz_t = outdic['S_k']
    Tz_t = outdic['T_k']
    Seff_t = outdic['S_k'] + outdic['T_k']

    table1 = [
        [r'L$$_z$$', r'S$_{eff}$', r'S$_{z}$', r'T$_{z}$'],
        [Lz_t, Seff_t, Sz_t, Tz_t]
    ]
    table2 = [
        [r'sL$_z$', 'sS$_{eff}$'], 
        [lz, szef]
    ]
    table3 = [
        [r's$_0$L$_z$', 's$_0$S$_{eff}$'], 
        [lz0, szef0]
    ]
    table4 = [
        [
            r'$\Delta$XAS (%)', 
            r'$\Delta$L$_{z}$ (%)', 
            r'$\Delta$S$_{eff}$ (%)',
            r'$\Delta_0$L$_{z}$ (%)',
            r'$\Delta_0$S$_{eff}$ (%)'
        ],
        [
            deltaXas * 100, 
            100 * (abs(Lz_t) - abs(lz)) / Lz_t,
            100 * (abs(Seff_t) - abs(szef)) / Seff_t,
            100 * (abs(Lz_t) - abs(lz0)) / Lz_t,
            100 * (abs(Seff_t) - abs(szef0)) / Seff_t
        ]
    ]
    
    tfmt = 'github'
    table_string = '\n'.join([
        "## Theoretical values (Quanty)",
        tabulate(table1, headers='firstrow', tablefmt=tfmt),
        "### Sum rules :",
        tabulate(table2, headers='firstrow', tablefmt=tfmt),
        "### Sum rules 0:",
        tabulate(table3, headers='firstrow', tablefmt=tfmt),
        "### Deviations:",
        tabulate(table4, headers='firstrow', tablefmt=tfmt)
    ])

    # Plots
    lines = [
        gen_line_data(xz[:, 0] + edge, xz[:, 2], 'r-', label='z-pol'),
        gen_line_data(xl[:, 0] + edge, xl[:, 2], 'b', label='left'),
        gen_line_data(xr[:, 0] + edge, xr[:, 2], 'g', label='right'),
    ]
    xlim = (-10 + edge, 20 + edge)
    axis1 = gen_plot_props('XAS', 'Energy [eV]', 'Intensity [a.u.]', xlim, None, *lines)

    lines = [
        # gen_line_data(xas[:, 0] + edge, xas[:, 2] / 3, 'k', label='average'),
        # gen_line_data(mcd[0:npts // 2, 0] + edge, mcd[0:npts // 2, 2], 'r', label=r'L$_3$'),
        # gen_line_data(mcd[npts // 2:, 0] + edge, mcd[npts // 2:, 2], 'b', label=r'L$_2$'),
        gen_line_data(mcd[:, 0] + edge, mcd[:, 2], label='XMCD', colour='purple'),
    ]
    axis2 = gen_plot_props('XMCD', 'Energy [eV]', 'Intensity [a.u.]', xlim, None, *lines)
    return table_string, axis1, axis2

    

def post_proc_output_only(ion: str, path: str, edge: str):

    label = ion + '_XAS'
    xz = np.loadtxt(os.path.join(path, label + '_iso.spec'), skiprows=5)
    mcd = np.loadtxt(os.path.join(path, label + '_cd.spec'), skiprows=5)
    xl = np.loadtxt(os.path.join(path, label + '_l.spec'), skiprows=5)
    xr = np.loadtxt(os.path.join(path, label + '_r.spec'), skiprows=5)

    mcd2 = xr.copy()
    mcd2[:, 2] = xl[:, 2] - xr[:, 2]

    # TOTAL spectra
    xas = xz.copy()
    xas[:, 2] = xz[:, 2] + xl[:, 2] + xr[:, 2]

    xas0 = xz.copy()
    xas0[:, 2] = (xl[:, 2] + xr[:, 2]) / 2 + xl[:, 2] + xr[:, 2]

    dx = xz.copy()
    dx[:, 2] = xl[:, 2] + xr[:, 2] - 2 * xz[:, 2]

    # element_data = self.xdat['elements'][self.ion]['charges'][self.charge]
    # iondata = element_data['symmetries'][self.symm]['experiments']['XAS']['edges']
    # edge = iondata['L2,3 (2p)']['axes'][0][4]

    output = {
        'zpol_energy': xz[:, 0] + edge,
        'zpol_xas': xz[:, 2],
        'xas_left_energy': xl[:, 0] + edge,
        'xas_left': xl[:, 2],
        'xas_right_energy': xr[:, 0] + edge,
        'xas_right': xr[:, 2],
        'average_energy': xas[:, 0] + edge,
        'average': xas[:, 2] / 3,
        'xmcd_energy': mcd[:, 0] + edge,
        'xmcd': mcd[:, 2],
    }
    return output


def treat_output(Rawout: subprocess.CompletedProcess):
    """
    From the standard output of a Quanty calculation with the XAS_Template,
    it extracts the relevant expctation value

    Arguments:
        Rawout   : a subprocess CompltedProcess object

    Returns:
        A dictionary with the relevant expectation values

    """
    out = Rawout.stdout.split('\n')
    rline = 0
    for iline in range(len(out)):
        if '!*!' in out[iline]:
            # print(out[iline-2])
            # print(out[iline])
            rline = iline

    Odata = out[rline].split()
    values = {
        'E': float(Odata[2]),
        'S2': float(Odata[3]),
        'L2': float(Odata[4]),
        'J2': float(Odata[5]),
        'S_k': float(Odata[6]),
        'L_k': float(Odata[7]),
        'J_k': float(Odata[8]),
        'T_k': float(Odata[9]),
        'LdotS': float(Odata[10]),
        'Seff': float(Odata[6]) + float(Odata[9])
    }
    return values


"""
Atomic & X-Ray parameters
"""

import os
import json

PARAMETERS_FILE = os.path.dirname(__file__) + '/xray_parameters.json'

# x-ray parameters
with open(PARAMETERS_FILE, 'r', encoding='utf-8') as f:
    XRAY_DATA = json.load(f)

# AVAILABLE_SYMMETRIES = {
#     el: {
#         ch: list(charge['symmetries'].keys()) for ch, charge in element['charges'].items()
#     } for el, element in XRAY_DATA['elements'].items()
# }

# Atomic parameters
ATOMIC_PARAMETERS = {
    "Cu": {
        "Nelec": 9,
        "zeta_3d": 0.102,
        "F2dd": 12.854,
        "F4dd": 7.980,
        "zeta_2p": 13.498,
        "F2pd": 8.177,
        "G1pd": 6.169,
        "G3pd": 3.510,
        "Xzeta_3d": 0.124,
        "XF2dd": 13.611,
        "XF4dd": 8.457
    },
    "Ni": {
        "Nelec": 8,
        "zeta_3d": 0.083,
        "F2dd": 12.233,
        "F4dd": 7.597,
        "zeta_2p": 11.507,
        "F2pd": 7.720,
        "G1pd": 5.783,
        "G3pd": 3.290,
        "Xzeta_3d": 0.102,
        "XF2dd": 13.005,
        "XF4dd": 8.084
    },
    "Co": {
        "Nelec": 7,
        "zeta_3d": 0.066,
        "F2dd": 11.604,
        "F4dd": 7.209,
        "zeta_2p": 9.748,
        "F2pd": 7.259,
        "G1pd": 5.394,
        "G3pd": 3.068,
        "Xzeta_3d": 0.083,
        "XF2dd": 12.395,
        "XF4dd": 7.707
    },
    "Fe": {
        "Nelec": 6,
        "zeta_3d": 0.052,
        "F2dd": 10.965,
        "F4dd": 6.815,
        "zeta_2p": 8.200,
        "F2pd": 6.792,
        "G1pd": 5.000,
        "G3pd": 2.843,
        "Xzeta_3d": 0.067,
        "XF2dd": 11.778,
        "XF4dd": 7.327
    },
    "Mn": {
        "Nelec": 5,
        "zeta_3d": 0.040,
        "F2dd": 10.315,
        "F4dd": 6.413,
        "zeta_2p": 6.846,
        "F2pd": 6.320,
        "G1pd": 4.603,
        "G3pd": 2.617,
        "Xzeta_3d": 0.053,
        "XF2dd": 11.154,
        "XF4dd": 6.942
    },
     "Cr": {
        "Nelec": 4,
        "zeta_3d": 0.030,
        "F2dd": 9.648,
        "F4dd": 6.001,
        "zeta_2p": 5.668,
        "F2pd": 5.840,
        "G1pd": 4.201,
        "G3pd": 2.387,
        "Xzeta_3d": 0.041,
        "XF2dd": 10.521,
        "XF4dd": 6.551
    },
    "V": {
        "Nelec": 3,
        "zeta_3d": 0.022,
        "F2dd": 8.961,
        "F4dd": 5.576,
        "zeta_2p": 4.650,
        "F2pd": 5.351,
        "G1pd": 3.792,
        "G3pd": 2.154,
        "Xzeta_3d": 0.031,
        "XF2dd": 9.875,
        "XF4dd": 6.152
    },
    "Ti": {
        "Nelec": 2,
        "zeta_3d": 0.016,
        "F2dd": 8.243,
        "F4dd": 5.132,
        "zeta_2p": 3.776,
        "F2pd": 4.849,
        "G1pd": 3.376,
        "G3pd": 1.917,
        "Xzeta_3d": 0.023,
        "XF2dd": 9.21,
        "XF4dd": 5.744
    },
    "Sc": {
        "Nelec": 1,
        "zeta_3d": 0.010,
        "F2dd": 0,
        "F4dd": 0,
        "zeta_2p": 3.032,
        "F2pd": 4.332,
        "G1pd": 2.950,
        "G3pd": 1.674,
        "Xzeta_3d": 0.017,
        "XF2dd": 8.530,
        "XF4dd": 5.321
    }
}


# Determine which symmetries are available for each element and charge
AVAILABLE_SYMMETRIES = {
    el: {
        ch: list(charge['symmetries'].keys()) for ch, charge in XRAY_DATA['elements'][el]['charges'].items()
    } for el in ATOMIC_PARAMETERS
}

# Determine which edges are available for each element
AVAILABLE_EDGES = {
    el: next(
        charge['symmetries']['Oh']['experiments']['XAS']['edges']['L2,3 (2p)']['axes'][0][4] 
        for charge in XRAY_DATA['elements'][el]['charges'].values()
    )
    for el in ATOMIC_PARAMETERS
}


def get_configurations(element: str, charge: str) -> tuple[str, str]:
    """
    Get atomic configurations for a given element and charge.

    Args:
        element (str): Element symbol.
        charge (str): Charge state.

    Returns:
        dict: Atomic configurations.
    """
    Nelec = ATOMIC_PARAMETERS[element]['Nelec']
    if str(int(charge[0]) - 2) != 0:
        conf = '3d' + str(Nelec - int(charge[0]) + 2)
        conf_xas = '2p5,3d' + str(Nelec - int(charge[0]) + 2 + 1)
    else:
        conf = '3d' + str(Nelec)
        conf_xas = '2p5,3d' + str(Nelec + 1)
    return conf, conf_xas


# Dq values for each symmetry
"""
Oh
    10Dq -> 10Dq(3d)
D3h
    Dmu -> Dμ(3d)
    Dnu -> Dν(3d)
D4h
    Dq -> Dq(3d)
    Ds -> Ds(3d)
    Dt -> Dt(3d)
Td
    Dq -> Dq(3d)
C3v
    Dq -> Dq(3d)
    Ds -> Ds(3d)
    Dtau -> Dτ(3d)
"""
# XRAY_DATA['elements'][element]['charges'][charge]['configurations'][conf]['terms']
#  ['Crystal Field']['symmetries'][symmetry]['parameters']['variable']
DQ_STRINGS = {
    # xray_data label: input label
    '10Dq(3d)': '10Dq',
    'Dμ(3d)': 'Dmu',
    'Dν(3d)': 'Dnu',
    'Dq(3d)': 'Dq',
    'Ds(3d)': 'Ds',
    'Dt(3d)': 'Dt',
    'Dσ(3d)': 'Dsigma',
    'Dτ(3d)': 'Dtau',
}
DQ_CONVERT = {
    # input label: xray_data label
    DQ_STRINGS[k]: k for k in DQ_STRINGS
}


def get_Dq_values(element: str, charge: str, symmetry: str) -> dict[str, dict[str, float]]:
    """
    Get Dq values for a given element, charge and symmetry.

    Args:
        element (str): Element symbol, e.g. 'Cu'.
        charge (str): Charge state, e.g. '2+'.
        symmetry (str): Symmetry. ['Oh', 'Td', 'D4h', 'D3d', 'C4v', 'C3v']

    Returns:
        dict: Dq values for the given element, charge and symmetry as dict.
        Dq = {
            'initial': {'D?': value, ...},
            'final': {'D?': value, ...}
        }
    """
    conf, conf_xas = get_configurations(element, charge)
    confs = XRAY_DATA['elements'][element]['charges'][charge]['configurations']
    ini_parameters = confs[conf]['terms']['Crystal Field']['symmetries'][symmetry]['parameters']['variable']
    fnl_parameters = confs[conf_xas]['terms']['Crystal Field']['symmetries'][symmetry]['parameters']['variable']
    parameters = {
        'initial': {
            DQ_STRINGS[k] + '_i': ini_parameters[k] for k in ini_parameters
        },
        'final': {
            DQ_STRINGS[k] + '_f': fnl_parameters[k] for k in fnl_parameters
        }
    }
    return parameters


# Determine Dq values for each symmetry at each charge state for each element
AVAILABLE_DQ = {
    el: {
        ch: {
            sym: get_Dq_values(el, ch, sym) for sym in charge['symmetries']
        } for ch, charge in XRAY_DATA['elements'][el]['charges'].items()
    } for el in ATOMIC_PARAMETERS
}

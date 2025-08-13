"""
Authors: Joshua Willwerth, Shibo Tan, Abrar Rauf
Last Modified: August 12, 2025
Description: This script is designed for the thermodynamic modeling of two-component systems.
It provides tools for fitting the non-ideal mixing parameters of the liquid phase from T=0K DFT-calculated phases and
digitized equilibrium phase boundary data. The data stored and produced may be visualized using the BLPlotter class
GitHub: https://github.com/willwerj 
ORCID: https://orcid.org/0009-0004-6334-9426
"""
from __future__ import annotations

import math
import time
import numbers
import numpy as np
import pandas as pd
import sympy as sp

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
import plotly.graph_objects as go
from itertools import combinations
from io import StringIO
from pymatgen.core import Composition

from gliquid_ternary_interpolation.phase_diagram import PDPlotter, PhaseDiagram, PDEntry  # Note that the PMG PDPlotter source code has been modified
import gliquid_ternary_interpolation.load_binary_data as lbd
from gliquid_ternary_interpolation.hsx import HSX

_x_step = 0.01  # Sets composition grid precision; has not been tested for values other than 0.01
_x_prec = len(str(_x_step).split('.')[-1])
_x_vals = np.arange(0, 1 + _x_step, _x_step)
tau = 8000

# Define base thermodynamic symbols
# These can be used by the BinaryLiquid class to construct specific expressions
xb_sym, t_sym, a_sym, b_sym, c_sym, d_sym = sp.symbols('x t a b c d')
_DEFAULT_LINEAR_PARAMS = [0, 0, 0, 0]  # Default linear parameters
_DEFAULT_EXP_PARAMS = [0, 10000, 0, 10000]  # Default exponential parameters
_DEFAULT_PARAM_GUESSES = {'L0_a': [2000, 10000], 'L0_b': [0, 5], 'L1_a': [-10000, 0, 10000], 'L1_b': [-5, 0, 5],
                          'tau': [6000, 8000, 8000]}

def build_init_triangle(param_format: str = 'linear') -> list[list[float]]:
    if param_format in ['linear', 'combined']:
        guess_keys = ['L0_b', 'L1_b']
    elif param_format == 'exponential':
        guess_keys = ['tau', 'tau']
    elif param_format == 'combined_no_1S':
        guess_keys = ['L0_b', 'L1_a']
    elif param_format == 'pseudo':
        guess_keys = ['L0_a', 'L1_a']
    else:
        raise ValueError(f"'param_format' must be either 'linear', 'exponential', 'combined', or 'combined_no_1S'.")

    pos_0 = _DEFAULT_PARAM_GUESSES[guess_keys[0]]
    pos_1 = _DEFAULT_PARAM_GUESSES[guess_keys[1]]
    return [[pos_0[0], pos_1[1]], [pos_0[-1], pos_1[0]], [pos_0[-1], pos_1[-1]]] # [min, mid], [max, min], [max, max]
    # return [[pos_0[0], pos_1[0]], [pos_0[0], pos_1[-1]], [pos_0[-1], pos_1[1]]] # [min, min], [min, max], [max, mid]
    # TODO: reimplement for linear model
    
R = 8.314  # J/(mol*K), universal gas constant

def linear_expr(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    return a + b * t_sym

def exponential_expr(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    return a * sp.exp(-t_sym / b)

def combined_expr(a: sp.Expr, b: sp.Expr, c: sp.Expr) -> sp.Expr:
    return exponential_expr(linear_expr(a, b), c)

_L0_LINEAR_EXPR = linear_expr(a_sym, b_sym)
_L1_LINEAR_EXPR = linear_expr(c_sym, d_sym)
_L0_EXP_EXPR = exponential_expr(a_sym, b_sym)
_L1_EXP_EXPR = exponential_expr(c_sym, d_sym)
_L0_LIN_EXP_EXPR = combined_expr(a_sym, b_sym, sp.Integer(tau))
_L1_LIN_EXP_EXPR = combined_expr(c_sym, d_sym, sp.Integer(tau))

def build_thermodynamic_expressions(param_format: str = 'linear',
                                    ga_expr: sp.Symbol = 0*t_sym,
                                    gb_expr: sp.Symbol =  0*t_sym) -> dict[str, sp.Expr]:
    """
    Builds a dictionary of thermodynamic Sympy expressions based on provided base expressions.

    Args:
        param_format (str): Format of the non-ideal mixing parameters.
        ga_expr: Expression for the referenced Gibbs free energy of component A.
        gb_expr: Expression for the referenced Gibbs free energy of component B.

    Returns:
        dict[str, sp.Expr]: A dictionary mapping equation names to their Sympy expressions.
    """

    if param_format == 'linear':
        l0_expr, l1_expr = _L0_LINEAR_EXPR, _L1_LINEAR_EXPR
    elif param_format == 'exponential':
        l0_expr, l1_expr = _L0_EXP_EXPR, _L1_EXP_EXPR
    elif param_format in ['combined', 'combined_no_1S']:
        l0_expr, l1_expr = _L0_LIN_EXP_EXPR, _L1_LIN_EXP_EXPR
    else:
        raise ValueError(f"'param_format' must be either 'linear', 'exponential', 'combined', or 'combined_no_1S'.")

    xa = 1 - xb_sym

    # Ideal mixing Gibbs energy
    g_ideal = R * t_sym * (xa * sp.log(xa) + xb_sym * sp.log(xb_sym))

    # Excess Gibbs energy (2-term Redlich-Kister polynomial)
    # g_xs = L0 * (xa * xb) + L1 * (xa * xb * (xa - xb))
    # Factored form: xa * xb * (L0 + L1 * (xa - xb))
    g_xs = xa * xb_sym * (l0_expr + l1_expr * (xa - xb_sym))
    
    # Total Gibbs energy of liquid phase
    g_liquid = ga_expr * xa + gb_expr * xb_sym + g_ideal + g_xs
    
    # Entropy of liquid phase: S = - (dG/dT)_P,x
    s_liquid = -sp.diff(g_liquid, t_sym)
    s_l0 = -sp.diff(l0_expr, t_sym)
    
    # Enthalpy of liquid phase: H = G + TS = G - T*(dG/dT)_P,x
    # Using s_liquid = - (dG/dT), H = g_liquid + t * s_liquid
    h_liquid = g_liquid + t_sym * s_liquid
    h_l0 = l0_expr + t_sym * s_l0
    
    # First derivative of G_liquid with respect to xb
    g_prime = sp.diff(g_liquid, xb_sym)
    
    # Second derivative of G_liquid with respect to xb
    g_double_prime = sp.diff(g_prime, xb_sym)

    return {
        'ga': ga_expr,
        'gb': gb_expr,
        'l0': l0_expr,
        'l1': l1_expr,
        'g_ideal': g_ideal,
        'g_xs': g_xs,
        's_l0': s_l0,
        's_l0_lambdified': sp.lambdify([t_sym, a_sym, b_sym], s_l0, modules='numpy'),
        'h_l0': h_l0,
        'h_l0_lambdified': sp.lambdify([t_sym, a_sym, b_sym], h_l0, modules='numpy'),
        'g_liquid': g_liquid,
        's_liquid': s_liquid,
        's_liq_lambdified': sp.lambdify([xb_sym, t_sym, a_sym, b_sym, c_sym, d_sym], s_liquid, modules='numpy'),
        'h_liquid': h_liquid,
        'h_liq_lambdified': sp.lambdify([xb_sym, t_sym, a_sym, b_sym, c_sym, d_sym], h_liquid, modules='numpy'),
        'g_prime': g_prime,
        'g_double_prime': g_double_prime,
    }


def validate_binary_mixing_parameters(input, param_format='linear') -> list[int | float]:
    """
    Args:
        input (list[int | float]): A list containing numerical values representing non-ideal mixing parameters.

    Returns:
        list[int | float]: A validated list of four numerical values representing non-ideal mixing parameters.
    """
    default = _DEFAULT_EXP_PARAMS if param_format == 'exponential' else _DEFAULT_LINEAR_PARAMS
    if isinstance(input, (list, tuple)):
        if len(input) == 0:
            return default
        if all(isinstance(item, numbers.Number) and not isinstance(item, bool) for item in input) and len(input) == 4:
            if param_format == 'exponential' and any(item <= 0 for item in [input[1], input[3]]):
                raise ValueError("Critical temperature parameters L0_b and L1_b must be positive!")
            return [float(i) for i in input]  # Creates a copy of the input parameter list
    raise ValueError("Parameters must be input as a list or tuple in the following format: [L0_a, L0_b, L1_a, L1_b]")

class BinaryLiquid:
    """
    Represents a binary liquid system for thermodynamic modeling and phase diagram generation.

    Attributes:
        init_error (bool): Flag indicating if an error occurred during initialization.
        sys_name (str): Binary system name.
        components (list): List of component names.
        component_data (dict): Thermodynamic data for components.
        mean_elt_tm (float): Mean elemental melting temperature.
        pd_ind (int | None): Index of the MPDS phase diagram used.
        mpds_json (dict): MPDS phase equilibrium data for the system.
        digitized_liq (list): Digitized liquidus data points.
        max_liq_temp (float | None): Maximum temperature on the liquidus line.
        min_liq_temp (float | None): Minimum temperature on the liquidus line.
        temp_range (list): Temperature range for calculations.
        comp_range_fit_lim (float): Composition range limit for fitting.
        ignored_comp_ranges (list): Ignored composition ranges.
        dft_type (str): Functional used for DFT calculations.
        dft_ch (PhaseDiagram): DFT convex hull data formatted with pymatgen.
        phases (list): List of phase data.
        _params (list): Liquid non-ideal mixing parameters [L0_a, L0_b, L1_a, L1_b].
        _param_format (str): Formalism used for non-ideal mixing parameters (e.g., 'linear', 'exponential').
        eqs (dict): Dictionary of thermodynamic Sympy expressions.
        invariants (list): Identified invariant points.
        guess_symbols (list): Sympy symbols for corresponding to guessed parameters.
        constraints (list): Sympy equations used to store parameter constraints.
        init_triangle (np.ndarray): Initial simplex for Nelder-Mead optimization.
        nmpath (np.ndarray): Nelder-Mead optimization path, stored after running Nelder-Mead for plotting purposes.
        hsx (HSX): HSX object for phase diagram calculations.
    """

    def __init__(self, sys_name: str, components: list, init_error=False, **kwargs):
        self.init_error = init_error
        self.sys_name = sys_name
        self.components = components
        self.component_data = kwargs.get('component_data', {})
        self.mean_elt_tm = np.mean([self.component_data[comp][1] for comp in self.components])
        self.pd_ind = kwargs.get('pd_ind', None)
        self.mpds_json = kwargs.get('mpds_json', {})
        self.digitized_liq = kwargs.get('digitized_liq', [])
        self.max_liq_temp = max(self.digitized_liq, key=lambda x: x[1])[1] if self.digitized_liq else None
        self.min_liq_temp = min(self.digitized_liq, key=lambda x: x[1])[1] if self.digitized_liq else None
        self.temp_range = kwargs.get('temp_range', [])
        self.comp_range_fit_lim = kwargs.get('comp_range_fit_lim', 0.3)
        self.ignored_comp_ranges = kwargs.get('ignored_comp_ranges', [])
        self.dft_type = kwargs.get('dft_type', "GGA")
        self.dft_ch = kwargs.get('dft_ch', None)
        self.phases = kwargs.get('phases', [])
        self._params = kwargs.get('params', [0, 0, 0, 0])
        self._param_format = kwargs.get('param_format', 'linear')
        self.init_triangle = kwargs.get('init_triangle', build_init_triangle(self._param_format))
        self.eqs = kwargs.get('eqs', build_thermodynamic_expressions(self._param_format))
        self.invariants = None
        self.guess_symbols = None
        self.constraints = None
        self.nmpath = None
        self.hsx = None


    @classmethod
    def from_cache(cls, input, dft_type='GGA', pd_ind=None, params=[], param_format='linear', reconstruction=False, 
                   comp_range_fit_lim=0.3) -> BinaryLiquid:
        """
        Initializes a BinaryLiquid object from cached data.

        Args:
            input (any): Binary system - can be either a list or hyphenated string
            dft_type (str): Type of DFT calculation.
            pd_ind (int | None): Index of MPDS binary phase diagram data in cache or from API call downloads
            params (list): Initial fitting parameters.
            param_format (str): Format of the excess mixing energy params, either 'linear', 'exponential', or 'combined'.
            reconstruction (bool): Flag for liquidus reconstruction from prediction - 
                uses cached elemental data for melting points and referenced entropy instead of digitized liquidus.

        Returns:
            BinaryLiquid: Initialized BinaryLiquid object.
        """
        components, sys_name, order_changed = lbd.validate_and_format_binary_system(input)
        params = validate_binary_mixing_parameters(params, param_format=param_format)
        if order_changed: # Flip L1 parameters if the order of components has been changed and parameters were input
            if param_format != 'exponential':
                params[2:] = [-1 * p for p in params[2:]] 
            else:
                params[2] *= -1

        ch, _ = lbd.get_dft_convexhull(components, dft_type)
        phases = []
        for entry in ch.stable_entries:
            composition = entry.composition.fractional_composition.as_dict().get(components[1], 0)
            phase = {
                'name': entry.name,
                'comp': composition,
                'points': [],
                'energy': 96485 * ch.get_form_energy_per_atom(entry),
            }
            phases.append(phase)
        phases.sort(key=lambda x: x['comp'])

        hull_points = np.array([[p['comp'], p['energy']] for p in phases])

        phases.append({'name': 'L', 'points': []})

        mpds_json, component_data, (digitized_liq, is_partial) = lbd.load_mpds_data(components, pd_ind=pd_ind)
        if not reconstruction and not is_partial and digitized_liq:
            component_data[components[0]][1] = digitized_liq[0][1]
            component_data[components[-1]][1] = digitized_liq[-1][1]

        if 'temp' in mpds_json:
            temp_range = [mpds_json['temp'][0] + 273.15, mpds_json['temp'][1] + 273.15]
        else:
            comp_tms = [component_data[comp][1] for comp in components]
            temp_range = [min(comp_tms) - 50, max(comp_tms) * 1.1 + 50]

        eqs = build_thermodynamic_expressions(
            param_format=param_format,
            ga_expr=component_data[components[0]][0] - \
                t_sym * component_data[components[0]][0] / component_data[components[0]][1],
            gb_expr=component_data[components[1]][0] - \
                t_sym * component_data[components[1]][0] / component_data[components[1]][1])
        
        eqs['h_hull_interp'] = np.interp(_x_vals[1:-1], hull_points[:, 0], hull_points[:, 1])
        init_triangle = build_init_triangle(param_format=param_format)

        comp_range = mpds_json.get('comp_range', [0, 100]) # Need to transform liquidus x when comp_range is partial?
        comp_range = [max(min(digitized_liq, key=lambda x: x[0])[0], comp_range[0]/100.0), 
                      min(max(digitized_liq, key=lambda x: x[0])[0], comp_range[1]/100.0)] if digitized_liq else comp_range

        if not digitized_liq or (comp_range[1] - comp_range[0]) < comp_range_fit_lim:
            return cls(sys_name, components, True, mpds_json=mpds_json, component_data=component_data,
                       temp_range=temp_range, dft_type=dft_type, dft_ch=ch, phases=phases, params=params, 
                       param_format=param_format, eqs=eqs, pd_ind=pd_ind, init_triangle=init_triangle,
                       comp_range_fit_lim=comp_range_fit_lim)

        return cls(sys_name, components, False, mpds_json=mpds_json, component_data=component_data,
                   digitized_liq=digitized_liq, temp_range=temp_range, dft_type=dft_type, dft_ch=ch, phases=phases,
                   params=params, param_format=param_format, eqs=eqs, pd_ind=pd_ind, init_triangle=init_triangle,
                   comp_range_fit_lim=comp_range_fit_lim)
    

    def to_HSX(self, fmt="dict") -> dict | pd.DataFrame:
        """
        Converts phase data into HSX format for further calculations.

        Args:
            fmt (str): Output format ('dict' or 'dataframe').

        Returns:
            dict | pd.DataFrame: Data in HSX format.
        """
        lambda_args_vals = [_x_vals[1:-1], self.mean_elt_tm, self.get_L0_a(), self.get_L0_b(), self.get_L1_a(), self.get_L1_b()]
        liq_h_vals = self.eqs['h_liq_lambdified'](*lambda_args_vals).flatten().tolist()
        liq_s_vals = self.eqs['s_liq_lambdified'](*lambda_args_vals).flatten().tolist()

        data = {
            'X': [x for x in _x_vals],
            'S': [
            0 if self.component_data[self.components[0]][0] == 0 else self.component_data[self.components[0]][0] 
            / self.component_data[self.components[0]][1]
            ] + liq_s_vals + [
            0 if self.component_data[self.components[1]][0] == 0 else self.component_data[self.components[1]][0] 
            / self.component_data[self.components[1]][1]
            ],
            'H': [self.component_data[self.components[0]][0]] + liq_h_vals + [self.component_data[self.components[1]][0]],
            'Phase Name': ['L'] * len(_x_vals)
        }

        for phase in self.phases:
            if phase['name'] == 'L':
                continue
            data['H'].append(phase['energy'])
            data['S'].append(0)
            data['X'].append(round(phase['comp'], _x_prec))
            data['Phase Name'].append(phase['name'])

        if fmt == "dict":
            return data
        if fmt == "dataframe":
            return pd.DataFrame(data)
        else:
            raise ValueError("kwarg 'fmt' must be either 'dict' or 'dataframe'!")
        
    def update_phase_points(self) -> dict:
        """
        Calculates the phase points for given parameter values using the HSX class.

        This method converts phase data into the HSX form and uses HSX code to calculate the liquidus
        and low-temperature DFT phase boundaries.

        Returns:
            data (dict): A dictionary containing the phase data in HSX format, including phase names and components.
        """
        data = self.to_HSX()
        hsx_dict = {
            'data': data,
            'phases': [phase['name'] for phase in self.phases],
            'comps': self.components
        }
        self.hsx = HSX(hsx_dict, [self.temp_range[0] - 273.15, self.temp_range[-1] - 273.15])
        phase_points = self.hsx.get_phase_points()
        for phase in self.phases:
            phase['points'] = phase_points[phase['name']]
        return data

    def get_L0_a(self) -> int | float:
        return self._params[0]

    def get_L0_b(self) -> int | float:
        return self._params[1]

    def get_L1_a(self) -> int | float:
        return self._params[2]

    def get_L1_b(self) -> int | float:
        return self._params[3]
    
    def get_params(self) -> list[int | float]:
        """
        Get a copy of the current parameters such that the BinaryLiquid object will not be modified accidentally.
        
        Returns:
            list: A list of non-ideal mixing parameters in the following format: [L0_a, L0_b, L1_a, L1_b]
        """
        return [p for p in self._params]
    
    def update_params(self, input) -> None:
        """
        Update the non-ideal mixing parameters with validity checks, 
        then recalculate phase boundaries for the new parameters.

        Args:
            input (list[int | float]): A list containing numerical values representing non-ideal mixing parameters.

        Returns:
            None
        """
        self._params = validate_binary_mixing_parameters(input, self._param_format)
        self.update_phase_points()

    def find_invariant_points(self, verbose=False, check_full_ss=True, t_tol=15) -> tuple[list[dict], list[dict]]:
        """
        Identifies invariant points in the MPDS data using the provided MPDS JSON and liquidus data.

        This function does not consider DFT phases, which may differ in composition from the MPDS data. It requires both 
        complete liquidus and JSON data for a binary system.

        Args:
            verbose (bool): If True, outputs additional debugging information.
            t_tol (int): Temperature tolerance for invariant point identification.
            check_full_ss (bool): If True, checks for full composition range solid solutions.

        Returns:
            tuple: A tuple containing two lists:
                - List of identified invariant points.
                - List of low-temperature phases from the MPDS JSON.
        """
        if self.mpds_json['reference'] is None:
            print("System JSON does not contain any data!\n")
            return []

        # Identify phases from MPDS JSON
        phases = lbd.identify_mpds_phases(self.mpds_json)
        invariants = [phase for phase in phases if phase['type'] == 'mig']  # Miscibility gaps are not phases. 
        # They are also not really 'invariant points' either but we classify them as such for algorithm purposes.

        # Filter low-temperature phases
        mpds_lowt_phases = [
            phase for phase in phases
            if (
                phase['type'] in ['lc', 'ss'] and
                phase['tbounds'][0][1] < (self.mpds_json['temp'][0] + 273.15) +
                (self.mpds_json['temp'][1] - self.mpds_json['temp'][0]) * 0.10
            ) or '(' in phase['name']
        ]

        # Terminal output to visualize phase mismatch
        if verbose:
            mpds_phases_strs = [" "] * len(_x_vals)
            mp_phases_strs = [" "] * len(_x_vals)
            # Note: this implementation will probably break if _x_step is modified. Happy debugging!
            for phase in mpds_lowt_phases:
                if 'cbounds' in phase:
                    min_c_ind = int(phase['cbounds'][0][0] * 100)
                    max_c_ind = min(int(phase['cbounds'][1][0] * 100 + 1), len(_x_vals))
                    mpds_phases_strs[min_c_ind:max_c_ind] = ["|"] * (max_c_ind - min_c_ind)
                else:
                    mpds_phases_strs[min(int(phase['comp']*100), len(_x_vals))] = "|"
            for phase in self.phases:
                if phase['name'] not in ['L'] + self.components:
                    mp_phases_strs[min(int(phase['comp']*100), len(_x_vals))] = "|"
            print("\n--- Low temperature phase mismatch ---")
            print("MPDS:", "[" + "".join(mpds_phases_strs) + "]")
            print("MP:  ", "[" + "".join(mp_phases_strs) + "]")
            print("COMP:", " 0" + " "*9 + "10" + " "*8 + "20" + " "*8 + "30" + " "*8 + "40" + " "*8 + "50" + " "*8 +
                   "60" + " "*8 + "70" + " "*8 + "80" + " "*8 + "90" + " "*8 + "100")

        if verbose:
            print('--- Low temperature phases including component solid solutions ---')
            for phase in mpds_lowt_phases:
                print(phase)

        # Identify full composition solid solutions
        phase_labels = [label[0] for label in self.mpds_json['labels']]
        ss_label = f"({self.components[0]}, {self.components[1]})"
        ss_label_inv = f"({self.components[1]}, {self.components[0]})"
        ss_labels = [
            ss_label, f"{ss_label} ht", f"{ss_label} rt",
            ss_label_inv, f"{ss_label_inv} ht", f"{ss_label_inv} rt"
        ]
        full_comp_ss = bool([label for label in phase_labels if label in ss_labels])
        if full_comp_ss and check_full_ss:
            print('Solidus processing not implemented!')
            self.init_error = True
            return invariants, mpds_lowt_phases

        def find_local_minima(points):
            """
            Args:
                points (list of tuples): List of (x, y) points.

            Returns:
                list: Local minima points.
            """
            def is_lt_prev(index):
                return index > 0 and points[index][1] < points[index - 1][1]

            local_minima = []
            current_section = []

            for i in range(len(points)):
                if is_lt_prev(i):
                    current_section = [points[i]]
                elif current_section and current_section[-1][1] == points[i][1]:
                    current_section.append(points[i])
                elif current_section:
                    local_minima.append(current_section[len(current_section) // 2])
                    current_section = []

            return local_minima

        def find_local_maxima(points):
            """
            Args:
                points (list of tuples): List of (x, y) points.

            Returns:
                list: Local maxima points.
            """
            def is_gt_prev(index):
                return index > 0 and points[index][1] > points[index - 1][1]

            local_maxima = []
            current_section = []

            for i in range(len(points)):
                if is_gt_prev(i):
                    current_section = [points[i]]
                elif current_section and current_section[-1][1] == points[i][1]:
                    current_section.append(points[i])
                elif current_section:
                    local_maxima.append(current_section[len(current_section) // 2])
                    current_section = []

            return local_maxima

        # Locate maxima and minima in liquidus
        maxima = find_local_maxima(self.digitized_liq)
        minima = find_local_minima(self.digitized_liq)

        # Assign congruent melting points
        if mpds_lowt_phases:
            for coords in maxima[:]:
                mpds_lowt_phases.sort(key=lambda x: abs(x['comp'] - coords[0]))
                phase = mpds_lowt_phases[0]
                if (
                    phase['type'] in ['lc', 'ss'] and
                    abs(phase['comp'] - coords[0]) <= 0.02 and
                    phase['tbounds'][1][1] + t_tol >= coords[1]
                ):
                    phase['type'] = 'cmp'
                    invariants.append({
                        'type': phase['type'],
                        'comp': phase['comp'],
                        'temp': phase['tbounds'][1][1],
                        'phases': [phase['name']],
                        'phase_comps': [phase['comp']]
                    })
                    maxima.remove(coords)

        # Sort by descending temperature for peritectic identification
        mpds_lowt_phases.sort(key=lambda x: x['tbounds'][1][1], reverse=True)

        def find_adj_phases(point: list | tuple) -> tuple[dict, dict]:
            """
            Finds adjacent phases near a given point.

            Args:
                point (list | tuple): A point in composition-temperature space.

            Returns:
                tuple: Two nearest adjacent phases.
            """
            all_lowt_phases = (
                mpds_lowt_phases +
                [
                    {'name': self.components[0], 'comp': 0, 'type': 'lc',
                        'tbounds': [[], [0, self.component_data[self.components[0]][1]]]},
                    {'name': self.components[1], 'comp': 1, 'type': 'lc',
                        'tbounds': [[], [1, self.component_data[self.components[1]][1]]]},
                ]
            )
            all_lowt_phases = [p for p in all_lowt_phases if p['tbounds'][1][1] + t_tol >= point[1]]
            lhs_phases = [phase for phase in all_lowt_phases if phase['comp'] < point[0]]
            adj_lhs_phase = None if not lhs_phases else min(lhs_phases, key=lambda x: abs(x['comp'] - point[0]))
            rhs_phases = [phase for phase in all_lowt_phases  if phase['comp'] > point[0]]
            adj_rhs_phase = None if not rhs_phases else min(rhs_phases, key=lambda x: abs(x['comp'] - point[0]))
            return adj_lhs_phase, adj_rhs_phase

        # Identify liquid-liquid miscibility gap labels
        misc_gap_labels = []
        for label in self.mpds_json['labels']:
            delim_label = label[0].split(' ')
            if len(delim_label) == 3 and delim_label[0][0] == 'L' and delim_label[2][0] == 'L':
                misc_gap_labels.append([label[1][0] / 100.0, label[1][1] + 273.15])

        # Process miscibility gap labels and find the nearest two-phase region which each corresponds to
        for mgl in misc_gap_labels:
            if len(maxima) < 1:
                break
            nearest_maxima = min(maxima, key=lambda x: abs(x[0] - mgl[0]))

            tbounds = [None, nearest_maxima]
            cbounds = None
            phases = None
            phase_comps = None

            for shape in self.mpds_json['shapes']:
                if shape['nphases'] != 2:
                    continue
                data = lbd.shape_to_list(shape['svgpath'])
                if not data:
                    continue
                data.sort(key=lambda x: x[1])
                if not (abs(data[-1][1] - nearest_maxima[1]) < t_tol and abs(data[-1][0] - nearest_maxima[0]) < 0.05):
                    continue
                tbounds = [data[0], data[-1]]
                data.sort(key=lambda x: x[0])
                if not data[0][0] < nearest_maxima[0] < data[-1][0]:
                    continue
                cbounds = [data[0], data[-1]]
                break

            if len(minima) >= 1:
                # Adjacent monotectic should be a minima point at minimum distance in x-t space from the misc gap dome
                adj_mono = min(
                    minima,
                    key=lambda x: abs(tbounds[1][0] - x[0]) + 2 * (abs(tbounds[1][1] - x[1]) / self.temp_range[1])
                )
                tbounds[0] = [tbounds[1][0], adj_mono[1]]
                adj_phases = find_adj_phases(adj_mono)

                if adj_mono[0] < tbounds[1][0]:
                    if adj_phases[0] is not None:
                        phase_comps = [adj_phases[0]['comp']]
                        phases = [adj_phases[0]['name']]
                    if not cbounds:
                        lhs_ind = self.digitized_liq.index(adj_mono)
                        for i in range(lhs_ind + 1, len(self.digitized_liq) - 1):
                            if self.digitized_liq[i + 1][1] < adj_mono[1] <= self.digitized_liq[i][1]:
                                m = ((self.digitized_liq[i + 1][1] - self.digitized_liq[i][1]) /
                                        (self.digitized_liq[i + 1][0] - self.digitized_liq[i][0]))
                                rhs_comp = (adj_mono[1] - self.digitized_liq[i][1]) / m + self.digitized_liq[i][0]
                                cbounds = [adj_mono, [rhs_comp, adj_mono[1]]]
                                break
                elif adj_mono[0] > tbounds[1][0]:
                    if adj_phases[1] is not None:
                        phase_comps = [adj_phases[1]['comp']]
                        phases = [adj_phases[1]['name']]
                    if not cbounds:
                        rhs_ind = self.digitized_liq.index(adj_mono)
                        for i in reversed(range(1, rhs_ind - 1)):
                            if self.digitized_liq[i - 1][1] < adj_mono[1] <= self.digitized_liq[i][1]:
                                m = ((self.digitized_liq[i - 1][1] - self.digitized_liq[i][1]) /
                                        (self.digitized_liq[i - 1][0] - self.digitized_liq[i][0]))
                                lhs_comp = (adj_mono[1] - self.digitized_liq[i][1]) / m + self.digitized_liq[i][0]
                                cbounds = [[lhs_comp, adj_mono[1]], adj_mono]
                                break
                if cbounds and cbounds[0][1] != cbounds[1][1]:
                    cbounds[0][1] = adj_mono[1]
                    cbounds[1][1] = adj_mono[1]
                minima.remove(adj_mono)

            if cbounds:
                invariants.append({
                    'type': 'mig',
                    'comp': tbounds[1][0],
                    'cbounds': cbounds,
                    'tbounds': tbounds,
                    'phases': phases,
                    'phase_comps': phase_comps
                })
                maxima.remove(nearest_maxima)
            break

        stable_phase_comps = []

        # Main loop for peritectic phase identification
        for phase in mpds_lowt_phases:
            if '(' in phase['name']:  # Ignore component SS phases
                continue

            # Congruent melting points will not be considered for peritectic formation but will limit comp search range
            if phase['type'] == 'cmp':
                stable_phase_comps.append(phase['comp'])
                continue

            sections, current_section = [], []
            phase_temp = phase['tbounds'][1][1]

            for i in range(len(self.digitized_liq) - 1):
                liq_point, next_liq_point = self.digitized_liq[i], self.digitized_liq[i + 1]
                liq_temp, next_liq_temp = liq_point[1], next_liq_point[1]

                # Liquidus point is above or equal to phase temp
                if liq_temp >= phase_temp:
                    current_section.append(liq_point)
                    if next_liq_temp >= phase_temp and i + 1 == len(self.digitized_liq) - 1:
                        current_section.append(next_liq_point)
                        sections.append(current_section)

                # Liquidus point is first point below phase temp
                elif current_section:
                    if abs(phase_temp - current_section[-1][1]) > abs(phase_temp - liq_temp):
                        current_section.append(liq_point)  # Add to section if closer to phase temp
                    sections.append(current_section)  # End section
                    current_section = []

                # Next liquidus point is above phase temp
                elif next_liq_temp >= phase_temp > liq_temp:
                    if abs(phase_temp - next_liq_temp) > abs(phase_temp - liq_temp):
                        current_section.append(liq_point)  # Add if below phase temp and closer than next point

            # Find endpoints of liquidus segments excluding the component ends
            endpoints = [
                section[i]
                for section in sections
                for i in [0, -1]
                if section[i] not in [self.digitized_liq[0], self.digitized_liq[-1]]
            ]

            # Filter endpoints if there exists a stable phase between the current phase and the liquidus
            for comp in stable_phase_comps:
                endpoints = [
                    ep for ep in endpoints
                    if abs(comp - ep[0]) > abs(phase['comp'] - ep[0])
                    or abs(comp - phase['comp']) > abs(phase['comp'] - ep[0])
                ]

            # Sort by increasing distance to liquidus to find the shortest distance
            endpoints.sort(key=lambda x: abs(x[0] - phase['comp']))

            # Take the closest liquidus point to the phase as the peritectic point
            if endpoints:
                invariants.append({
                    'type': 'per',
                    'comp': endpoints[0][0],
                    'temp': phase_temp,
                    'phases': [phase['name']],
                    'phase_comps': [phase['comp']]
                })

            stable_phase_comps.append(phase['comp'])

        # Identify eutectic points
        for coords in minima:
            adj_phases = find_adj_phases(coords)
            phases, phase_comps = zip(*[
                (None, None) if phase is None else (phase['name'], phase['comp'])
                for phase in adj_phases
            ])

            invariants.append({
                'type': 'eut',
                'comp': coords[0],
                'temp': coords[1],
                'phases': list(phases),
                'phase_comps': list(phase_comps)
            })

        invariants.sort(key=lambda x: x['comp'])
        invariants = [inv for inv in invariants if inv['type'] not in ['lc', 'ss']]
        if verbose:
            print('--- Identified invariant points ---')
            for inv in invariants:
                print(inv)
            print()
        return invariants, mpds_lowt_phases

    def solve_params_from_constraints(self, guessed_vals: dict) -> None:
        """
        Updates the parameters of the object based on guessed values and constraints.

        Args:
            guessed_vals (dict): A dictionary containing guessed values for the parameters.
        """
        for ind, symbol in enumerate([a_sym, b_sym, c_sym, d_sym]):
            try:
                if symbol in guessed_vals:
                    self._params[ind] = float(guessed_vals[symbol])
                elif self.constraints:
                    self._params[ind] = float(self.constraints[symbol].subs(guessed_vals))
            except TypeError:
                raise RuntimeError("Error in constraint equations!")

    def obeys_lupis_elliott(self, tol=1e-6) -> bool:
        """
        Checks if the liquidus parameters obey the Lupis-Elliott sign constraints.

        Args:
            tol (float): A small tolerance for floating-point comparisons.

        Returns:
            bool: True if the parameters obey the Lupis-Elliott sign constraints, False otherwise.
        """

        lambda_args_vals = [0, self.get_L0_a(), self.get_L0_b()] # Per Kaptay, T=0K is stricter than self.mean_elt_tm
        h_l0 = self.eqs['h_l0_lambdified'](*lambda_args_vals)
        s_l0 = self.eqs['s_l0_lambdified'](*lambda_args_vals)
        return (h_l0 <= tol and s_l0 <= tol) or (h_l0 >= -tol and s_l0 >= -tol)

    def h0_below_ch(self, tol=1e-6) -> bool:
        """
        Checks if the liquid enthalpy curve at T=0K falls below the solid convex hull.

        Args:
            tol (float): Minimum distance that liquid enthalpy curve must be above the solid convex hull.

        Returns:
            bool: True if any part of the liquid enthalpy curve falls below the solid
                  convex hull, False otherwise.
        """
        # Calculate the liquid enthalpy at T=0K.
        lambda_args_vals = [_x_vals[1:-1], 0, self.get_L0_a(), self.get_L0_b(), self.get_L1_a(), self.get_L1_b()]
        h_vals_0k = self.eqs['h_liq_lambdified'](*lambda_args_vals)
        return np.any(h_vals_0k < self.eqs['h_hull_interp'] + tol)
    
    def liquidus_is_continuous(self, tol=2 * _x_step) -> bool:
        """
        Checks if the liquidus line is continuous within a given tolerance.

        Args:
            tol (float): Tolerance for liquidus continuity. Default is twice the step size.

        Returns:
            bool: True if the generated liquidus line is compositionally-continuous, False otherwise.
        """
        last_coords = None
        for coords in self.phases[-1]['points']:
            if last_coords and coords[0] - last_coords[0] > tol:
                return False
            last_coords = coords
        return True
    
    def calculate_deviation_metrics(self, num_points=30, **kwargs) -> tuple[float, float]:
        """
        Calculates the deviation metrics between the digitized (measured) liquidus and the generated liquidus.

        Args:
            num_points (int): Number of points to sample for deviation metrics. Default is 30.
            **kwargs: Additional keyword arguments:
                - ignored_ranges (bool): If True, ignores the composition ranges specified in self.ignored_comp_ranges.
                - small_range (bool): If True, allows for small composition ranges in the generated points.

        Returns:
            tuple: Mean absolute error (MAE) and root mean square error (RMSE).
        """
        # Convert liquidus data to numpy arrays
        digitized = np.array(self.digitized_liq)
        generated = np.array(self.phases[-1]['points'])

        # Filter out endpoint compositions (x=0 and x=1)
        mask = (digitized[:, 0] != 0) & (digitized[:, 0] != 1)
        if not np.any(mask):
            return float('inf'), float('inf')
        digitized = digitized[mask]

        if len(digitized) > 1: # Composition range limits imposed by input liquidus data
            digitized_liq_lims = [digitized[0, 0], digitized[-1, 0]] 
        else: # 1 point only limit
            digitized_liq_lims = [digitized[0, 0] - 5 * _x_step, digitized[0, 0] + 5 * _x_step]

        # Composition range limits imposed by ignored ranges
        if kwargs.get('ignored_ranges', True) and self.ignored_comp_ranges: 
            mask = np.ones_like(generated[:, 0], dtype=bool)
            for lower, upper in self.ignored_comp_ranges:
                mask = mask & ((generated[:, 0] < lower) | (generated[:, 0] > upper))
            generated = generated[mask] 
            ignored_comp_lims = [generated[0, 0], generated[-1, 0]]
        else:
            ignored_comp_lims = [0, 1]

        x_min = max(_x_step, digitized_liq_lims[0], ignored_comp_lims[0])
        x_max = min(1 - _x_step, digitized_liq_lims[1], ignored_comp_lims[1])
        if x_max - x_min < self.comp_range_fit_lim:
            print(f"Error: Large composition range filtered out (remaining range = {[x_min, x_max]})")
            return float('inf'), float('inf')

        # Reduce the composition range of the generated points to the composition range of digitized points
        generated = generated[(generated[:, 0] >= digitized[0, 0]) & (generated[:, 0] <= digitized[-1, 0])]
        if not kwargs.get('small_range', False) and len(generated) <= num_points:
            print("Error: Not enough generated points for accurate deviation metrics calculation.")
            return float('inf'), float('inf')

        # Determine x-coordinates for evaluation not exceeding the number of generated points
        x_mesh = np.linspace(x_min, x_max, min(num_points, len(generated) + 1))

        # Find closest temperature values for each evaluation point
        Y1 = np.array([digitized[np.argmin(np.abs(digitized[:, 0] - x)), 1] for x in x_mesh])
        Y2 = np.array([generated[np.argmin(np.abs(generated[:, 0] - x)), 1] for x in x_mesh])
        diffs = np.abs(Y1 - Y2)
        return float(np.mean(diffs)), float(np.sqrt(np.mean(diffs**2)))
    
    def f(self, guess: list | tuple, **kwargs) -> float:
        """
        Objective function for parameter fitting.

        Args:
            guess (list | tuple): Guessed parameter values to evaluate.
            **kwargs (dict): Additional keyworded arguments to dictate which constraints are applied.

        Returns:
            float: Generated liquidus mean absolute error (MAE) for the given parameter values.
        """
        guess_dict = {symbol: guess for symbol, guess in zip(self.guess_symbols, guess)}
        # Check if the parameter guesses are mathematically valid
        if self._param_format == 'exponential':
            if any(guess_dict.get(sym, 1) <= 0 for sym in [b_sym, d_sym]):
                print(f'Parameter sign constraint violated for params {self.get_params()}')
                return float('inf')
        # Solve for non-guessed parameter values from constraints
        self.solve_params_from_constraints(guess_dict) 
        # Check if the parameters are physically valid
        if kwargs.get('check_lupis_elliott', True) and not self.obeys_lupis_elliott():
            print(f'Lupis-Elliott sign constraint violated for params {self.get_params()}')
            return float('inf')
        if kwargs.get('check_h0_below_ch', True) and self.h0_below_ch():
            print(f'T=0K enthalpy constraint violated for params {self.get_params()}')
            return float('inf')
        # Update HSX object and generate new phase points
        try:
            self.update_phase_points()
        except (ValueError, TypeError) as e:
            print(e)
            return float('inf')
        # Check if generated liquidus is continuous
        if kwargs.get('check_liquidus_continuity', True) and not self.liquidus_is_continuous():
            print(f'Liquidus continuity constraint violated for guess {self.get_params()}')
            return float('inf')
        # Evaluate the liquidus temperature deviation metrics
        mae, _ = self.calculate_deviation_metrics(**kwargs)
        return mae

    def nelder_mead(self, max_iter=64, tol=0.05, verbose=False, 
                    initial_guesses=[], **kwargs) -> tuple[float, float, np.ndarray]:
        """
        Nelder-Mead algorithm for fitting the liquid non-ideal mixing parameters.

        Args:
            max_iter (int): Maximum number of iterations. Default is 64.
            tol (float): Tolerance for algorithm convergence. Default is 0.05.
            verbose (bool): If True, print updates during optimization. Default is False.
            initial_guesses (list): Reasonable initial values for guessed parameters, determined by self.guess_symbols.
            **kwargs (dict): Additional keyworded arguments passed to f().

        Returns:
            tuple: MAE, RMSE, and Nelder-Mead optimization path.
        """
        if not initial_guesses:
            initial_guesses = self.init_triangle

        # Initial guesses for parameters
        x0 = np.array(initial_guesses, dtype=float)
        self.nmpath = np.empty((3, 5, max_iter), dtype=float)
        initial_time = time.time()

        print("--- Beginning Nelder-Mead optimization ---")

        for i in range(max_iter):
            start_time = time.time()
            if verbose:
                print("Iteration #", i)

            f_vals = np.empty(x0.shape[0])
            param_vals = np.empty((x0.shape[0], 4))
            for idx, x in enumerate(x0):
                f_vals[idx] = self.f(x, **kwargs) # 3 f() calls
                param_vals[idx] = self.get_params()
            self.nmpath[:, :-1, i] = param_vals
            self.nmpath[:, -1, i] = f_vals
            iworst = np.argmax(f_vals)
            ibest = np.argmin(f_vals)

            # Check if all current simplex vertices are invalid
            if iworst == ibest:
                self.nmpath = self.nmpath[:, :, :i]
                if i == 0:
                    raise RuntimeError("Nelder-Mead algorithm is unable to find physical parameter values.")
                else: # Revert to last valid simplex
                    print("--- Nelder-Mead stopped after %s seconds ---" % (time.time() - initial_time))
                    best_recent_idx = np.argmin(self.nmpath[:, -1, i-1])
                    self.f(self.nmpath[best_recent_idx, :-1, i-1], **kwargs)
                    mae, rmse = self.calculate_deviation_metrics(**kwargs)
                    print("Mean temperature deviation per point between liquidus curves =", mae, '\n')
                    return mae, rmse, self.nmpath[:, :, :i]

            centroid = np.mean(x0[f_vals != f_vals[iworst]], axis=0)
            xreflect = centroid + 1.0 * (centroid - x0[iworst, :])
            f_xreflect = self.f(xreflect, **kwargs) # 1 f() call

            # Simplex reflection step
            if f_vals[iworst] <= f_xreflect < f_vals[2]:
                x0[iworst, :] = xreflect
            # Simplex expansion step
            elif f_xreflect < f_vals[ibest]:
                xexp = centroid + 2.0 * (xreflect - centroid)
                if self.f(xexp, **kwargs) < f_xreflect: # 1 f() call
                    x0[iworst, :] = xexp
                else:
                    x0[iworst, :] = xreflect
            # Simplex contraction step
            else:
                if f_xreflect < f_vals[2]:
                    xcontract = centroid + 0.5 * (xreflect - centroid)
                    if self.f(xcontract, **kwargs) < self.f(x0[iworst, :], **kwargs): # 2 f() calls
                        x0[iworst, :] = xcontract
                    else:  # Simplex shrink step
                        x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                        [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                        x0[iworst, :] = x0[imid, :] + 0.5 * (x0[imid, :] - x0[ibest, :])
                else:
                    xcontract = centroid + 0.5 * (x0[iworst, :] - centroid)
                    if self.f(xcontract, **kwargs) < self.f(x0[iworst, :], **kwargs): # 2 f() calls
                        x0[iworst, :] = xcontract
                    else:  # Simplex shrink step
                        x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                        [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                        x0[imid, :] = x0[ibest, :] + 0.5 * (x0[imid, :] - x0[ibest, :])

            if verbose:
                guess_dict = {symbol: guess for symbol, guess in zip(self.guess_symbols, x0[ibest, :])}
                print("Best guess:", guess_dict, f'f={f_vals[ibest]}')
                print("Height of triangle =", 2 * np.max(np.abs(x0 - centroid)))
                print("--- %s seconds elapsed ---" % (time.time() - start_time))

            # Convergence check
            if np.max(np.abs(x0 - centroid)) < tol:
                self.f(x0[ibest, :], **kwargs)
                print("--- Nelder-Mead converged in %s seconds ---" % (time.time() - initial_time))
                mae, rmse = self.calculate_deviation_metrics(**kwargs)
                print("Mean temperature deviation per point between liquidus curves =", mae, '\n')
                self.nmpath = self.nmpath[:, :, :i]
                return mae, rmse, self.nmpath
            
            if i >= 1: # Overexpansion limit - if MAE is not improving significantly, stop Nelder-Mead
                f_improvement = np.min(self.nmpath[:, -1, i-1])/self.nmpath[ibest, -1, i]
                if 1 < f_improvement < 1.00001:
                    print("--- Nelder-Mead stopped after %s seconds ---" % (time.time() - initial_time))
                    self.f(x0[ibest, :], **kwargs)
                    mae, rmse = self.calculate_deviation_metrics(**kwargs)
                    print("Mean temperature deviation per point between liquidus curves =", mae, '\n')
                    self.nmpath = self.nmpath[:, :, :i]
                    return mae, rmse, self.nmpath
                
        raise RuntimeError("Nelder-Mead algorithm did not converge within limit.")

    def fit_parameters(self, verbose=False, n_opts=1, t_tol=15, **kwargs) -> list[dict]:
        """
        Fit the liquidus non-ideal mixing parameters for a binary system.

        This function utilizes the Nelder-Mead algorithm to minimize the temperature deviation in the liquidus.

        Args:
            verbose (bool): If True, prints detailed progress and results.
            n_opts (int): Number of optimization attempts. Updates the BinaryLiquid object to reflect the lowest MAE fit
            t_tol (float): Temperature tolerance for invariant point identification.
            **kwargs: Additional keyword arguments for fitting options or arguments passed to nelder-mead.
                - ignore_euts (bool): If True, ignores eutectics as constraints if critical eutectic phases are missing.
                - ignored_ranges (bool): If True, ignores the composition ranges specified in self.ignored_comp_ranges.
                - small_range (bool): If True, allows for small composition ranges in the generated points.
                - check_full_ss (bool): If True, checks if the full solid solution is present in the system.
                - check_phase_mismatch (bool): If True, checks phase mismatch between invariant points and self.phases.
                - check_lupis_elliott (bool): If True, checks Lupis-Elliott sign constraints.
                - check_h0_below_ch (bool): If True, checks if the liquid enthalpy at T=0K is below the solid convex hull.
                - check_liquidus_continuity (bool): If True, checks if the generated liquidus is continuous.

        Returns:
            list[dict]: Parameter fitting data containing results of all optimization attempts.
        """

        if self.digitized_liq is None:
            print("System missing liquidus data! Ensure that 'BinaryLiquid.digitized_liq' is not empty!\n")
            return []
        
        def find_nearest_phase(composition, tol=0.02):
            sorted_phases = sorted(self.phases[:-1], key=lambda x: abs(x['comp'] - composition))
            nearest = sorted_phases[0]
            deviation = abs(nearest['comp'] - composition)
            if deviation > tol:
                return {}, deviation
            return nearest, deviation
        
        def phase_decomp_near_liq(phase, tol=10):
            for i in range(len(self.digitized_liq) - 1):
                if self.digitized_liq[i][0] == phase['tbounds'][1][0]:
                    return abs(self.digitized_liq[i][1] - phase['tbounds'][1][1]) < tol
                elif self.digitized_liq[i][0] < phase['tbounds'][1][0] < self.digitized_liq[i + 1][0]:
                    return abs((self.digitized_liq[i][1] + self.digitized_liq[i + 1][1]) / 2 - phase['tbounds'][1][1]) < tol

        # Find invariant points, can set self.init_error to True if system is identified to be isomorphous
        if self.invariants is None:
            self.invariants, mpds_lowt_phases = self.find_invariant_points(
                verbose=True, t_tol=t_tol, check_full_ss=kwargs.get('check_full_ss', True))
            # Low T phases that decompose near the liquidus line and aren't component solid solutions are critical
            critical_phases = [p for p in mpds_lowt_phases if '(' not in p['name'] and 
                               phase_decomp_near_liq(p, tol=(self.temp_range[1] - self.temp_range[0]) * 0.05)]
        
            # If over half of the low-temperature phases are not represented in DFT, the fit will likely not be the best
            dft_matching_phases = [p for p in critical_phases if find_nearest_phase(p['comp'], tol=0.04)[0]]
            if len(dft_matching_phases) < len(critical_phases)/2.0 and kwargs.get('check_phase_mismatch', True):
                self.init_error = True
        else:
            dft_matching_phases = []

        if self.init_error:
            return []

        # Compare invariant points to self.phases to assess solving conditions
        eqs = []
        for inv in self.invariants:
            if inv['type'] == 'mig':
                x1, t1 = inv['cbounds'][0]  # Bottom left of dome
                x2, t2 = inv['tbounds'][1]  # Top of dome
                x3, t3 = inv['cbounds'][1]  # Bottom right of dome

                eqn1 = sp.Eq(self.eqs['g_double_prime'].subs({xb_sym: x2, t_sym: t2}), 0)
                eqn4 = sp.Eq(self.eqs['g_prime'].subs({xb_sym: x1, t_sym: t1}), self.eqs['g_prime'].subs({xb_sym: x3, t_sym: t3}))

                eqs.append([f'mig - {round(x2, 2)}', '2nd order', t2, eqn1])
                eqs.append([f'mig - {round(x1, 2)}-{round(x3, 2)}', '1st order', t1, eqn4])

            if inv['type'] == 'cmp':
                if '(' in inv['phases'][0]:
                    if inv['comp'] < 0.5:
                        self.ignored_comp_ranges.append([0, inv['comp']])
                    elif inv['comp'] > 0.5:
                        self.ignored_comp_ranges.append([inv['comp'], 1])
                    continue

                nearest_phase, _ = find_nearest_phase(inv['comp'])
                if not nearest_phase:
                    continue

                x1, t1 = nearest_phase['comp'], inv['temp']
                eqn = sp.Eq(self.eqs['g_liquid'].subs({xb_sym: x1, t_sym: t1}), nearest_phase['energy'])
                eqs.append(['cmp', f'{round(x1, 2)} - 0th order', t1, eqn])

            if inv['type'] == 'per':
                if '(' in inv['phases'][0]:
                    if inv['phase_comps'][0] < inv['comp']:
                        self.ignored_comp_ranges.append([0, inv['comp']])
                    elif inv['phase_comps'][0] > inv['comp']:
                        self.ignored_comp_ranges.append([inv['comp'], 1])
                    continue

                per_phase, _ = find_nearest_phase(inv['phase_comps'][0], tol=0.04)
                if not per_phase:
                    continue

                x1, t1 = inv['comp'], inv['temp']
                x2, g2 = per_phase['comp'], per_phase['energy']

                eqn1 = sp.Eq(self.eqs['g_liquid'].subs({xb_sym: x1, t_sym: t1}) + self.eqs['g_prime'].subs({xb_sym: x1, t_sym: t1}) * (x2 - x1), g2)
                eqn2 = sp.Eq(self.eqs['g_liquid'].subs({xb_sym: x1, t_sym: t1}), g2)

                liq_point_at_phase = min(self.digitized_liq, key=lambda x: abs(x[0] - x2))
                temp_below_liq = liq_point_at_phase[1] - t1

                if temp_below_liq > t_tol:
                    eqs.append([f'per - {round(x1, 2)}', '0th order', t1, eqn1])
                else:
                    eqs.append([f'per - {round(x1, 2)}', 'pseudo 0th order', t1, eqn2])

            if inv['type'] == 'eut':
                if None in inv['phase_comps']:
                    continue

                lhs_phase, _ = find_nearest_phase(inv['phase_comps'][0], tol=0.04)
                rhs_phase, _ = find_nearest_phase(inv['phase_comps'][1], tol=0.04)

                if kwargs.get('ignore_euts', True):
                    invalid_eut = False
                    if not lhs_phase or lhs_phase['comp'] > inv['comp']:
                        self.ignored_comp_ranges.append([inv['phase_comps'][0], inv['comp']])
                        invalid_eut = True
                    elif '(' in inv['phases'][0] and inv['phase_comps'][0] > 0.05:
                        invalid_eut = True
                    if not rhs_phase or rhs_phase['comp'] < inv['comp']:
                        self.ignored_comp_ranges.append([inv['comp'], inv['phase_comps'][1]])
                        invalid_eut = True
                    elif '(' in inv['phases'][1] and inv['phase_comps'][1] < 0.95:
                        invalid_eut = True
                    if invalid_eut:
                        continue

                x1, g1 = lhs_phase['comp'], lhs_phase['energy']
                x2, t2 = inv['comp'], inv['temp']
                x3, g3 = rhs_phase['comp'], rhs_phase['energy']

                eqn1 = sp.Eq(self.eqs['g_prime'].subs({xb_sym: x2, t_sym: t2}), (g3 - g1) / (x3 - x1))
                eqn2 = sp.Eq(self.eqs['g_liquid'].subs({xb_sym: x2, t_sym: t2}) + 
                             self.eqs['g_liquid'].subs({xb_sym: x2, t_sym: t2}) * (x1 - x2), g1)
                eqn3 = sp.Eq(self.eqs['g_liquid'].subs({xb_sym: x2, t_sym: t2}) + 
                             self.eqs['g_liquid'].subs({xb_sym: x2, t_sym: t2}) * (x3 - x2), g3)

                eqs.append([f'eut - {round(x2, 2)}', '1st order', t2, eqn1])
                if g1 <= g3:
                    eqs.append([f'eut - {round(x2, 2)}', '0th order lhs', t2, eqn2])
                else:
                    eqs.append([f'eut - {round(x2, 2)}', '0th order rhs', t2, eqn3])


        # # Determine the sign of the enthalpy of mixing and modify default initial guesses accordingly
        # currently only used for linear model defult initial guesses, TODO: find a better way to do this
        self.update_params([])
        self.update_phase_points()
        # BLPlotter(self).show('fit+liq') # Show the current HSX object
        mid_digitized_liq = min(self.digitized_liq, key=lambda x: abs(x[0] - 0.5)) # closest to middle as possible
        nearest_generated_liq = min(self.phases[-1]['points'], key=lambda x: abs(x[0] - mid_digitized_liq[0]))
        l0_a_positive =  mid_digitized_liq[1] > nearest_generated_liq[1] # if ideal mixing liq falls below measured liq, L0_a should be positive
        if l0_a_positive and self._param_format == 'linear': # L0_b sign should be negative for initial guesses
            self.init_triangle = [[-p, q] for p, q in self.init_triangle] # TODO: check if this is best practice
        
        def get_hull_skew():
            hull_skew = (self.component_data[self.components[1]][0] - self.component_data[self.components[0]][0]) / 4.0
            for phase in self.phases[:-1]:
                hull_skew += (phase['comp'] - 0.5) * phase['energy']
            return hull_skew
    
        # Test invariant-derived equations for validity as constraints
        two_constr_methods = ['combined', 'exponential', 'linear']
        one_constr_methods = ['combined_no_1S']
        no1S_constr = ['no1S - L0_b = 0', '0th order', float('inf'), sp.Eq(d_sym, 0)] # this enforces L1_b = 0
        no_L0Sxs_eq = sp.Eq(sp.diff(self.eqs['l0'], t_sym).subs({t_sym: 0}), 0) # this will enforce L0_b != 0 for combined models, such that S0xs at 0K is 0
        no_L1Sxs_eq = sp.Eq(sp.diff(self.eqs['l1'], t_sym).subs({t_sym: 0}), 0) # this will enforce L1_b != 0 for combined models such that S1xs at 0K is 0
        nelder_mead_ics = []
        eqs = [eq for eq in eqs if not eq[3] == False] # Remove invalid equations
        if eqs: # 1 or more valid invariant-derived constraint equations
            highest_tm_eq = eqs.pop(eqs.index(max(eqs, key=lambda x: x[2])))
            self.guess_symbols = [b_sym, d_sym]

            for eq in eqs: # 2 or more valid invariant-derived constraint equations
                try:
                    self.constraints = sp.solve([eq[3], highest_tm_eq[3], no_L0Sxs_eq, no_L1Sxs_eq], (a_sym, b_sym, c_sym, d_sym), rational=False, simplify=False)
                    init_mae = self.f([], **kwargs)
                    if init_mae == float('inf'):
                        continue
                    elif self._param_format in two_constr_methods: # if combined, then should be nonzero? # if linear, this won't work
                        if self._param_format == 'linear':
                            init_tri = self.init_triangle
                        else:
                            init_tri = [[self.get_L0_b(), self.get_L1_b()],
                                        [self.get_L0_b()*0.8, self.get_L1_b()],
                                        [self.get_L0_b(), self.get_L1_b()*0.8]]
                        nelder_mead_ics.append({'mae': init_mae, 'constrs': [eq, highest_tm_eq], 'init_tri': init_tri})
                    elif self._param_format in one_constr_methods:
                        init_tri = [[self.get_L0_b(), self.get_L1_a()],
                                    [self.get_L0_b()*0.8, self.get_L1_a()],
                                    [self.get_L0_b(), self.get_L1_a()*0.8]]
                        nelder_mead_ics.append({'mae': init_mae, 'constrs': [eq, no1S_constr], 'init_tri': init_tri})
                except RuntimeError as e:
                    print("Error while evaluting invariant constraints", e)
                    continue
            
            # Create a constraint for the highest temperature equation
            if self._param_format in one_constr_methods: # TODO: check if this needs to be revised
                if nelder_mead_ics: # Choose the best init triangle determined by double constraints
                    min_mae_ics = min(nelder_mead_ics, key=lambda x: x['mae'])
                    nelder_mead_ics.append({'mae': min_mae_ics['mae'] - 1E-6, # Slightly lower to preference this choice
                                            'constrs': [highest_tm_eq, no1S_constr],
                                            'init_tri': min_mae_ics['init_tri']})
                else: # If only a single constraint is available, use default init triangle and determine init mae
                    try:
                        self.constraints = sp.solve([sp.Eq(c_sym, get_hull_skew() * 5), highest_tm_eq[3], 
                                                      no_L0Sxs_eq, no_L1Sxs_eq],
                                                     (a_sym, b_sym, c_sym, d_sym), rational=False, simplify=False)
                        init_mae = self.f([], **kwargs)
                        init_tri = [[self.get_L0_b(), self.get_L1_a()],
                                    [self.get_L0_b()*0.8, self.get_L1_a()],
                                    [self.get_L0_b(), self.get_L1_a()*0.8]]
                        if init_mae != float('inf'):
                            nelder_mead_ics.append({'mae': init_mae, 'constrs': [highest_tm_eq, no1S_constr], 'init_tri': init_tri})
                    except RuntimeError as e:
                        print("Error while evaluting invariant constraints", e)

        # Derive pseudo-constraints using Nelder-Mead for the enthalpy of mixing
        mean_liq_temp = (self.min_liq_temp + self.max_liq_temp) / 2
        print(f"\nMaximum composition range fitted: {[self.digitized_liq[0][0], self.digitized_liq[-1][0]]}")
        print(f"Ignored composition ranges: {self.ignored_comp_ranges}\n")

        def gen_la_triangle():
            
            def get_hull_midpoint_depth_from_h0():
                lhs_ref = self.dft_ch.get_hull_energy_per_atom(Composition({self.components[0]: 1}))
                rhs_ref = self.dft_ch.get_hull_energy_per_atom(Composition({self.components[1]: 1}))
                hull_mid_ref = self.dft_ch.get_hull_energy_per_atom(Composition({c: 0.5 for c in self.components}))
                e_depth = hull_mid_ref * 96485 - (lhs_ref * 96485 + self.component_data[self.components[0]][0] + 
                                                rhs_ref * 96485 + self.component_data[self.components[1]][0]) / 2 
                return e_depth
            
            skew_e = get_hull_skew()
            e_depth = get_hull_midpoint_depth_from_h0()
            no_stable_dft_compounds = len([p for p in self.phases if p['name'] not in self.components + ['L']]) == 0
        
            if no_stable_dft_compounds:
                l_guesses = [[e_depth * -2, skew_e * 5], # positive L0_a
                            [e_depth * -2, skew_e * -5], # positive L0_a
                            [e_depth * 2, skew_e * 5]] # negative L0_a
            else:
                l_guesses = [[0.0, 0.0], # ideal mixing
                        [e_depth, skew_e * -5], # negative L0_a
                        [e_depth * 2, skew_e * 5]] # more negative L0_a
  
            return l_guesses

        try:
            self.update_params([]) # Restore parameter defaults
            self.guess_symbols = [a_sym, c_sym]
            self.constraints = sp.solve([no_L0Sxs_eq, no_L1Sxs_eq], (b_sym, d_sym), rational=False, simplify=False)
            psuedo_triangle = gen_la_triangle()
            print("Initial triangle for pseudo-constraints:", psuedo_triangle)
            init_mae, _, _ = self.nelder_mead(tol=10, verbose=True, initial_guesses=psuedo_triangle, **kwargs)
            eq1 = sp.Eq(self.eqs['l0'].subs({t_sym: mean_liq_temp}),
                        self.eqs['l0'].subs({t_sym: mean_liq_temp, a_sym: self.get_L0_a(), b_sym: self.get_L0_b()}))
            
            if self._param_format in two_constr_methods:
                eq2 = sp.Eq(self.eqs['l1'].subs({t_sym: mean_liq_temp}),
                            self.eqs['l1'].subs({t_sym: mean_liq_temp, c_sym: self.get_L1_a(), d_sym: self.get_L1_b()}))
                if self._param_format == 'linear':
                    init_tri = self.init_triangle
                else:
                    init_tri = [[self.get_L0_b(), self.get_L1_b()],
                                [self.get_L0_b()*0.8, self.get_L1_b()],
                                [self.get_L0_b(), self.get_L1_b()*0.8]]
                nelder_mead_ics.append({'mae': init_mae,
                                        'constrs': [['pseudo', '0th order', mean_liq_temp, e] for e in [eq1, eq2]],
                                        'init_tri': init_tri})
            elif self._param_format in one_constr_methods:
                init_tri = [[self.get_L0_b(), self.get_L1_a()],
                            [self.get_L0_b()*0.8, self.get_L1_a()],
                            [self.get_L0_b(), self.get_L1_a()*0.8]]
                nelder_mead_ics.append({'mae': init_mae,
                                        'constrs': [['pseudo', '0th order', mean_liq_temp, eq1], no1S_constr],
                                        'init_tri': init_tri})
        except RuntimeError as e:
            print("Nelder-Mead process encountered a fatal error while deriving psuedo-constraints: ", e)

        # Sort by ascending initial MAE such that the 'best' constraints are used first (if limited on number of attempts)
        nelder_mead_ics.sort(key=lambda x: x['mae']) 
        self.guess_symbols = [b_sym, d_sym] if self._param_format not in one_constr_methods else [b_sym, c_sym]
        solve_symbols = [sym for sym in [a_sym, b_sym, c_sym, d_sym] if sym not in self.guess_symbols]
        fitting_data = []
        
        for i in range(n_opts):
            if not nelder_mead_ics:
                break
            selected_ics = nelder_mead_ics.pop(0)
            constrs_str = '/'.join([c[0] for c in selected_ics['constrs']])
            constr_algo = 'pseudo_constr' if constrs_str.startswith('pseudo') else 'inv_constr'
            if verbose:
                print(f"--- Nelder-Mead ICs Attempt #{i + 1} (initial mae = {round(selected_ics['mae'], 2)}) ---")
                for (source, order, temp, eq) in selected_ics['constrs']:
                    print(f"Source: {source}, Order: {order}, Temperature: {round(temp, 1)}, Equation: {eq}")
                print("Initial triangle:", selected_ics['init_tri'])

            selected_eqs = [eq[3] for eq in selected_ics['constrs']]
            self.constraints = sp.solve(selected_eqs, solve_symbols, rational=False, simplify=False)
            convergence_tol = 5 if self._param_format == 'exponential' else 5E-2 # Changed from 5E-1
            try:
                mae, rmse, path = self.nelder_mead(verbose=verbose, tol=convergence_tol,
                                                   initial_guesses=selected_ics['init_tri'], **kwargs)
            except RuntimeError as e:
                print("Nelder-Mead process encountered a fatal error: ", e)
                continue
            norm_mae = mae / self.max_liq_temp
            norm_rmse = rmse / self.max_liq_temp
            l0 = float(self.eqs['l0'].subs({t_sym: mean_liq_temp, a_sym: self.get_L0_a(), b_sym: self.get_L0_b()}))
            l1 = float(self.eqs['l1'].subs({t_sym: mean_liq_temp, c_sym: self.get_L1_a(), d_sym: self.get_L1_b()}))
            fit_invs = self.hsx.liquidus_invariants()[0]
            fitting_data.append({'mae': mae, 'rmse': rmse, 'norm_mae': norm_mae, 'norm_rmse': norm_rmse, 
                                 'constrs': constrs_str, 'algo': constr_algo, 'n_iters': path.shape[2], 'nmpath': path,
                                 'L0_a': self.get_L0_a(), 'L0_b': self.get_L0_b(), 'L1_a': self.get_L1_a(),
                                 'L1_b': self.get_L1_b(), 'L0': l0, 'L1': l1,
                                 'euts': fit_invs['Eutectics'], 'pers': fit_invs['Peritectics'],
                                 'cmps': fit_invs['Congruent Melting'], 'migs': fit_invs['Misc Gaps']})

        if fitting_data:
            best_fit = min(fitting_data, key=lambda x: x['mae'])
            self._params = [best_fit['L0_a'], best_fit['L0_b'], best_fit['L1_a'], best_fit['L1_b']]
            self.nmpath = best_fit['nmpath']
            self.update_phase_points()
        return fitting_data
    
class BLPlotter:
    """
    A plotting class for BinaryLiquid objects.

    This class contains methods to create various subfigures and visualizations for analyzing
    BinaryLiquid system data. It uses both static matplotlib and interactive Plotly plots.
    """

    def __init__(self, binaryliquid: BinaryLiquid, **plotkwargs):
        """
        Args:
            binaryliquid (BinaryLiquid): BinaryLiquid object containing the system data.
            plotkwargs (dict): Optional keyword arguments for plot customization (e.g., axis margins).
        """
        self._bl = binaryliquid
        self.plotkwargs = plotkwargs or {
            'axes': {'xmargin': 0.005, 'ymargin': 0}
        }

    def get_plot(self, plot_type: str, **kwargs) -> go.Figure | plt.Axes:
        """
        Generates the specified plot for the BinaryLiquid object.

        Args:
            plot_type (str): The type of plot to generate. Supported types include:
                - 'pc': Low-temperature phase comparison plot
                - 'ch', 'ch+g', 'vch': T=0K DFT convex hull plots
                - 'fit', 'fit+liq', 'pred', 'pred+liq': Generated phase diagram plots
                - 'nmp': Nelder-Mead path visualization plot
            kwargs: Additional keyword arguments for customization.

        Returns:
            go.Figure | plt.Axes: The generated plot object (Plotly or Matplotlib).
        """
        valid_plot_types = [
            'pc',
            'ch', 'ch+g', 'vch',
            'fit', 'fit+liq', 'pred', 'pred+liq',
            'nmp'
        ]
        if plot_type not in valid_plot_types:
            raise ValueError(f"Invalid plot type '{plot_type}'. Supported types: {valid_plot_types}")

        fig = None

        # Phase comparison plot
        if plot_type == 'pc':
            fig = self._generate_phase_comparison_plot()

        # Convex hull plots
        elif plot_type in ['ch', 'ch+g', 'vch']:
            fig = self._generate_convex_hull_plot(plot_type, **kwargs)

        # Liquidus fitting and prediction plots
        elif plot_type in ['fit', 'fit+liq', 'pred', 'pred+liq']:
            fig = self._generate_liquidus_fit_plot(plot_type, **kwargs)

        # Nelder-Mead path visualization
        elif plot_type == 'nmp':
            fig = self._generate_nelder_mead_path_plot(**kwargs)

        return fig

    def show(self, plot_type: str, **kwargs) -> None:
        """
        Displays the generated plot.

        Args:
            plot_type (str): The type of plot to generate. Supported types include:
                - 'pc': Low-temperature phase comparison plot
                - 'ch', 'ch+g', 'vch': T=0K DFT convex hull plots
                - 'fit', 'fit+liq', 'pred', 'pred+liq': Generated phase diagram plots
                - 'nmp': Nelder-Mead path visualization plot
            kwargs: Additional keyword arguments passed to `get_plot`.
        """
        fig = self.get_plot(plot_type, **kwargs)

        if isinstance(fig, go.Figure):
            fig.show()
        elif isinstance(fig, plt.Figure):
            fig.figure.show()

    def write_image(self, plot_type: str, stream: str | StringIO, image_format: str = "svg", **kwargs) -> None:
        """
        Saves the generated plot as an image.

        Args:
            plot_type (str): The type of plot to save.
            stream (str | StringIO): The file path or stream to save the image.
            image_format (str): The format of the image (default is 'svg').
            kwargs: Additional keyword arguments passed to `get_plot`.
        """
        fig = self.get_plot(plot_type, **kwargs)
        
        if isinstance(fig, go.Figure):
            if plot_type in ['ch', 'ch+g', 'vch']:
                fig.write_image(stream, format=image_format, width=480 * 1.8, height=300 * 1.7) # 960, 700?
            else:
                fig.write_image(stream, format=image_format)
        elif isinstance(fig, plt.Figure):
            fig.figure.savefig(stream, format=image_format)
            plt.close(fig)

    def _generate_phase_comparison_plot(self) -> plt.Figure:
        """
        Generates a phase comparison plot showing congruent and incongruent phases
        from MPDS and MP data. The plot consists of two subplots displaying phases
        in different temperature and magnitude ranges.

        Returns:
            plt.Figure: The generated phase comparison plot.
        """
        # Extract low-temperature phase data from MPDS and MP
        ([mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp],
        [mp_phases, mp_phases_ebelow, min_form_e]) = lbd.get_low_temp_phase_data(self._bl.mpds_json, self._bl.dft_ch)

        # Filter out phases containing parentheses
        mpds_congruent_phases = {key: value for key, value in mpds_congruent_phases.items() if '(' not in key}
        mpds_incongruent_phases = {key: value for key, value in mpds_incongruent_phases.items() if '(' not in key}

        # Create subplots with specific layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2), gridspec_kw={'hspace': 0})

        def plot_phases(ax, source, color, alpha=0.5):
            """
            Args:
                ax (matplotlib.axes.Axes): The axis to plot on.
                source (dict): Phase data with keys as phase names and values as bounds/magnitudes.
                color (str): Color for the phase fill.
                alpha (float): Transparency for the fill.
            """
            for _, ((lb, ub), mag) in source.items():
                # Ensure a minimum width for labeling
                if ub - lb < 0.026:
                    ave = (ub + lb) / 2
                    lb = ave - 0.013
                    ub = ave + 0.013
                ax.fill_betweenx([min(0, mag), max(0, mag)], lb, ub, color=color, alpha=alpha)
                ax.set_xlim(0, 1)
                ax.margins(x=0, y=0)

        # Plot phases for both subplots
        plot_phases(ax1, mpds_congruent_phases, 'blue')
        plot_phases(ax1, mpds_incongruent_phases, 'purple')
        plot_phases(ax2, mp_phases, 'orange')
        plot_phases(ax2, mp_phases_ebelow, 'red')

        # Check if MPDS phases exist
        mpds_phases = bool(mpds_congruent_phases or mpds_incongruent_phases)

        # Configure y-axis for the first subplot
        if mpds_phases:
            tick_range = np.linspace(0, max_phase_temp, 4)[1:]
            ax1.set_yticks(tick_range)
            ax1.set_yticklabels([format(tick, '.1e') for tick in tick_range])
            ax1.set_ylim(0, 1.1 * max_phase_temp)
        else:
            ax1.set_yticks([])

        ax1.set_ylabel('MPDS', fontsize=11, rotation=90, labelpad=5, fontweight='semibold')
        ax1.yaxis.set_label_position('right')
        ax1.set_xticks([])

        # Configure y-axis for the second subplot
        if mp_phases:
            tick_range = np.linspace(0, min_form_e, 4)
            ax2.set_yticks(tick_range)
            ax2.set_yticklabels([format(tick, '.1e') for tick in tick_range])
            ax2.set_ylim(1.1 * min_form_e, 0)
        elif mpds_phases:
            ax2.set_yticks([0])
            ax2.set_yticklabels([format(0, '.1e')])
            ax2.set_ylim(-1, 0)
        else:
            ax2.set_yticks([])

        ax2.set_ylabel('MP', fontsize=11, rotation=90, labelpad=5, fontweight='semibold')
        ax2.yaxis.set_label_position('right')
        ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax2.set_xticklabels([0, 20, 40, 60, 80, 100])

        # Add a title to the figure
        fig.suptitle('Low Temperature Phase Comparison', fontweight='semibold')

        return fig

    def _generate_convex_hull_plot(self, plot_type: str, **kwargs) -> go.Figure:
            """
            Generates a convex hull plot or phase diagram visualization.

            Args:
                plot_type (str): The type of plot to generate ('ch', 'vch', 'ch+g').
                kwargs: Additional arguments for customization, such as 't_vals' or 't_units' for temperature values.

            Returns:
                go.Figure: The generated Plotly figure.
            """
            if any(len(Composition(comp).as_dict()) > 1 for comp in self._bl.components):
                raise NotImplementedError("This feature is not presently supported for compound components")
            if plot_type == 'vch':
                # Generate volume-referenced convex hull
                ch, atomic_vols = lbd.get_dft_convexhull(self._bl.sys_name, self._bl.dft_type, inc_structure_data=True)

                new_entries = [PDEntry(
                    composition=e.composition,
                    energy=atomic_vols[e.composition.reduced_formula] * e.composition.num_atoms
                ) for e in ch.stable_entries]

                vch = PhaseDiagram(new_entries)
                pdp = PDPlotter(vch)
            else:
                # Use the standard convex hull
                pdp = PDPlotter(self._bl.dft_ch)

            if plot_type in ['ch', 'vch']:
                # Generate a basic convex hull plot
                fig = pdp.get_plot()

                if plot_type == 'vch':
                    fig.update_yaxes(title={'text': 'Referenced Atomic Volume (Å^3/atom)'})

            elif (not self._bl.component_data or not self._bl.digitized_liq) and 't_vals' not in kwargs:
                # Handle uninitialized BinaryLiquid data
                print("BinaryLiquid object phase diagram not initialized! Returning plot without liquid energy")
                fig = pdp.get_plot()

            else:
                # Generate convex hull plot with liquidus curves
                t_vals = kwargs.get('t_vals', [])
                if not isinstance(t_vals, list) or not all(isinstance(t, (int, float)) for t in t_vals):
                    raise ValueError("kwarg 't_vals' must be a list of valid temperatures, either as ints or floats!")
                t_units = kwargs.get('t_units', 'C')
                if not t_units or not isinstance(t_units, str) or t_units not in ['C', 'K']:
                    raise ValueError("kwarg 't_units' must be a string, either 'C' for Celsius or 'K' for Kelvin")
                if t_units and not t_vals:
                    print("No arguments specified for 't_vals', setting 't_units' to 'K'")

                max_phase_temp = 0
                if not t_vals:
                    t_units = 'K'

                    # Estimate temperature range from liquidus data
                    asc_temp = sorted(self._bl.digitized_liq, key=lambda x: x[1])
                    mpds_phases = lbd.identify_mpds_phases(self._bl.mpds_json)

                    if mpds_phases:
                        max_phase_temp = max(mpds_phases, key=lambda x: x['tbounds'][1][1])['tbounds'][1][1]
                    else:
                        max_phase_temp = asc_temp[0][1]
                if t_units == 'C':
                    t_vals = [t + 273.15 for t in t_vals if t >= 0]
                else:
                    t_vals = [t for t in t_vals if t >= 0]

                params = self._bl._params

                def get_g_curve(A=0, B=0, C=0, D=0, T=0, name="") -> go.Scatter:
                    """
                    Args:
                        A, B, C, D (float): Non-ideal mixing parameters.
                        T (float): Temperature in Kelvin.

                    Returns:
                        go.Scatter: Plotly scatter trace for the Gibbs free energy curve.
                    """
                    g = self._bl.eqs['g_liquid'].subs({t_sym: T, a_sym: A, b_sym: B, c_sym: C, d_sym: D})
                    gliq_vals = sp.lambdify(xb_sym, g, 'numpy')(_x_vals[1:-1]) if g.has(xb_sym) else [0] * len(_x_vals[1:-1])
                    ga = np.float64(self._bl.eqs['ga'].subs({t_sym: T}) / 96485)
                    gb = np.float64(self._bl.eqs['gb'].subs({t_sym: T}) / 96485)
                    if name == "":
                        name += "Ideal " if A == 0 and C == 0 else ""
                        name += "Liquid T=" + str(int(T)) + "K" if t_units == 'K' else str(int(T-273.15)) + "C"

                    return go.Scatter(
                        x=_x_vals,
                        y=[ga] + [g / 96485 for g in gliq_vals] + [gb],
                        mode='lines',
                        name=name
                    )

                # Generate liquidus curves
                if t_vals:
                    traces = [get_g_curve(A=params[0], B=params[1], C=params[2], D=params[3], T=temp)
                            for temp in reversed(t_vals)]
                else:
                    traces = [
                        # get_g_curve(B=1E-10, D=1E-10, T=max_phase_temp),
                        get_g_curve(A=params[0], B=params[1], C=params[2], D=params[3], T=max_phase_temp),
                        # get_g_curve(B=1E-10, D=1E-10),
                        get_g_curve(A=params[0], B=params[1], C=params[2], D=params[3]),
                    ]

                # Add traces to convex hull plot
                fig = pdp.get_plot(data=traces)

            fig.update_layout(plot_bgcolor="white",
                              paper_bgcolor="white", 
                              xaxis=dict(title=dict(text="Composition (fraction)")),
                              width=960, height=700)
            return fig

    def _generate_liquidus_fit_plot(self, plot_type: str) -> go.Figure:
        """
        Generates liquidus fitting and prediction plots.

        Args:
            plot_type (str): The type of liquidus plot to generate ('fit', 'fit+liq', 'pred', 'pred+liq').

        Returns:
            go.Figure: The generated plot object.
        """

        # Initialize variables for liquidus lines and gas temperature
        gas_temp = None

        # Check if the plot type includes the MPDS liquidus
        if plot_type in ['fit+liq', 'pred+liq'] and not self._bl.digitized_liq:
            print("Digitized_liquidus is not initialized! Returning plot without digitized liquidus")

        # Determine if prediction is required
        pred_pd = bool(plot_type in ['pred', 'pred+liq'])

        # If predicted phase diagram, calculate the minimum gas temperature as an advisory for liquidus accuracy
        if pred_pd:
            gas_temp = min([cd[2] for cd in self._bl.component_data.values()])

        # Ensure phase points are updated if not already done
        if self._bl.hsx is None:
            self._bl.update_phase_points()

        # Generate the plot using the HSX plot method
        fig = self._bl.hsx.plot_tx(
            digitized_liquidus=self._bl.digitized_liq if plot_type in ['fit+liq', 'pred+liq'] else None,
            pred=pred_pd,  # Determines generated liquidus color and temperature axis scaling
            gas_temp=gas_temp  # Include gas temperature if applicable
        )

        return fig

    def _generate_nelder_mead_path_plot(self, **kwargs) -> plt.Figure:
        """
        Generates a visualization of the Nelder-Mead optimization path.

        This method plots the progression of the Nelder-Mead optimization algorithm in the parameter
        space, using triangles to represent each iteration and color coding for iterations and errors.
        To use, BinaryLiquid object field 'nmpath' must be initialized.

        Returns:
            plt.Figure: The generated plot figure.
        """
        if self._bl.nmpath is None:
            raise ValueError("Underlying BinaryLiquid object has no Nelder-Mead path! Generate using `fit_parameters`")
        plot_a_params = kwargs.get('plot_a_params', False)
        fig, ax = plt.subplots(figsize=(8, 5))
        num_iters = self._bl.nmpath.shape[2]

        # Determine the range of temperature deviations (tdev_range)
        tdev_range = [None, None]
        for i in range(num_iters):
            if plot_a_params: # L0_a, L1_a parameters
                path_i = self._bl.nmpath[:, [0, 2, -1], i]
            elif self._bl._param_format == 'combined_no_1S': # L0_b, L1_a parameters
                path_i = self._bl.nmpath[:, [1, 2, -1], i]
            else: # L0_b, L1_b parameters
                path_i = self._bl.nmpath[:, [1, 3, -1], i]

            t_devs = [num for num in path_i[:, -1:] if num != float('inf')]
            if t_devs:
                tdev_range[0] = min(t_devs) if tdev_range[0] is None else min(tdev_range[0], min(t_devs))
                tdev_range[1] = max(t_devs) if tdev_range[1] is None else max(tdev_range[1], max(t_devs))

        # Triangle color mapping (iteration-based)
        sm1 = cm.ScalarMappable(cmap=cm.get_cmap('winter'), norm=LogNorm(vmin=1, vmax=num_iters))
        triangle_colors = sm1.to_rgba(np.arange(1, num_iters + 1, 1))
        ticks = [2 ** exp for exp in np.arange(0, math.ceil(np.log2(num_iters)), 1)]
        cbar1 = fig.colorbar(sm1, ax=ax, aspect=14)
        cbar1.minorticks_off()
        cbar1.set_ticks(ticks)
        cbar1.set_ticklabels(ticks)
        cbar1.set_label('Nelder-Mead Iteration', style='italic', labelpad=8, fontsize=12)

        # Marker color mapping (temperature deviation-based)
        sm2 = cm.ScalarMappable(cmap=cm.get_cmap('autumn'), norm=plt.Normalize(tdev_range[0], tdev_range[1]))
        marker_colors = sm2.to_rgba(np.arange(tdev_range[0], tdev_range[1], 1))
        cbar2 = fig.colorbar(sm2, ax=ax, aspect=14)
        cbar2.set_label(
            f"MAE From MPDS Liquidus ({chr(176)}C)",
            style='italic',
            labelpad=10,
            fontsize=12
        )

        plotted_points = []

        for i in range(num_iters):
            if plot_a_params: # L0_a, L1_a parameters
                path_i = self._bl.nmpath[:, [0, 2, -1], i]
            elif self._bl._param_format == 'combined_no_1S': # L0_b, L1_a parameters
                path_i = self._bl.nmpath[:, [1, 2, -1], i]
            else: # L0_b, L1_b parameters
                path_i = self._bl.nmpath[:, [1, 3, -1], i]

            triangle = path_i[:, :-1]  # Extract triangle vertices
            t_devs = path_i[:, -1:]  # Extract temperature deviations

            # Plot triangles connecting vertices
            coordinates = [triangle[j, :] for j in range(triangle.shape[0])]
            pair_combinations = list(combinations(coordinates, 2))
            for combo in pair_combinations:
                line = np.array(combo)
                ax.plot(
                    line[:, 0], line[:, 1],
                    color=triangle_colors[i],
                    linewidth=(2 - 1.7 * (i / num_iters)),
                    zorder=0
                )

            # Plot markers at triangle vertices
            for point, t_dev in zip(triangle, t_devs):
                if list(point) in plotted_points:
                    continue
                if t_dev != float('inf'):
                    c_ind = int(t_dev - tdev_range[0])
                    marker_color = marker_colors[c_ind]
                    ax.scatter(
                        point[0],
                        point[1],
                        s=(55 - 54.7 * (i / num_iters)),
                        color=marker_color,
                        marker='^',
                        edgecolor='black',
                        linewidth=0.3,
                        zorder=1
                    )
                else:
                    ax.scatter(
                        point[0],
                        point[1],
                        s=(45 - 44.7 * (i / num_iters)),
                        color='black',
                        label='Incalculable MAE',
                        marker='^',
                        zorder=1
                    )
                plotted_points.append(list(point))

        # Add legend and adjust axis labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

        # Adjust axis limits for better scaling
        ax.autoscale()
        ly, uy = ax.get_ylim()
        ax.set_ylim((uy + ly) / 2 - (uy - ly) / 2 * 1.1, (uy + ly) / 2 + (uy - ly) / 2 * 1.1)
        lx, ux = ax.get_xlim()
        ax.set_xlim((ux + lx) / 2 - (ux - lx) / 2 * 1.1, (ux + lx) / 2 + (ux - lx) / 2 * 1.1)
        if plot_a_params: # L0_a, L1_a parameters
            axes_labels = ['L0_a', 'L1_a'] 
        elif self._bl._param_format == 'combined_no_1S': # L0_b, L1_a parameters
            axes_labels = ['L0_b', 'L1_a'] 
        else: # L0_b, L1_b parameters
            axes_labels = ['L0_b', 'L1_b'] 
        ax.set_xlabel(axes_labels[0], fontweight='semibold', fontsize=12)
        ax.set_ylabel(axes_labels[1], fontweight='semibold', fontsize=12)
        fig.tight_layout()

        # Use scientific notation for tick labels
        def set_sci_notation(axis, which='both'):
            # ticks = axis.get_majorticklocs()
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            if which in ('x', 'both'):
                ax.xaxis.set_major_formatter(formatter)
            if which in ('y', 'both'):
                ax.yaxis.set_major_formatter(formatter)
            fig.canvas.draw_idle()

        set_sci_notation(ax.xaxis, which='x')
        set_sci_notation(ax.yaxis, which='y')

        return fig

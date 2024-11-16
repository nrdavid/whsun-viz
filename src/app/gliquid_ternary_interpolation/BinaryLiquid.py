"""
author: Joshua Willwerth
this script merges the previously implemented Gliquid_Optimization, LIFT_Optimization_HSX.py, and Gliquid_Reconstruction
into a single file for maximum functionality.
a separate script will be created for TernaryLiquid to display and fit ternary liquidus data
"""
import math
import time

import numpy as np
import pandas as pd
import sympy as sp

import gliquid_ternary_interpolation.Data_Management as dm
from gliquid_ternary_interpolation.hsx_gliq_final import HSX
import plotly.io as pio
import warnings
import matplotlib.pyplot as plt
from pymatgen.core import Element

# reduce verbosity of pulling DFT energies from the MP API
warnings.filterwarnings("ignore")

# some fine-tuning of the plyplots so they look nicer
plt.rcParams['axes.xmargin'] = 0.005
plt.rcParams['axes.ymargin'] = 0

X_step = 0.01
X_vals = np.arange(0, 1 + X_step, X_step)
X_logs = np.log(X_vals[1:-1])


def find_local_minima(points):
    # Function to check if a point is a local minimum
    def is_lt_prev(index):
        if index == 0:
            return False
        else:
            return points[index][1] < points[index - 1][1]

    local_minima = []
    current_section = []

    for i in range(len(points)):
        # if lower temp than prev
        if is_lt_prev(i):
            current_section = [points[i]]
        # if current section exists and point is same temp
        elif current_section and current_section[-1][1] == points[i][1]:
            current_section.append(points[i])
        # higher temp and current section exists
        elif current_section:
            local_minima.append(current_section[int(len(current_section) / 2)])
            current_section = []

    return local_minima


def find_local_maxima(points):
    # Function to check if a point is a local minimum
    def is_gt_prev(index):
        if index == 0:
            return False
        else:
            return points[index][1] > points[index - 1][1]

    local_maxima = []
    current_section = []

    for i in range(len(points)):
        # if higher temp than prev
        if is_gt_prev(i):
            current_section = [points[i]]
        # if current section exists and point is same temp
        elif current_section and current_section[-1][1] == points[i][1]:
            current_section.append(points[i])
        # lower temp and current section exists
        elif current_section:
            local_maxima.append(current_section[int(len(current_section) / 2)])
            current_section = []

    return local_maxima


# TODO: allow for flexible construction that ignores alphabetical constraint
# TODO: allow for construction without MPDS PD data

class BinaryLiquid:

    def __init__(self, system, dft_type="GGA/GGA+U", mpds_pd=0, params=None):

        self.constraints = None
        if isinstance(system, str):
            self.sys_name = system
            self.components = system.split('-')
        elif isinstance(system, list):
            self.components = system
            self.sys_name = '-'.join(system)
        else:
            print("Error: sys must be a hyphenated string or a list")
            self.init_error = True
            return

        for ele in self.components:
            try:
                Element(ele)
            except ValueError as e:
                print(f"Error: {e}")
                self.init_error = True
                return

        # pull MPDS Data
        # components will be sorted alphabetically at this step
        # TODO: enforce ordering speficied prior to this call
        # self.MPDS_json / MPDS_liquidus may be undefined here, initialize for specified param values
        # TODO: commented the below line out (fix later)
        self.MPDS_json, self.component_data, self.MPDS_liquidus = dm.get_MPDS_data(self.components, pd_ind=mpds_pd)
        if not self.MPDS_liquidus:
            if mpds_pd == 0:
                print(f"Error: MPDS liquidus data for system {self.sys_name} is incomplete")
            else:
                print(f"Error: MPDS liquidus data for entry {self.MPDS_json['entry']} is incomplete")
            self.init_error = True
        else:
            MPDS_liq_T = [coord[1] for coord in self.MPDS_liquidus]
            # set component melting temperatures to match the phase diagram
            self.component_data[self.components[0]][1] = MPDS_liq_T[0]
            self.component_data[self.components[-1]][1] = MPDS_liq_T[-1]

        if 'temp' in self.MPDS_json:
            self.T_bounds = [self.MPDS_json['temp'][0] + 273.15, self.MPDS_json['temp'][1] + 273.15]
        else:
            comp_tms = [self.component_data[comp][1] for comp in self.components]
            self.T_bounds = [min(comp_tms) - 50, max(comp_tms) * 1.1 + 50]

        # self.components = self.MPDS_json['chemical_elements'] compare these to check ordering?

        # pull DFT Convexhull Data
        self.dft_type = dft_type
        mp_pd = dm.get_dft_convexhull(self.components, self.dft_type)
        self.phases = []

        # initialize phases from DFT entries on the hull
        for entry in mp_pd.stable_entries:

            try:
                composition = entry.composition.fractional_composition.as_dict()[self.components[1]]
            except KeyError:
                composition = 0

            phase = {'name': entry.name, 'comp': composition, 'points': [],
                     'energy': 96485 * mp_pd.get_form_energy_per_atom(entry)}
            # convert eV/atom to J/mol (96,485 J/mol per 1 eV/atom)

            self.phases.append(phase)

        self.phases.sort(key=lambda x: x['comp'])
        self.phases.append({'name': 'L', 'points': []})

        # initialize the dict for plotted points
        if params is None:
            params = [0] * 4
        self.L0_a, self.L0_b, self.L1_a, self.L1_b = params
        self.opt_path = None
        self.hsx = None

        # initialize phase points and self.hsx
        self.update_phase_points()

    def to_HSX(self, output="dict"):
        data = {'X': list(X_vals), 'S': [], 'H': [], 'Phase Name': ['L' for _ in X_vals]}

        H_a_liq = self.component_data[self.components[0]][0]
        H_b_liq = self.component_data[self.components[1]][0]
        H_lc = (H_a_liq * X_vals[-2:0:-1] +
                H_b_liq * X_vals[1:-1])
        H_xs = X_vals[1:-1] * X_vals[-2:0:-1] * (self.L0_a + self.L1_a * (1 - 2 * X_vals[1:-1]))
        data['H'] = list(H_lc + H_xs)
        data['H'].insert(0, H_a_liq)
        data['H'].append(H_b_liq)

        R = 8.314

        S_a_liq = self.component_data[self.components[0]][0] / self.component_data[self.components[0]][1]
        S_b_liq = self.component_data[self.components[1]][0] / self.component_data[self.components[1]][1]
        S_lc = (S_a_liq * X_vals[-2:0:-1] +
                S_b_liq * X_vals[1:-1])
        S_ideal = -R * (X_vals[1:-1] * X_logs + X_vals[-2:0:-1] * X_logs[::-1])
        S_xs = -X_vals[1:-1] * X_vals[-2:0:-1] * (self.L0_b + self.L1_b * (1 - 2 * X_vals[1:-1]))
        data['S'] = list(S_lc + S_ideal + S_xs)
        data['S'].insert(0, S_a_liq)
        data['S'].append(S_b_liq)

        for x in X_vals:
            for phase in self.phases:
                if phase['name'] == 'L':
                    continue
                if round(phase['comp'], 2) == round(x, 2):
                    data['X'].append(round(x, 2))
                    data['H'].append(phase['energy'])
                    data['S'].append(0)
                    data['Phase Name'].append(phase['name'])

        if output == "dict":
            return data
        if output == "dataframe":
            return pd.DataFrame(data)

    def update_phase_points(self):
        """
        This function uses the HSX class to calculate the phase points for given parameter values.
        Formerly called 'convex_hull', this converts phase data into the HSX form and uses Abrar's HSX code to
        calculate the liquidus and intermetallic phase decompositions.
        :return: None
        """
        data = self.to_HSX()
        hsx_dict = {'data': data, 'phases': [phase['name'] for phase in self.phases], 'comps': self.components}
        self.hsx = HSX(hsx_dict, [self.T_bounds[0] - 273.15, self.T_bounds[-1] - 273.15])

        phase_points = self.hsx.get_phase_points()
        for phase in self.phases:
            phase['points'] = phase_points[phase['name']]

    def find_invariant_points(self):
        """
        This function uses the MPDS json and MPDS liquidus to identify invariant points in the MPDS data.
        This does not take into account DFT phases, which may differ in composition from the phases in the MPDS data.
        To use this, there must be both valid liquid and json data for a binary system. If there is a valid json but
        no liquidus data, use Data_Management.identify_MPDS_phases() instead
        :return: List of invariant points
        """
        if self.MPDS_json['reference'] is None:
            print("system JSON does not contain any data!\n")
            return []

        invariants = dm.identify_MPDS_phases(self.MPDS_json, verbose=True)

        phase_labels = [label[0] for label in self.MPDS_json['labels']]
        ss_label = "(" + self.components[0] + ", " + self.components[1] + ")"
        ss_label_inv = "(" + self.components[1] + ", " + self.components[0] + ")"
        ss_labels = [ss_label, ss_label + ' ht', ss_label + ' rt',
                     ss_label_inv, ss_label_inv + ' ht', ss_label_inv + ' rt']
        full_comp_ss = bool([label for label in phase_labels if label in ss_labels])

        if not full_comp_ss:
            # locate local maxima and minima in liquidus
            maxima = find_local_maxima(self.MPDS_liquidus)
            minima = find_local_minima(self.MPDS_liquidus)

            # assign line compounds that overlap with liquidus maxima points as congruent melting points
            for coords in reversed(maxima):
                for inv in invariants:
                    if (inv['type'] in ['lc', 'ss'] and abs(inv['comp'] - coords[0]) < 0.01 and
                            abs(inv['tbounds'][1][1] - coords[1]) < 5):
                        inv['type'] = 'cmp'
                        maxima.remove(coords)
                        break

            # check to see if any miscibilty gaps are improperly bounded
            for inv in invariants:
                if inv['type'] == 'mig' and inv['comp'] == inv['cbounds'][0][0]:

                    # if there is one maxima and one or more minima that can be used as reference points:
                    if len(maxima) == 1 and len(minima) > 0:
                        inv['comp'] = maxima[0][0]
                        inv['tbounds'][1] = maxima[0]
                        # find closest minima as mig edge region:
                        closest = [0, 0]
                        for coords in minima:
                            if abs(coords[0] - inv['comp']) < abs(closest[0] - inv['comp']):
                                closest = coords
                        # find next region at same temp:
                        if closest[0] - inv['comp'] > 0:
                            inv['cbounds'][1] = closest  # right of mig
                            try:
                                ind = self.MPDS_liquidus.index(closest)
                            except ValueError:
                                print("Error with miscibility gap boundary detection")
                                self.init_error = True
                                break
                            for i in reversed(range(0, ind - 1)):
                                if self.MPDS_liquidus[i][1] <= closest[1]:
                                    inv['cbounds'][0] = self.MPDS_liquidus[i]
                                    break
                        else:
                            inv['cbounds'][0] = closest  # left of mig
                            try:
                                ind = self.MPDS_liquidus.index(closest)
                            except ValueError:
                                print("Error with miscibility gap boundary detection")
                                self.init_error = True
                                break
                            for i in range(ind + 1, len(self.MPDS_liquidus)):
                                if self.MPDS_liquidus[i][1] <= closest[1]:
                                    inv['cbounds'][1] = self.MPDS_liquidus[i]
                                    break
                    else:
                        print('Error in mig detection with insufficient liquidus points to correct')
                        self.init_error = True

            migs = [inv for inv in invariants if inv['type'] == 'mig']
            if len(migs) > 1:
                for mig in migs:
                    ind = len(self.MPDS_liquidus) - 1
                    for i in range(len(self.MPDS_liquidus) - 1):
                        if self.MPDS_liquidus[i][0] <= mig['comp'] < self.MPDS_liquidus[i + 1][0]:
                            ind = i
                            break
                    if abs(self.MPDS_liquidus[ind][1] - mig['tbounds'][1][1]) > 15:
                        invariants.remove(mig)

            # peritectics will primarily be identified by line compounds that don't contact the liquidus
            mpds_phases = [inv for inv in invariants if inv['type'] in ['lc', 'ss', 'cmp']]
            # sort by descending temperature
            mpds_phases.sort(key=lambda x: x['tbounds'][1][1], reverse=True)
            stable_phase_comps = []
            # main loop
            for phase in mpds_phases:
                # congruent melting points will not be considered for peritectic formation but will limit others
                if phase['type'] == 'cmp':
                    stable_phase_comps.append(phase['comp'])
                    continue

                # minimum temperature difference between phase decomp temp and liquidus temp needed to consider for peri
                min_diff = 10
                liq_temp = self.MPDS_liquidus[0][1]
                for i in range(len(self.MPDS_liquidus) - 1):
                    if self.MPDS_liquidus[i + 1][0] >= phase['comp'] > self.MPDS_liquidus[i][0]:
                        liq_temp = min(self.MPDS_liquidus[i + 1][1], self.MPDS_liquidus[i][1])
                        break
                if liq_temp - phase['tbounds'][1][1] < min_diff:
                    stable_phase_comps.append(phase['comp'])
                    continue

                sections = []
                current_section = []
                for i in range(len(self.MPDS_liquidus) - 1):
                    # liquidus point is above or equal to phase temp
                    if self.MPDS_liquidus[i][1] >= phase['tbounds'][1][1]:
                        current_section.append(self.MPDS_liquidus[i])
                        if self.MPDS_liquidus[i + 1][1] >= phase['tbounds'][1][1] \
                                and i + 1 == len(self.MPDS_liquidus) - 1:
                            current_section.append(self.MPDS_liquidus[i + 1])
                            sections.append(current_section)
                    # liquidus point is first point below phase temp
                    elif current_section:
                        # add to section if closer to phase temp than last point above temp:
                        if (abs(phase['tbounds'][1][1] - current_section[-1][1]) >
                                abs(phase['tbounds'][1][1] - self.MPDS_liquidus[i][1])):
                            current_section.append(self.MPDS_liquidus[i])
                        # end section
                        sections.append(current_section)
                        current_section = []
                    # next liquidus point is above temp
                    elif self.MPDS_liquidus[i + 1][1] >= phase['tbounds'][1][1] > self.MPDS_liquidus[i][1]:
                        # add to section if current point below phase temp is closer than next point above
                        if (abs(phase['tbounds'][1][1] - self.MPDS_liquidus[i + 1][1]) >
                                abs(phase['tbounds'][1][1] - self.MPDS_liquidus[i][1])):
                            current_section.append(self.MPDS_liquidus[i])

                # find endpoints of liquidus segments excluding the component ends
                endpoints = []
                for section in sections:
                    if section[0] != self.MPDS_liquidus[0]:
                        endpoints.append(section[0])
                    if section[-1] != self.MPDS_liquidus[-1]:
                        endpoints.append(section[-1])

                for comp in stable_phase_comps:
                    # filter out endpoints if there exists a stable phase between the current phase and the liquidus
                    endpoints = [ep for ep in endpoints
                                 if abs(comp - ep[0]) > abs(phase['comp'] - ep[0]) or
                                 abs(comp - phase['comp']) > abs(phase['comp'] - ep[0])]

                # sort by increasing distance to liquidus to find the shortest distance
                endpoints.sort(key=lambda x: abs(x[0] - phase['comp']))

                # take the closest liquidus point to the phase as the peritectic point
                if endpoints:
                    invariants.append({'type': 'per', 'comp': endpoints[0][0], 'temp': endpoints[0][1],
                                       'phase': phase['name'], 'phase_comp': phase['comp']})
                stable_phase_comps.append(phase['comp'])

            # filter minima points and add as eutectic points
            for coords in reversed(minima):
                # filter out points in misc gap boundaries
                for inv in invariants:
                    if inv['type'] == 'mig':
                        lhs, rhs = inv['cbounds'][0][0], inv['cbounds'][1][0]
                        if abs(lhs - coords[0]) < 0.05 or abs(rhs - coords[0]) < 0.05:
                            minima.remove(coords)
                            break
                # filter out edge eutectics - they are not helpful when choosing points to solve from
                # if coords[0] > 0.95 or coords[0] < 0.05:
                #     minima.remove(coords)
            for coords in minima:
                invariants.append({'type': 'eut', 'comp': coords[0], 'temp': coords[1]})
            n_euts = len([inv for inv in invariants if inv['type'] == 'eut'])
            n_comps = len([inv for inv in invariants if inv['type'] in ['lc', 'ss']])
            if n_euts > n_comps + 1:
                print("too many eutectic points identified")
                self.init_error = True

        else:  # aze
            minima = find_local_minima(self.MPDS_liquidus)
            if len(minima) > 1:
                print("too many azeotrope points identified")
                self.init_error = True
            elif len(minima) == 1:
                invariants.append({'type': 'aze', 'comp': minima[0][0], 'temp': minima[0][1]})
            # process solidus here
            print('solidus processing not implemented')

        invariants.sort(key=lambda x: x['comp'])
        return invariants

    def solve_params_from_constraints(self, guessed_vals, symbols):
        a, b, c, d = symbols
        # guessed values for b, & d
        print(self.constraints[a].subs(guessed_vals))
        print(self.constraints[c].subs(guessed_vals))
        self.L0_a = self.constraints[a].subs(guessed_vals)
        self.L0_b = guessed_vals[b]
        self.L1_a = self.constraints[c].subs(guessed_vals)
        self.L1_b = guessed_vals[d]

    def liquidus_is_continuous(self, tol=2 * X_step):
        last_coords = None
        for coords in self.phases[-1]['points']:
            if last_coords and coords[0] - last_coords[0] > tol:
                return False
            last_coords = coords
        return True

    def calculate_deviation_metrics(self, num_points=30):
        x1, T1 = zip(*self.MPDS_liquidus)
        x2, T2 = zip(*self.phases[-1]['points'])

        # compare (up to) num_points points evenly spaced across composition space
        x_coords = np.linspace(x1[0], x1[-1], num_points)
        Y1 = []
        Y2 = []

        for i in range(len(x_coords)):
            MPDS_ind = fit_ind = -1

            for j in range(len(self.MPDS_liquidus) - 1):
                if x1[j] <= x_coords[i] < x1[j + 1]:
                    # print(x1[j], x_coords[i], x1[j + 1], j)
                    MPDS_ind = j
                    break
            for j in range(len(self.phases[-1]['points']) - 1):
                if x2[j] <= x_coords[i] < x2[j + 1]:
                    # print(x2[j], x_coords[i], x2[j + 1], j)
                    fit_ind = j
                    break

            # print("i = ", i, "x[i] = ", x_coords[i], "MPDS_ind = ", MPDS_ind, "fit_ind = ", fit_ind)
            if MPDS_ind != -1 and fit_ind != -1:
                m1 = (T1[MPDS_ind] - T1[MPDS_ind + 1]) / (x1[MPDS_ind] - x1[MPDS_ind + 1])
                b1 = (x1[MPDS_ind] * T1[MPDS_ind + 1] - x1[MPDS_ind + 1] * T1[MPDS_ind]) / (
                        x1[MPDS_ind] - x1[MPDS_ind + 1])
                m2 = (T2[fit_ind] - T2[fit_ind + 1]) / (x2[fit_ind] - x2[fit_ind + 1])
                b2 = (x2[fit_ind] * T2[fit_ind + 1] - x2[fit_ind + 1] * T2[fit_ind]) / (
                        x2[fit_ind] - x2[fit_ind + 1])
                Y1.append(m1 * x_coords[i] + b1)
                Y2.append(m2 * x_coords[i] + b2)

        # find absolute difference at each point
        point_diffs = [abs(Y1[i] - Y2[i]) for i in range(len(Y2))]
        squared_point_diffs = [(Y1[i] - Y2[i]) ** 2 for i in range(len(Y2))]

        return np.mean(point_diffs), math.sqrt(np.mean(squared_point_diffs))


    def f(self, point):
        # if len(point) != 2: #TODO implement this
        #     raise NotImplementedError
        # self.L0_b, self.L1_b = point
        # self.calculate_L_a_parameters()
        try:
            self.update_phase_points()
        except ValueError as e:
            print(e)
            return float('inf')
        if not self.liquidus_is_continuous():
            return float('inf')
        mae, rmse = self.calculate_deviation_metrics()
        return mae

    def nelder_mead(self, max_iter=100, tol=5e-2, verbose=False):
        """
        Nelder-Mead algorithm for fitting the liquid non-ideal mixing parameters.

        Args:
            max_iter: Maximum number of iterations.
            tol: Tolerance for convergence.
            verbose: Determines if updates are printed to the terminal

        Returns:
            mae: The mean average error (MAE) from the fitted liquidus to the MPDS liquidus.
            rmse: The standard deviation from the fitted liquidus to the MPDS liquidus.
            opt_path: An [3 x (num_dims + 1) x num_iterations] Numpy Array of the optimization path
        """
        x0 = np.array([[-20, -20], [-20, 20], [20, -20]], dtype=float)
        n = x0.shape[1]  # Number of dimensions
        self.opt_path = np.empty((3, n + 1, max_iter), dtype=float)
        initial_time = time.time()

        print("--- begin nelder mead optimization ---")

        if n == 1:
            # Line search for 1D case
            while True:
                f_vals = np.array([self.f(x) for x in x0])
                iworst = np.argmax(f_vals)
                xbest = x0[iworst, :]
                step = 0.1 * (xbest - x0[0, :])
                xnew = xbest - step
                f_new = self.f(xnew)
                if f_new < f_vals[iworst]:
                    x0[iworst, :] = xnew
                else:
                    step = -step
                    xnew = xbest + step
                    if self.f(xnew) < f_vals[iworst]:
                        x0[iworst, :] = xnew
                    else:
                        break
            return xbest, self.f(xbest)
        elif 1 < n <= 4:
            # Nelder-Mead for higher dimensions
            for i in range(max_iter):
                start_time = time.time()
                if verbose:
                    print("iteration #", i)

                f_vals = np.array([self.f(x) for x in x0])
                self.opt_path[:, :n, i] = x0
                self.opt_path[:, n:, i] = np.array([[f] for f in f_vals])
                iworst = np.argmax(f_vals)
                ibest = np.argmin(f_vals)
                centroid = np.mean(x0[f_vals != f_vals[iworst]], axis=0)
                xreflect = centroid + 1.0 * (centroid - x0[iworst, :])
                f_xreflect = self.f(xreflect)

                if iworst == ibest:
                    self.opt_path = self.opt_path[:, :, :i]
                    raise RuntimeError("Nelder-Mead algorithm is unable to find physical parameter values.")

                # Reflection
                if f_vals[iworst] <= f_xreflect < f_vals[n]:
                    x0[iworst, :] = xreflect
                # Expansion
                elif f_xreflect < f_vals[ibest]:
                    xexp = centroid + 2.0 * (xreflect - centroid)
                    if self.f(xexp) < f_xreflect:
                        x0[iworst, :] = xexp
                    else:
                        x0[iworst, :] = xreflect
                # Contraction
                else:
                    if f_xreflect < f_vals[n]:
                        xcontract = centroid + 0.5 * (xreflect - centroid)
                        if self.f(xcontract) < self.f(x0[iworst, :]):
                            x0[iworst, :] = xcontract
                        else:  # Shrink Step
                            x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                            [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                            x0[iworst, :] = x0[imid, :] + 0.5 * (x0[imid, :] - x0[ibest, :])
                    else:
                        xcontract = centroid + 0.5 * (x0[iworst, :] - centroid)
                        if self.f(xcontract) < self.f(x0[iworst, :]):
                            x0[iworst, :] = xcontract
                        else:  # Shrink Step
                            x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                            [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                            x0[imid, :] = x0[ibest, :] + 0.5 * (x0[imid, :] - x0[ibest, :])

                if verbose:
                    for row, f_val in zip(x0, f_vals):
                        print(row, f_val)
                    print("1/2 height of triangle = ", np.max(np.abs(x0 - centroid)))
                    print("--- %s seconds ---" % (time.time() - start_time))

                # Check convergence
                if np.max(np.abs(x0 - centroid)) < tol:
                    print("--- total time %s seconds ---" % (time.time() - initial_time))
                    mae, rmse = self.calculate_deviation_metrics()
                    print("mean temperature deviation per point between curves =", mae, '\n')
                    self.opt_path = self.opt_path[:, :, :i]
                    return mae, rmse, self.opt_path
            raise RuntimeError("Nelder-Mead algorithm did not converge within limit.")
        else:
            raise ValueError("Nelder-Mead algorithm is not implemented for dimensions > 4.")

    def fit_parameters(self):
        """Fit the liquidus non-ideal mixing parameters for a binary system. This function utilizes the nelder-mead
         algorithm to minimize the temperature deviation in the liquidus"""

        if self.MPDS_liquidus is None:
            print("system missing liquidus data!\n")
            return

        # find invariant points
        invariants = self.find_invariant_points()
        for phase in self.phases:
            print(phase)
        for inv in invariants:
            print(inv)

        # self.plot_hsx_tx()
        # eqs = []

        x, t, a, b, c, d = sp.symbols('x t a b c d')
        # L0_a = a, L0_b = b, L1_a = c, L1_b = d

        R = 8.314
        G_a = (self.component_data[self.components[0]][0] -
                   t * self.component_data[self.components[0]][0] / self.component_data[self.components[0]][1])
        G_b = (self.component_data[self.components[1]][0] -
                   t * self.component_data[self.components[1]][0] / self.component_data[self.components[1]][1])
        G_ideal = R * t * (x * sp.log(x) + (1 - x) * sp.log(1 - x))
        G_xs = x * (1 - x) * ((a + b * t) + (1 - 2 * x) * (c + d * t))
        G_liq = G_a * x + G_b * (1 - x) + G_ideal + G_xs

        G_prime = sp.diff(G_liq, x)
        # print(G_prime)
        G_double_prime = sp.diff(G_prime, x)
        # print(G_double_prime)

        print()

        # compare invariant points to self.phases to assess solving conditions
        for inv in invariants:
            if inv['type'] == 'mig':
                x0 = 1
                x1, t1 = inv['cbounds'][0]
                x2, t2 = inv['tbounds'][1]
                x3, t3 = inv['cbounds'][1]
                print(x1, t1)
                print(x2, t2)
                print(x3, t3)

                eqn1 = sp.Eq(G_double_prime.subs({x: x2, t: t2}), 0)
                eqn2 = sp.Eq(G_liq.subs({x: x1, t: t1}) + G_prime.subs({x: x1, t: t1}) * (x0 - x1), 0)
                eqn3 = sp.Eq(G_liq.subs({x: x3, t: t3}) + G_prime.subs({x: x3, t: t3}) * (x0 - x3), 0)
                eqn4 = sp.Eq(G_prime.subs({x: x1, t: t1}), G_prime.subs({x: x3, t: t3}))
                # equation 1 and equation 4 are the best so far...

                equations = [eqn1, eqn2, eqn3, eqn4]
                labels = ['eqn1', 'eqn2', 'eqn3', 'eqn4']
                for i in range(4):
                    for j in range(i):
                        eqs = [equations[i], equations[j]]

                        self.constraints = sp.solve(eqs, (a, c))
                        self.solve_params_from_constraints({b: 0, d: 0}, [a, b, c, d])
                        print(self.L0_a, self.L0_b, self.L1_a, self.L1_b)

                        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                        fig.suptitle(f"{labels[i]}-{labels[j]}")
                        for T, ax in zip([t1, t2], axes):
                            gliq_fx = sp.lambdify(x, G_liq.subs({t: T, a: self.L0_a, b: self.L0_b, c: self.L1_a, d: self.L1_b}),
                                                  'numpy')
                            gliq_vals = gliq_fx(X_vals)
                            ax.plot(X_vals, gliq_vals, label=T)
                            ax.set_title(f"T = {T}")
                        plt.show()
                        self.update_phase_points()
                        self.plot_hsx_tx()
                        # self.hsx.plot_hsx()
                # print(f"L0_a: {sol[a]}, L0_b: {sol[b]}, L1_a: {sol[c]}, L1_b: {sol[d]}")

                # print(G_liq.subs({t: t1, a: sol[a], b: sol[b], c: sol[c], d: sol[d]}))
                # print(G_a.subs({t: t1}))
                # print(G_b.subs({t: t1}))
                # print(self.component_data)

        # rank points and select the best 1-4

        # set up equations

        # call nelder-mead and save MAE, RMSE, & path
        # mae, rmse, self.opt_path = self.nelder_mead()

    def plot_hsx_tx(self, save=False, results_dir=None):
        fig = self.hsx.plot_tx(mpds_liquidus=self.MPDS_liquidus)

        if not save:
            fig.show()
            return

        fname = f"{self.sys_name}_FITTED_MP_GGA.svg"
        if results_dir:
            fname = f"{results_dir}/{fname}"
        # pio.write_image(fig=fig, file=fname)




# sys = 'Mo-Tb'

# sys = 'Ag-Co'
# sys = 'Cr-Cu'
# sys = 'In-V'
# # sys = 'Cu-Ho'
# sys = 'Au-Ru'
# sys = 'Cr-Y'

# # sys = 'Ag-Co'
# # sys = 'Gd-Li'
# # sys = 'Cu-Ho'
# # sys = 'Cu-Mg'
# # sys = 'Ge-Hf'

# # sys = 'Fe-Ti'
# sys = 'Bi-Cd'

# # read in xlsx file as dataframe
# df = pd.read_excel("fitted_system_data_new.xlsx")

# # extract row "system" column containing sys 
# rows = df.loc[df['system'] == sys]

# # keep the second row
# if len(rows) != 2:
#     row = rows.iloc[0]
# else:
#     row = rows.iloc[1]

# # # extract values from column 'L0_a', 'L0_b', 'L1_a', 'L1_b' and put them in a list
# fitted_params = [row['L0_a'], row['L0_b'], row['L1_a'], row['L1_b']]
# params = [18929.60888, -19.17111136, -6699.016816, -27.02468149]
# print(fitted_params)

# sys_d = BinaryLiquid(sys, params=fitted_params)
# fig = sys_d.hsx.plot_tx()
# fig.show()
# sys_d.hsx.plot_tx_scatter()


# Cu_mg = BinaryLiquid('Cu-Mg', params=[-31050, -10.91, -68310, 66.19])
# print(Cu_mg.phases)
# fig = Cu_mg.hsx.plot_tx()
# fig.show()
# Cu_mg.hsx.plot_hsx()

# Cu_mg = BinaryLiquid(['Cu', 'Mg'])
# Cu_mg.fit_parameters()
# print(Cu_mg.phases)
# fig = Cu_mg.hsx.plot_tx()
# fig.show()

# Ce_Mo = BinaryLiquid(['Ce', 'Mo'])
# Ce_Mo.fit_parameters()

# print(Ce_Mo.phases)


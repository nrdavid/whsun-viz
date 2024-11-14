"""
@author: Joshua Willwerth

This script provides functions to interface with APIs and locally cache data
"""

from pymatgen.ext.matproj import MPRester as Legacy_MPRester
from mp_api.client import MPRester
from mpds_client import MPDSDataRetrieval, MPDSDataTypes, APIError
from urllib.parse import urlencode
from emmet.core.thermo import ThermoType
from pymatgen.entries.mixing_scheme import MaterialsProjectDFTMixingScheme
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element
import httplib2
import os
import json
import time
import numpy as np
import re
from gliquid_ternary_interpolation.auth import MPDS_api_key, New_MP_api_key, Legacy_MP_api_key

# API keys here
MPDS_api_key = MPDS_api_key
New_MP_api_key = New_MP_api_key
Legacy_MP_api_key = Legacy_MP_api_key

data_dir = "gliquid_ternary_interpolation/matrix_data_jsons"

enthalpies_file = f"{data_dir}/fusion_enthalpies.json"
melt_temps_file = f"{data_dir}/fusion_temperatures.json"

if not os.path.exists(enthalpies_file):
    enthalpies = {}
else:
    with open(enthalpies_file, "r") as file:
        enthalpies = json.load(file)

if not os.path.exists(melt_temps_file):
    melt_temps = {}
else:
    with open(melt_temps_file, "r") as file:
        melt_temps = json.load(file)


def t_at_boundary(t, boundary):
    return t <= boundary[0] + 2 or t >= boundary[1] - 2


def section_liquidus(points):
    sections = []
    current_section = []

    x1, y1 = points[0]
    x2, y2 = points[1]

    if x2 > x1:
        direction = "increasing"
        current_section.append(points[0])
    elif x2 < x1:
        direction = "decreasing"
        current_section.append(points[0])
    else:
        direction = None
        sections.append([points[0]])

    for i in range(1, len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        if x2 > x1:
            new_direction = "increasing"
        elif x2 < x1:
            new_direction = "decreasing"
        else:
            new_direction = None
        # add x1, y1
        current_section.append(points[i])

        if new_direction != direction or new_direction is None:
            if current_section:
                sections.append(current_section)
                current_section = []
        direction = new_direction

    if current_section:
        current_section.append(points[-1])
        sections.append(current_section)
    return sections


def within_tol_from_line(p1, p2, p3, tol):
    try:
        m = (p1[1] - p2[1]) / (p1[0] - p2[0])
    except ZeroDivisionError:
        return tol >= abs(p1[1] - p2[1])
    y_h = m * (p3[0] - p1[0]) + p1[1]
    return tol >= abs(p3[1] - y_h)


def fill_liquidus(p1, p2, max_interval):
    num_between = int(np.floor((p2[0] - p1[0]) / max_interval))
    filled_X = np.linspace(p1[0], p2[0], num_between + 2)
    filled_T = np.linspace(p1[1], p2[1], num_between + 2)
    filled_section = [[filled_X[i], filled_T[i]] for i in range(len(filled_X))]
    return filled_section[1:-1]


def get_ref_data(mpds_json):
    if mpds_json['reference'] is None or 'citation' in mpds_json['reference']:
        return
    req = httplib2.Http()
    url = ("https://api.mpds.io/v0/download/c?q=" + mpds_json['entry'] +
           "&fmt=bib&sid=3bVkBuEjMC3XWdhXpwtUOhhkwvbMc3E1oled0eN514WBsCiA&ed=0")
    # "&fmt=bib&sid=biRbiriNitekrAretmEYenCOkraSGeleYAZilmIssAtIrlaR&ed=0")
    parsed_data = {}
    for _ in range(2):
        response, content = req.request(
            uri=url,
            method='GET',
        )
        content = content.decode('utf-8')
        pattern = re.compile(r'(\w+)\s*=\s*\{\s*(.*?)\s*\}', re.DOTALL)
        matches = pattern.findall(content)
        parsed_data = {key: value.replace('\n', ' ').strip() for key, value in matches}
        if parsed_data:
            time.sleep(3)
            break
        else:
            time.sleep(6)
            print("Server error, resending request")
    if not parsed_data:
        print("Error, no bibliography data found!")
        return

    authors = parsed_data['author'].split(',')
    if len(authors) > 1:
        authorship = authors[0].split(' ')[0] + ' et al.'
    else:
        authorship = authors[0]
    year = parsed_data['year']
    citation = f"{authorship} ({year})"
    mpds_json['reference']['citation'] = citation


# load the MPDS data for system from the API or local cache
# returns dict, dict, 2D list
# ind is a testing feature used to select a phase diagram at a specific rank. highest ranked PD selected by default
def get_MPDS_data(components, pd_ind=0, dtype='all'):
    """Retrive MPDS data from cache or API.
    Required: List of components. Must provide only two components if querying binary phase diagrams.
    pd -> index of PD sorted by selection criteria. Specifying 'None' will cache all PDs and return nothing.
    dtype -> type of data returned by function. 'comp' or 'all_thermo' will return only component data,
    with 'all_thermo' returning additional component data. Default return for 'all' is a tuple of three fields:
    MPDS JSON (dict), Component data (dict), MPDS Liquidus (2D list)"""
    client = MPDSDataRetrieval(api_key=MPDS_api_key)
    client.dtype = MPDSDataTypes.PEER_REVIEWED
    sys = '-'.join(sorted(components))
    component_data = {}

    for comp in components:
        if comp not in enthalpies or comp not in melt_temps:
            print("searching for " + comp + " data from MPDS...")

            # component properties (sample Al)
            #         "property": {
            #           "name": "enthalpy change at melting point",
            #           "units": "kJ g-at.-1",
            #           "domain": "thermal and thermodynamic properties",
            #           "scalar": 10.71,
            #           "category": "enthalpy change at phase transition"
            #         },
            #         "condition": [
            #           {
            #             "name": "Temperature",
            #             "units": "K",
            #             "scalar": 933.5

            comp_fields = {
                'P': ['sample.measurement[0].property.scalar', 'sample.measurement[0].condition[0].scalar']}
            data = client.get_data(search={'formulae': comp,
                                           'props': 'enthalpy change at melting point'}, fields=comp_fields)

            # either take the first value or median value
            if comp not in enthalpies:
                print("caching enthalpy of fusion data")
                enthalpies[comp] = float(data[0][0]) * 1000
                with open(enthalpies_file, "w") as f:
                    json.dump(enthalpies, f)
            if comp not in melt_temps:
                print("caching melting temperature data")
                melt_temps[comp] = float(data[0][1])
                with open(melt_temps_file, "w") as f:
                    json.dump(melt_temps, f)

            print(comp + ": enthalpy of fusion = " + "{:,.0f}".format(enthalpies[comp])
                  + " J/mol at " + "{:,.0f}".format(melt_temps[comp]) + " K")

        component_data[comp] = [enthalpies[comp], melt_temps[comp]]

        if dtype == 'all_thermo':
            comp_fields = {
                'P': ['sample.measurement[0].property.scalar', 'sample.measurement[0].condition[0].scalar']}
            data = client.get_data(search={'formulae': comp,
                                           'props': 'entropy'}, fields=comp_fields)

            best_entry = [float(data[0][0]), float(data[0][1])]
            for d in data:
                if float(d[1]) < 300 and float(d[0]) > best_entry[0]:
                    best_entry = [float(d[0]), float(d[1])]

            component_data[comp].extend(best_entry)

    # will skip system data if only component data is desired (comp) or additional data as specified above (all_thermo)
    if dtype == 'comp' or dtype == 'all_thermo':
        return component_data

    # TODO: Removing dump to run on
    # look for cached json for the system
    sys_dir = f"{data_dir}/{sys}"
    if not os.path.exists(sys_dir):
        os.makedirs(sys_dir)
    ## print current working directory
    print(os.getcwd())
    print(os.listdir())
    print(sys_dir)
    sys_file = os.path.join(f"{data_dir}/{sys}", f"{sys}_MPDS_PD_{pd_ind}.json")
    if os.path.exists(sys_file):
        print("True")
        # get MPDS data from stored jsons in liquidus curves folder
        print("\nloading JSON from cache...")
        with open(sys_file, 'r') as f:
            mpds_json = json.load(f)
        # ---------------------------- TEMPORARY -------------------------------
        # get_ref_data(mpds_json)
        # with open(sys_file, "w") as f:
        #     json.dump(mpds_json, f)
        # ------------------------------------------------------------------------
        return mpds_json, component_data, extract_MPDS_liquidus(mpds_json, components)

    else:
        print("\nsearching for phase diagram JSON from MPDS...")
        # phase diagram properties
        # arity - num elements - 2
        # naxes - num axis - 2
        # diatype - phase diagram type - "binary"
        # comp_range - 0-100
        # reference - link to entry
        # shapes - phase boundary info
        # chemical_elements - alphabetized chemcial elements in system
        # temp - temp range of diagram

        sys_fields = {'C': ['chemical_elements', 'entry', 'comp_range', 'temp', 'labels', 'shapes', 'reference']}
        valid_JSONs = []
        best_incomplete = {'reference': None}

        # search for phase diagrams and filter out empty entries
        try:
            diagrams = [d for d in client.get_data(
                search={'elements': sys, 'classes': 'binary'}, fields=sys_fields) if d]

            # step 1: determine which phase diagrams have usable liquidus data
            for d in diagrams:
                dia_json = {}
                for i in range(len(sys_fields['C'])):
                    dia_json[sys_fields['C'][i]] = d[i]

                # find next best phase diagram in the case none have a complete liquidus (for 70x70 matrix) (min 10%)
                if dia_json['comp_range'][1] - dia_json['comp_range'][0] > 10:
                    if best_incomplete['reference'] is None:
                        best_incomplete = dia_json
                    elif (dia_json['comp_range'][1] - dia_json['comp_range'][0] >
                          best_incomplete['comp_range'][1] - best_incomplete['comp_range'][0]):
                        best_incomplete = dia_json

                # remove any diagrams that don't span the entire composition range
                if dia_json['comp_range'] != [0, 100]:
                    continue

                # if liquidus curve is extracted from the diagram json without issue, assign score and add to list
                dia_liquidus = extract_MPDS_liquidus(dia_json, verbose=False)
                if dia_liquidus:
                    valid_JSONs.append(dia_json)

            if pd_ind is not None and len(valid_JSONs) <= pd_ind:
                print(f"specified index {str(pd_ind)} exceeds number of valid JSONs found for the {sys} system")
                pd_ind = 0

        except APIError:
            print(" Got 0 hits")

        # add best incomplete to the list, could be {'reference': None} for no data
        if len(valid_JSONs) == 0:
            valid_JSONs.append(best_incomplete)

        # step 2: query additional information on PDs
        endpoint = "https://api.mpds.io/v0/search/facet"
        req = httplib2.Http()

        for dia_json in valid_JSONs:
            if 'entry' not in dia_json:  # skip the fake entries
                continue

            url = endpoint + '?' + urlencode(
                {
                    'q': json.dumps({'entry': dia_json['entry']}),
                    'pagesize': 10,
                    'dtype': 1
                })
            response, content = req.request(
                uri=url,
                method='GET',
                headers={'Key': MPDS_api_key}
            )

            time.sleep(2)

            try:
                info = json.loads(content)
            except json.decoder.JSONDecodeError as e:
                print(e)
                continue

            components = dia_json['chemical_elements']
            dia_json['jcode'] = info['out'][0][5]
            dia_json['year'] = int(info['out'][0][6])
            dia_liquidus = extract_MPDS_liquidus(dia_json, verbose=False)
            if dia_liquidus:
                dia_json['tm_agreement'] = [dia_liquidus[0][1] - melt_temps[components[0]],  # both in Kelvin
                                            dia_liquidus[-1][1] - melt_temps[components[1]]]
            dia_json['unlocked'] = bool(info['out'][0][4]) or False
            # dia_json['pdtype'] = info['out'][0][2].split('.')[0]

        # step 3: select the 'best' phase diagram from valid diagrams)
        def jcode_rank(jcode, ul):
            score = 0
            if jcode == "MAS390":
                score = 2
            elif jcode == "JPEQE6":
                score = 0.5
            elif jcode == "CCCTD6":
                score = -1
            if ul:
                score += 0.75
            return score

        # sort by jcode score, then by year, tm_agreement (preferred smallest diff)
        # jcode score is the most reliable metric, since it is consistent for reporting measurement practices
        if len(valid_JSONs) > 1:
            valid_JSONs.sort(
                key=lambda x: (-jcode_rank(x['jcode'], x['unlocked']), -x['year'], sum(map(abs, x['tm_agreement']))))
            # for dia_json in valid_JSONs:
            #     print(dia_json['reference']['entry'], dia_json['jcode'], dia_json['year'], dia_json['tm_agreement'])

        # cache all PDs
        file_ind = 0
        for dia_json in valid_JSONs:
            sys_file = os.path.join(f"{data_dir}/{sys}", f"{sys}_MPDS_PD_{file_ind}.json")
            if dia_json['reference'] is not None:
                print(f"caching MPDS liquidus from entry at {dia_json['reference']['entry']} as {sys_file}...")
            else:
                print(f"No valid phase diagrams found, caching empty json")
                break
            with open(sys_file, "w") as f:
                get_ref_data(dia_json)
                json.dump(dia_json, f)
            file_ind += 1

        # return pd data at index specified, or return nothing if 'None' specified
        if pd_ind is None:
            return {}, component_data, []

        mpds_json = valid_JSONs[pd_ind]
        return mpds_json, component_data, extract_MPDS_liquidus(mpds_json, verbose=False)


def shape_to_list(svgpath):
    # split svgpath into tags, ordered pairs
    data = svgpath.split(' ')

    # remove 'L' and 'M' tags, so only ordered pairs remain
    data = [s for s in data if not (s == 'L' or s == 'M')]

    # convert string pairs into [X, T] float pairs and store as list
    X = [float(i.split(',')[0]) / 100.0 for i in data]
    T = [float(i.split(',')[1]) + 273.15 for i in data]
    return [[X[i], T[i]] for i in range(len(X))]


# pull liquidus curve data from MPDS json and convert to list of [X, T] coordinates in ascending composition order
# returns 2D list
def extract_MPDS_liquidus(MPDS_json, verbose=True):
    if MPDS_json['reference'] is None:
        if verbose:
            print("system JSON does not contain any data!\n")
        return None

    components = MPDS_json['chemical_elements']
    if verbose:
        print("reading MPDS liquidus from entry at " + MPDS_json['reference']['entry'] + "...\n")

    # extract liquidus curve svgpath from system JSON
    data = ""
    for boundary in MPDS_json['shapes']:
        if 'label' in boundary and boundary['label'] == 'L':
            data = boundary['svgpath']
            break
    if not data:
        if verbose:
            print("no liquidus data found in JSON!")
        return None

    MPDS_liquidus = shape_to_list(data)

    # remove points at the edge of the graph boundaries
    MPDS_liquidus = [coord for coord in MPDS_liquidus if not t_at_boundary(coord[1] - 273.15, MPDS_json['temp'])]

    if len(MPDS_liquidus) < 3:
        if verbose:
            print("MPDS liquidus does not span the entire composition range!")
        return None

    # split liquidus into segments of continuous points
    sections = section_liquidus(MPDS_liquidus)

    # sort sections by descending size
    sections.sort(key=len, reverse=True)

    # sort each section by ascending composition
    for section in sections:
        section.sort()

    # record endpoints of main section
    MPDS_liquidus = sections.pop(0)

    lhs = [0, melt_temps[components[0]]]
    rhs = [1, melt_temps[components[1]]]

    # append sections to the liquidus if not overlapping in range
    for section in sections:

        # if section upper bound is less than main section lower bound
        if section[-1][0] <= MPDS_liquidus[0][0] and within_tol_from_line(MPDS_liquidus[0], lhs, section[-1], 250):
            MPDS_liquidus = section + MPDS_liquidus

        # if section lower bound is greater than main section upper bound
        elif section[0][0] >= MPDS_liquidus[-1][0] and within_tol_from_line(MPDS_liquidus[-1], rhs, section[0], 250):
            MPDS_liquidus.extend(section)

        # i'll admit it at this point I am feeling pretty dumb about not using a svgpath parser because I have
        # do all of these strange exceptions to make this work. don't worry about this, it's just some edge case
        elif len(section) == 2:
            if section[0][0] < MPDS_liquidus[0][0] and within_tol_from_line(MPDS_liquidus[0], lhs, section[0], 170):
                MPDS_liquidus = [section[0]] + MPDS_liquidus

            elif section[-1][0] > MPDS_liquidus[-1][0] and within_tol_from_line(MPDS_liquidus[-1], rhs, section[-1],
                                                                                170):
                MPDS_liquidus.extend(section)

    # if the liquidus does not have endpoints near the ends of the composition range, melting temps won't be good
    if 100 * MPDS_liquidus[0][0] > 3 or 100 * MPDS_liquidus[-1][0] < 97:
        if verbose:
            print(f"MPDS liquidus does not span the entire composition range! "
                  f"({100 * MPDS_liquidus[0][0]}-{100 * MPDS_liquidus[-1][0]})")
        return None

    MPDS_liquidus.sort()

    # fill in ranges with missing points
    for i in reversed(range(len(MPDS_liquidus) - 1)):
        if MPDS_liquidus[i + 1][0] - MPDS_liquidus[i][0] > 0.06:
            filler = fill_liquidus(MPDS_liquidus[i], MPDS_liquidus[i + 1], 0.03)
            for point in reversed(filler):
                MPDS_liquidus.insert(i + 1, point)

    # filter out duplicate values in the liquidus curve; greatly improves runtime efficiency
    for i in reversed(range(len(MPDS_liquidus) - 1)):
        if MPDS_liquidus[i][0] == 0 or MPDS_liquidus[i][1] == 0:
            continue
        if abs(1 - MPDS_liquidus[i + 1][0] / MPDS_liquidus[i][0]) < 0.0005 and \
                abs(1 - MPDS_liquidus[i + 1][1] / MPDS_liquidus[i][1]) < 0.0005:
            del (MPDS_liquidus[i + 1])

    return MPDS_liquidus


# dft_types = ["GGA", "GGA/GGA+U", "R2SCAN", "GGA/GGA+U/R2SCAN"]

# returns the DFT convex hull of a given system with specified functionals
def get_dft_convexhull(components, verbose=False):
    dft_type = "GGA/GGA+U"

    if 'Yb' in components:
        dft_type = "GGA"
    if verbose:
        print("using DFT entries solved with", dft_type, "functionals")

    components.sort()
    sys = '-'.join(components)
    dft_entries_file = os.path.join(f"{data_dir}/{sys}", f"{sys}_ENTRIES_MP_GGA.json")

    if os.path.exists(dft_entries_file):
        with open(dft_entries_file, "r") as f:
            dft_entries = json.load(f)

            # if Mg149 phase in system - remove
            dft_entries = [e for e in dft_entries if e['composition'].get('Mg', 0) != 149]

        try:
            pd = PhaseDiagram(elements=[Element(c) for c in components],
                              entries=[ComputedEntry.from_dict(e) for e in dft_entries])
            if verbose:
                print(len(pd.stable_entries) - 2, "stable line compound(s) on the DFT convex hull\n")
            return pd
        except ValueError as e:
            print(f"error loading DFT entries from cache: {e}")

    # no cache or invalid cached data
    entries = []

    # using legacy MP energies (GGA)
    if dft_type == "GGA":
        with Legacy_MPRester(Legacy_MP_api_key) as MPR:
            entries = MPR.get_entries_in_chemsys(components, inc_structure=True)

    # using new MP energies (GGA/GGA+U, R2SCAN, GGA/GGA+U/R2SCAN)
    else:
        with MPRester(New_MP_api_key) as MPR:
            # if dft_type == "R2SCAN" or dft_type == "GGA/GGA+U/R2SCAN":
            #     scan_entries = MPR.get_entries_in_chemsys(components,
            #                                               additional_criteria={
            #                                                   'thermo_types': [ThermoType.R2SCAN]})
            if dft_type == "GGA/GGA+U" or dft_type == "GGA/GGA+U/R2SCAN":
                gga_entries = MPR.get_entries_in_chemsys(components,
                                                         additional_criteria={
                                                             'thermo_types': [ThermoType.GGA_GGA_U]})

        # if dft_type == "GGA/GGA+U/R2SCAN":
        #     entries = MaterialsProjectDFTMixingScheme().process_entries(scan_entries + gga_entries,
        #                                                                 verbose=verbose)
        # if dft_type == "GGA/GGA+U":
        # entries = MaterialsProjectDFTMixingScheme().process_entries(gga_entries, verbose=verbose)
        # elif dft_type == "R2SCAN":
        #     entries = MaterialsProjectDFTMixingScheme().process_entries(scan_entries, verbose=verbose)

    if verbose:
        print(f"caching DFT entry data as {dft_entries_file}...")
    dft_entries = [e.as_dict() for e in entries]

    # if Mg149 phase in system - remove
    dft_entries = [e for e in dft_entries if e['composition'].get('Mg', 0) != 149]

    for e in dft_entries:
        e.pop('structure')  # TODO: use structure in future analyses
        e.pop('data')
    with open(dft_entries_file, "w") as f:
        json.dump(dft_entries, f)

    try:
        pd = PhaseDiagram(elements=[Element(c) for c in components],
                          entries=[ComputedEntry.from_dict(e) for e in dft_entries])
        if verbose:
            print(len(pd.stable_entries) - 2, "stable line compound(s) on the DFT convex hull\n")
        return pd
    except ValueError as e:
        print(f"error with DFT entries downloaded from API: {e}")
        return None


def identify_MPDS_phases(MPDS_json, verbose=False):
    if MPDS_json['reference'] is None:
        if verbose:
            print("system JSON does not contain any data!\n")
        return []

    phases = []
    data = ""
    for shape in MPDS_json['shapes']:

        if 'nphases' in shape and 'is_solid' in shape:
            # indentify line compounds and single-phase solid solutions
            if shape['nphases'] == 1 and shape['is_solid'] and 'label' in shape:
                # if '(' in shape['label'].split(' ')[0]:
                #     continue

                # split svgpath into tags, ordered pairs
                data = shape_to_list(shape['svgpath'])

                if not data:
                    if verbose:
                        print(f"no point data found for phase {shape['label']} in JSON!")
                    continue

                # sort by ascending T value
                data.sort(key=lambda x: x[1])
                tbounds = [data[0], data[-1]]
                # comp = Composition(shape['label'].split(' ')[0]).fractional_composition.as_dict()[components[1]]
                if shape['kind'] == 'phase':
                    data.sort(key=lambda x: x[0])
                    cbounds = [data[0], data[-1]]
                    if cbounds[-1][0] < 0.03 or cbounds[0][0] > 0.97:  # TODO: determine if this threshold works
                        continue

                    phases.append({'type': 'ss', 'name': shape['label'].split(' ')[0], 'comp': tbounds[1][0],
                                   'cbounds': cbounds, 'tbounds': tbounds})
                else:  # kind == compound
                    phases.append({'type': 'lc', 'name': shape['label'].split(' ')[0], 'comp': tbounds[1][0],
                                   'tbounds': tbounds})

    if not data:
        if verbose:
            print("no phase data found in JSON!")
        return phases

    phases.sort(key=lambda x: x['comp'])
    return phases


def get_low_temp_phase_data(mpds_json, mp_ch):
    mpds_congruent_phases = {}
    mpds_incongruent_phases = {}
    max_phase_temp = 0

    identified_phases = identify_MPDS_phases(mpds_json)
    mpds_liquidus = extract_MPDS_liquidus(mpds_json, verbose=False)

    def phase_decomp_on_liq(phase, liq):
        if liq is None:
            return False
        for i in range(len(liq) - 1):
            if liq[i][0] == phase['tbounds'][1][0]:
                return abs(liq[i][1] - phase['tbounds'][1][1]) < 10
            # composition falls between two points:
            elif liq[i][0] < phase['tbounds'][1][0] < liq[i + 1][0]:
                return abs((liq[i][1] + liq[i + 1][1]) / 2 - phase['tbounds'][1][1]) < 10

    for phase in identified_phases:
        # Check to see if these are low temperature phases (phase lower bound must be within lower 10% of temp range)
        if (phase['type'] in ['lc', 'ss'] and phase['tbounds'][0][1] < (mpds_json['temp'][0] + 273.15) +
                (mpds_json['temp'][1] - mpds_json['temp'][0]) * 0.10):
            if phase_decomp_on_liq(phase, mpds_liquidus):
                if phase['type'] == 'ss':
                    mpds_congruent_phases[phase['name']] = (
                        (phase['cbounds'][0][0], phase['cbounds'][1][0]), phase['tbounds'][1][1])
                else:
                    mpds_congruent_phases[phase['name']] = ((phase['comp'], phase['comp']), phase['tbounds'][1][1])
            else:
                if phase['type'] == 'ss':
                    mpds_incongruent_phases[phase['name']] = (
                        (phase['cbounds'][0][0], phase['cbounds'][1][0]), phase['tbounds'][1][1])
                else:
                    mpds_incongruent_phases[phase['name']] = (
                        (phase['comp'], phase['comp']), phase['tbounds'][1][1])
            max_phase_temp = max(phase['tbounds'][1][1], max_phase_temp)

    if max_phase_temp == 0 and mpds_liquidus:
        asc_temp = sorted(mpds_liquidus, key=lambda x: x[1])
        max_phase_temp = asc_temp[0][1]

    mp_phases = {}
    mp_phases_ebelow = {}
    min_form_e = 0

    for entry in mp_ch.stable_entries:
        # skip pure components
        if len(entry.composition.fractional_composition.as_dict()) == 1:
            continue
        # TODO: change this so the ordering is flexible
        comp = entry.composition.fractional_composition.as_dict()[mp_ch.elements[1].symbol]

        form_e = mp_ch.get_form_energy_per_atom(entry)
        mp_phases[entry.name] = ((comp, comp), form_e)
        min_form_e = min(form_e, min_form_e)

        ch_copy = PhaseDiagram([e for e in mp_ch.stable_entries if e != entry])
        e_below_hull = -abs(mp_ch.get_hull_energy_per_atom(entry.composition) -
                            ch_copy.get_hull_energy_per_atom(entry.composition))
        mp_phases_ebelow[entry.name] = ((comp, comp), e_below_hull)

    return [mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp], [mp_phases, mp_phases_ebelow, min_form_e]

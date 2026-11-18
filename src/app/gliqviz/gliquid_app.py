'''
Authors: Abrar Rauf & Joshua Willwerth

The Dash app of the G-Liquid ternary interpolation: a user enters a ternary system and gets
its interpolated liquidus and the three binary phase diagrams.

This package is not named ``gliquid``: /app is the WORKDIR and first on ``sys.path``, so that
name would shadow the installed package.
'''
import dash
import functools
from urllib.parse import quote
from dash import dcc, html
from dash.dependencies import ALL, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import threading
import ctypes
import sys
import flask
import gliquid.config as gliquid_config
from gliquid.cache import resolve_backend
from gliquid.plotting.binary_tx import CORNER_NOTE_NAME
from gliquid.plotting.style import abbreviate_phase_name, standard_formula
from gliquid.plotting.ternary_surface import (IMPUTED_SUFFIX, MODIFIED_SUFFIX,
                                              USER_DEFINED_SUFFIX, dash_clientside_function)
from pymatgen.core import Composition, Element
from gliquid.api import hull_entry_name
from gliquid.cache import non_dft_entry_kind
from gliquid.phase import EV_ATOM_TO_J_MOL
from gliquid.ternary import PRESCREEN_CEILING_K, TernaryLiquidInterpolation, TLIPlotter
import math
import re
import warnings
import base64
import pandas as pd
import json
import uuid
import traceback
from pathlib import Path
from collections import OrderedDict
import time

# --- app paths -----------------------------------------------------------------------
# Data and code share this package directory, which main.py mounts at /gliquid/.
SITE_DIR = Path(__file__).resolve().parent
if not (SITE_DIR / "params").is_dir():
    raise RuntimeError(
        f"The /gliquid/ data is missing from {SITE_DIR}: no params/ directory. "
        f"main.py mounts this package as StaticFiles(directory='gliqviz')."
    )

# The DFT and digitized-MPDS corpus as one read-only SQLite file.
CACHE_STORE = SITE_DIR / "gliquid_cache.sqlite"
# Fitted and ML-predicted Redlich-Kister parameters.
PARAMS_DIR = SITE_DIR / "params"

# --- gliquid configuration -----------------------------------------------------------
gliquid_config.set_cache_dir(CACHE_STORE)  # a .sqlite path also sets cache_mode='sqlite'
gliquid_config.set_offline(True)

# --- app-wide model settings ---------------------------------------------------------
# The system shown before the user has typed one, rendered from the store once at startup.
LANDING_SYSTEM = "Ce-Fe-Si"

INTERP_TYPE = 'linear'
PARAM_FORMAT = 'combined'
DELTA = 0.025
# Hull evaluation backend of TernaryLiquidInterpolation.
HULL_MODE = 'gtx'

# The DFT functional TernaryLiquidInterpolation reads ternary entries for.
TERNARY_DFT_TYPE = "GGA"


def parse_system(text):
    """The elements typed into the system box, sorted, with blanks around and between dashes
    dropped: ``" Sn-Bi - Cd"`` -> ``['Bi', 'Cd', 'Sn']``."""
    return sorted(part.strip() for part in (text or "").split("-") if part.strip())


def describe_system_problem(components):
    """Return a user-facing reason sorted ``components`` cannot be plotted, or None if it can.

    Checks, in order: that there are three distinct valid element symbols, that the store's
    ternary entry pool covers the system, and that each binary edge has fitted or predicted
    parameters.
    """
    if not components:
        return "Enter a ternary system, e.g. Bi-Cd-Sn."
    if len(components) != 3:
        return (f"Enter three elements separated by dashes, e.g. Bi-Cd-Sn. "
                f"Got {len(components)}: {'-'.join(components)}")
    bad = [e for e in components if not Element.is_valid_symbol(e)]
    if bad:
        return f"Not valid element symbols: {', '.join(bad)}. Use e.g. Bi-Cd-Sn."
    if len(set(components)) != 3:
        return f"Three DIFFERENT elements are required; got {'-'.join(components)}."

    sys_name = "-".join(sorted(components))
    pool_covers = getattr(resolve_backend(CACHE_STORE), "pool_covers", None)
    if pool_covers is not None and not pool_covers(sys_name, TERNARY_DFT_TYPE):
        return f"{sys_name} is not in this server's ternary DFT dataset."
    missing = [edge for edge in binary_edges(sorted(components))
               if find_edge_parameters(edge) is None]
    if missing:
        return f"No fitted or predicted parameters for {', '.join(missing)}."
    return None


def require_landing_system_covered(system):
    """Raise ``RuntimeError`` if ``system`` is not in this store's ternary entry pool.

    Ternary DFT hulls come only from the pool, and the app runs offline, so an uncovered
    landing system cannot be rendered. ``render_landing_figures`` calls this outside its
    fallback, so the error stops the boot.
    """
    sys_name = "-".join(sorted(system.split("-")))
    backend = resolve_backend(CACHE_STORE)
    pool_covers = getattr(backend, "pool_covers", None)
    if pool_covers is None:
        # A directory store has no pool; the render reports what it finds.
        return
    if pool_covers(sys_name, TERNARY_DFT_TYPE):
        return

    covered = [row[0] for row in backend.pool_systems()]
    raise RuntimeError(
        f"LANDING_SYSTEM {system!r} (resolved to {sys_name!r}) is NOT covered by the "
        f"ternary entry pool in {CACHE_STORE}.\n"
        f"The app runs offline, so the missing entries cannot be fetched.\n"
        f"The pool covers {len(covered)} ternary system(s)"
        + (f", e.g. {', '.join(covered[:5])}" if covered else "")
        + ".\n"
        f"Set LANDING_SYSTEM to one of those, or rebuild the store with a pool that "
        f"covers {sys_name}."
    )


# --- Redlich-Kister parameter lookup -------------------------------------------------
# The columns both parameter files must have.
_REQUIRED_PARAM_COLS = ('system', 'L0_a', 'L0_b', 'L1_a', 'L1_b')
FITTED_PARAMS = "fitted_params.xlsx"
PREDICTED_PARAMS = "predicted_params.xlsx"


@functools.lru_cache(maxsize=None)
def read_param_table(name):
    """The parameter table ``PARAMS_DIR / name``, read once per process. Shared: never modify it."""
    df = pd.read_excel(PARAMS_DIR / name)
    missing = set(_REQUIRED_PARAM_COLS) - set(df.columns)
    assert not missing, f"{name} is missing columns: {missing}"
    return df


def binary_edges(components):
    """The cyclic edges ``A-B``, ``B-C``, ``C-A`` of sorted ``components``."""
    a, b, c = components
    return [f"{a}-{b}", f"{b}-{c}", f"{c}-{a}"]


def find_edge_parameters(edge):
    """``(row, "fit" | "pred")``: the parameter row of binary ``edge``, stored in either element
    order, from the fitted table if it has one, else the predicted table. None if neither has
    the edge."""
    keys = (edge, "-".join(sorted(edge.split("-"))))
    for name, kind in ((FITTED_PARAMS, "fit"), (PREDICTED_PARAMS, "pred")):
        df = read_param_table(name)
        for key in keys:
            rows = df[df['system'] == key]
            if len(rows):
                return rows.iloc[0], kind
    return None


def lookup_binary_parameters(components):
    """``(binary_L_dict, fit_or_pred)`` for the three binary edges of ``components``.

    Called by both render paths: the landing figures and the figures of a typed system.

    ``components`` must already be sorted. Keys are the cyclic edges of that sorted
    system -- ``A-B``, ``B-C``, ``C-A`` -- the package's ``xs_mix`` convention. The third
    edge is not alphabetical: its parameters are stored under the inverse pair, so ``L1`` is
    negated to orient them onto the edge as keyed. ``L0`` is symmetric.

    Fitted parameters are used over predicted ones for a system that has both.
    """
    assert all(len(e) <= 2 and e.isalpha() for e in components), \
        f"Invalid element symbols: {components}"

    binary_sys_labels = binary_edges(components)
    print("Binary System Labels: ", binary_sys_labels)

    binary_L_dict = {}
    fit_or_pred = {}

    for bin_sys in binary_sys_labels:
        found = find_edge_parameters(bin_sys)
        if found is None:
            raise ValueError(f"Binary system {bin_sys} not found in the parameter dataframe.")
        params, fit_or_pred[bin_sys] = found
        key = params['system']

        L0_a, L0_b = float(params["L0_a"]), float(params["L0_b"])
        L1_a, L1_b = float(params["L1_a"]), float(params["L1_b"])

        # L1 is odd under component swap: negate when the matched row's order differs from
        # the edge's. Compare the matched row's key, not the requested name.
        if key != bin_sys:
            L1_a, L1_b = -L1_a, -L1_b

        binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]

    print(fit_or_pred)
    print("Binary Interaction Parameters: ", binary_L_dict)

    # A repeated element ("Fe-Fe-Fe") collapses the edges onto one key.
    assert len(binary_L_dict) == 3, \
        f"Expected 3 binary systems in L_dict, got {len(binary_L_dict)}: {list(binary_L_dict)}"

    return binary_L_dict, fit_or_pred


SUB_WIDTH, SUB_HEIGHT = 480, 360
# The top margin holds the MAPE chip a fitted edge carries above the plot frame.
SUB_MARGIN = dict(t=24, b=50, l=52, r=12)
# Fonts for the binary sub-plot: on-plot phase/polymorph/compound labels, the liquid region
# label, the component names under the axis, and the MAPE chip.
SUB_LABEL_FONT, SUB_L_FONT, SUB_COMPONENT_FONT, SUB_NOTE_FONT = 9, 11, 11, 10
# The component names sit SUB_COMPONENT_DY px below the axis, at this paper-y.
SUB_COMPONENT_DY = 23
SUB_COMPONENT_Y = -SUB_COMPONENT_DY / (SUB_HEIGHT - SUB_MARGIN['t'] - SUB_MARGIN['b'])
# The ternary scene's x and y aspect ratios are TERNARY_XY_STRETCH times gliquid's, against an
# unchanged z, so the box is wider than tall.
TERNARY_XY_STRETCH = 1.3

# Page layout, in px. The sidebar's width includes its padding and scrollbar gutter and holds
# IDLE_MESSAGE on one line. PLOT_GAP separates the sidebar, the binary column and the ternary
# plot, which sits TERNARY_TOP below the top of the binary column. Nothing scales with the window.
SIDEBAR_WIDTH, SIDEBAR_PADDING = 330, 10
PLOT_GAP = 20
# Height of the liquidus key above the binary column, below LEGEND_BAR_TOP of padding.
LEGEND_BAR_HEIGHT, LEGEND_BAR_TOP = 24, 10
# The ternary scene spans the binary plot frames, from the top of the first to the bottom of
# the last, below a title band TERNARY_TITLE_MARGIN high; the plot's width/height is
# TERNARY_ASPECT.
TERNARY_ASPECT = 950 / 900
TERNARY_TITLE_MARGIN = 40
TERNARY_TOP = LEGEND_BAR_TOP + LEGEND_BAR_HEIGHT + SUB_MARGIN['t'] - TERNARY_TITLE_MARGIN
TERNARY_HEIGHT = (LEGEND_BAR_TOP + LEGEND_BAR_HEIGHT + 3 * SUB_HEIGHT - SUB_MARGIN['b']
                  - TERNARY_TOP)
TERNARY_WIDTH = round(TERNARY_ASPECT * TERNARY_HEIGHT)


#: The font of the page and of its plots.
FONT = "Arial, Helvetica, sans-serif"


def style_binary_subplot(fig):
    """Shrink a package binary phase diagram to a ternary-app sub-plot.

    ``plot_tx`` sets explicit font sizes on its titles and labels, which a figure-level
    ``font`` does not override, so each is resized here.
    """
    fig.update_layout(
        showlegend=False,
        width=SUB_WIDTH,
        height=SUB_HEIGHT,
        font=dict(family=FONT, size=10, color='black'),
        title=None,
        margin=SUB_MARGIN,
    )
    fig.update_xaxes(title_font=dict(size=10))
    fig.update_yaxes(title_font=dict(size=10))

    # 100 K of headroom above the plotted range.
    rng = fig.layout.yaxis.range
    if rng is not None:
        fig.update_yaxes(range=[rng[0], rng[1] + 100])

    for a in fig.layout.annotations:
        if a.name == CORNER_NOTE_NAME:
            # The MAPE chip keeps the package's anchor above the plot frame's top-left corner.
            a.font.size = SUB_NOTE_FONT
        elif a.xref == "paper" and a.yref == "paper":
            # The component names, centred under the 0 and 100 composition ticks.
            a.x = 0.0 if a.x < 0.5 else 1.0
            a.xanchor = "center"
            a.yanchor = "top"
            a.y = SUB_COMPONENT_Y
            a.font.size = SUB_COMPONENT_FONT
        elif a.text == "L":
            a.font.size = SUB_L_FONT
        else:
            a.font.size = SUB_LABEL_FONT
    return fig


# --- phase editor ------------------------------------------------------------------
#: Most phases a tab may add to one system.
MAX_ADDED_PHASES = 10
#: Largest move, K, of a locked temperature that an update may make.
LOCK_TOLERANCE_K = 5.0
#: Largest gap, K, between a temperature asked for and the end of the phase's drawn field at
#: which the phase list shows the temperature asked for: a small field's ends lie between the
#: vertices of the drawn surface.
ASKED_TOLERANCE_K = 25.0
#: Temperatures a phase may be given, C.
TEMPERATURE_LIMITS = (-273.0, 5000.0)
#: Largest denominator tried when a composition is named as a whole-number ratio, and how far
#: each fraction may lie from that ratio.
NAME_DENOMINATOR, NAME_TOLERANCE = 40, 1e-3
#: Largest fraction difference at which an added phase sits on a phase already there.
SAME_COMPOSITION = 1e-3
#: Phase names a user may type.
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 ().+_-]{0,29}")


#: The phase edits made since the plots on screen were drawn, when there are none: phases
#: added (as read_added_phase returns them), names marked for removal, and names of removed
#: phases to keep again.
NO_PENDING = {"added": [], "removed": [], "restored": []}


#: The four values the editor shows for a phase, in row order: the temperatures in C, the
#: formation enthalpy in eV/atom and the entropy in J/mol-atom/K.
FIELDS = ("solidus", "liquidus", "enthalpy", "entropy")
#: What a message calls each of FIELDS.
FIELD_NAMES = {"solidus": "min liquidus temp", "liquidus": "max liquidus temp",
               "enthalpy": "formation enthalpy", "entropy": "entropy"}
#: The key ``edit_phases`` takes each of FIELDS under.
FIELD_KEYS = {"solidus": "solidus", "liquidus": "temperature",
              "enthalpy": "enthalpy", "entropy": "entropy"}
#: The unit systems the editor shows an enthalpy and an entropy in. Rows and targets hold
#: eV/atom and J/mol-atom/K.
UNITS = ("eV/atom", "J/mol")
#: The unit of an enthalpy and an entropy in each of UNITS, both per mole of atoms in J/mol.
UNIT_LABELS = {"eV/atom": {"enthalpy": "eV/atom", "entropy": "meV/atom·K"},
               "J/mol": {"enthalpy": "kJ/mol", "entropy": "J/mol·K"}}
#: What an enthalpy (eV/atom) and an entropy (J/mol-atom/K) are multiplied by to show them in
#: each of UNITS.
UNIT_SCALES = {"eV/atom": {"enthalpy": 1.0, "entropy": 1000.0 / EV_ATOM_TO_J_MOL},
               "J/mol": {"enthalpy": EV_ATOM_TO_J_MOL / 1000.0, "entropy": 1.0}}
#: Decimal places the editor shows an enthalpy and an entropy to, by unit.
DECIMALS = {"eV/atom": {"enthalpy": 4, "entropy": 4}, "J/mol": {"enthalpy": 2, "entropy": 2}}
#: Enthalpies (eV/atom) and entropies (J/mol-atom/K) a phase may be given.
ENTHALPY_LIMITS, ENTROPY_LIMITS = (-10.0, 10.0), (-200.0, 200.0)
#: How many of a phase's FIELDS can be set at once: the rest follow from them.
FREE_FIELDS = 2
#: A formation enthalpy within this of the 0 K hull, J/mol-atom, is not on it.
HULL_CLEARANCE_J = 1.0


def no_edits():
    """A system's phases as the store holds them: nothing added, changed or removed. ``added``
    holds phases as read_added_phase returns them, ``targets`` maps a changed phase to its
    target, and ``removed`` names the phases left out. A target sets one or two of FIELDS,
    ``{field: value}``; a liquidus alone leaves the phase stable from 0 K with no entropy. A
    target's ``'force'`` marks a phase read from an uploaded list."""
    return {"added": [], "targets": {}, "removed": []}


class EditStopped(Exception):
    """A phase edit that leaves the plots on screen."""


def shown_value(which, value, units):
    """A value of field ``which`` as the editor shows it in ``units`` (one of UNITS): a
    temperature to the nearest whole degree, halves up, an enthalpy and an entropy converted
    and rounded to DECIMALS."""
    if value is None:
        return None
    if which in ("solidus", "liquidus"):
        return math.floor(value + 0.5)
    return round(value * UNIT_SCALES[units][which], DECIMALS[units][which]) + 0.0


def shown_values(values):
    """``values`` (field -> value) as the editor shows them, by unit."""
    return {units: {which: shown_value(which, values.get(which), units) for which in FIELDS}
            for units in UNITS}


def phase_name(fractions):
    """The formula naming element ``fractions``: on whole numbers where every fraction lies
    within NAME_TOLERANCE of a ratio with a denominator up to NAME_DENOMINATOR."""
    present = {el: f for el, f in fractions.items() if f > 0}
    for n in range(1, NAME_DENOMINATOR + 1):
        counts = {el: f * n for el, f in present.items()}
        if all(round(c) >= 1 and abs(c - round(c)) <= NAME_TOLERANCE * n
               for c in counts.values()):
            return Composition({el: round(c) for el, c in counts.items()}).reduced_formula
    return Composition(present).reduced_formula


def read_fractions(amounts, elements):
    """``(fractions, problem)``: the element fractions three typed ``amounts`` stand for, in the
    order of ``elements``, or why they name no compound."""
    try:
        amounts = [0.0 if a in (None, "") else float(a) for a in amounts]
    except (TypeError, ValueError):
        return None, "Enter the composition as numbers."
    if len(amounts) != 3 or not all(math.isfinite(a) and a >= 0 for a in amounts):
        return None, "Enter the composition as numbers of zero or more."
    total = sum(amounts)
    if total <= 0 or sum(a > 0 for a in amounts) < 2:
        return None, "A phase needs at least two of the elements."
    return {el: a / total for el, a in zip(elements, amounts)}, None


def read_temperature(value):
    """``(degrees C, problem)`` of a typed temperature."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None, "Enter the temperature in \u00b0C."
    if not TEMPERATURE_LIMITS[0] < value < TEMPERATURE_LIMITS[1]:
        return None, (f"The temperature must lie between {TEMPERATURE_LIMITS[0]:.0f} and "
                      f"{TEMPERATURE_LIMITS[1]:.0f} \u00b0C.")
    return value, None


def read_value(which, value, units):
    """``(value, problem)`` of a value of field ``which`` typed in ``units``: a temperature in
    C, an enthalpy in eV/atom, an entropy in J/mol-atom/K."""
    if which in ("solidus", "liquidus"):
        return read_temperature(value)
    unit = UNIT_LABELS[units][which]
    try:
        value = float(value) / UNIT_SCALES[units][which]
    except (TypeError, ValueError):
        return None, f"Enter the {FIELD_NAMES[which]} in {unit}."
    low, high = ENTHALPY_LIMITS if which == "enthalpy" else ENTROPY_LIMITS
    if not low < value < high:
        return None, (f"The {FIELD_NAMES[which]} must lie between "
                      f"{shown_value(which, low, units):g} and "
                      f"{shown_value(which, high, units):g} {unit}.")
    return value, None


def read_target(given, units):
    """``(target, problem)``: the target (see no_edits) the values ``given`` (field -> value
    typed in ``units``, blanks included) ask for an added phase: two of ADD_FIELDS, or a
    liquidus or an enthalpy alone."""
    target = {}
    for which in ADD_FIELDS:
        if given.get(which) in (None, ""):
            continue
        target[which], problem = read_value(which, given[which], units)
        if problem:
            return None, problem
    if set(target) == {"solidus"}:
        return None, "Give the max liquidus temp or ΔH with the min liquidus temp."
    if not target or len(target) > FREE_FIELDS:
        return None, "Give one or two of the min liquidus temp, the max liquidus temp and ΔH."
    if set(target) == {"enthalpy"}:
        target["entropy"] = 0.0
    if set(target) == {"solidus", "liquidus"} and not target["solidus"] < target["liquidus"]:
        return None, "The min liquidus temp must lie below the max."
    return target, None


def read_added_phase(amounts, elements, name, given, units, rows, taken):
    """``(phase, problem)``: the phase the add form describes, as ``{'name', 'fractions',
    'target'}`` with the fractions keyed by element and the target read_target reads from
    ``given`` and ``units``, or why it cannot be added. ``rows`` are the phases on screen and
    ``taken`` the names in use."""
    fractions, problem = read_fractions(amounts, elements)
    if problem:
        return None, problem
    target, problem = read_target(given, units)
    if problem:
        return None, problem
    ordered = sorted(elements)
    for row in rows:
        gap = max(abs(fractions[el] - f) for el, f in zip(ordered, row["fractions"]))
        if gap <= SAME_COMPOSITION:
            return None, f"{row['name']} already sits at this composition: edit it instead."
    name = (name or "").strip() or phase_name(fractions)
    if not NAME_PATTERN.fullmatch(name):
        return None, "Use up to 30 letters, digits, spaces and ( ) . + _ - in the name."
    if name in taken:
        return None, f"{name} already names a phase: give this one another name."
    return {"name": name, "fractions": fractions, "target": target}, None


def gliquid_target(target):
    """A ``target`` (see no_edits) as the keys ``edit_phases`` takes: temperatures in K, the
    enthalpy in J/mol-atom."""
    out = {}
    for which in FIELDS:
        value = target.get(which)
        if value is None:
            continue
        if which in ("solidus", "liquidus"):
            value += 273.15
        elif which == "enthalpy":
            value *= EV_ATOM_TO_J_MOL
        out[FIELD_KEYS[which]] = value
    if target.get("force"):
        out["force"] = True
    return out


def edit_list(edits):
    """A session's ``edits`` as the list ``TernaryLiquidInterpolation.edit_phases`` takes."""
    added = {phase["name"] for phase in edits["added"]}
    out = [{"remove": name} for name in edits["removed"]]
    out += [{"phase": name, **gliquid_target(target)}
            for name, target in edits["targets"].items()
            if name not in edits["removed"] and name not in added]
    out += [{"composition": phase["fractions"], "name": phase["name"],
             **gliquid_target(phase["target"])} for phase in edits["added"]]
    return out


#: Why a phase edit was refused, by gliquid's rejection code.
REFUSALS = {
    "below_solidus": "no liquid reaches its composition at that temperature",
    "unstable": "at that temperature it would not be stable at 0 K",
    "not_stable": "it would not be stable once the other edits are in",
    "no_entropy": ("the enthalpy that would bring it onto the hull at that min liquidus temp is "
                   "on the 0 K hull"),
    "name_taken": "its name is already in use",
    "dft_covers": "a phase already sits at its composition",
    "occupied": "a phase already sits at its composition",
}


def shown_amount(which, value, units):
    """An enthalpy (J/mol-atom) or entropy (J/mol-atom/K) ``value`` as a message gives it in
    ``units``, with its unit."""
    if which == "enthalpy":
        value /= EV_ATOM_TO_J_MOL
    return f"{shown_value(which, value, units):g} {UNIT_LABELS[units][which]}"


def refusal(record, edit, units):
    """Why the phase ``edit`` of ``record`` is not applied, for the sidebar with the energies
    in ``units``; None for one that is. ``edit`` is keyed as ``edit_phases`` takes it."""
    def nearest(text, otherwise):
        limit = record.solidus_limit_k
        return otherwise if limit is None else text.format(limit - 273.15)

    if record.accepted:
        top = record.temperature_k
        if top is not None and record.melts is False:
            return (f"no liquid reaches its composition at {top - 273.15:.0f} \u00b0C, so that "
                    f"cannot be its max liquidus temp")
        floats = record.enthalpy >= record.hull_enthalpy - HULL_CLEARANCE_J
        if floats and record.solidus_k is None and not record.entropy > 0.0:
            hull = shown_amount("enthalpy", record.hull_enthalpy, units)
            return (f"its \u0394H of {shown_amount('enthalpy', record.enthalpy, units)} is not below "
                    f"the 0 K hull ({hull}) and its \u0394S is not positive, so it is never "
                    f"stable; give a \u0394H below {hull} or a positive \u0394S")
        return None
    if "enthalpy" in edit or "entropy" in edit:
        if record.code == "no_entropy" and "0 K hull" in record.reason:
            return (f"that \u0394H is on the 0 K hull, so the phase is stable below any min "
                    f"liquidus temp; a \u0394H above "
                    f"{shown_amount('enthalpy', record.hull_enthalpy, units)} is off it")
        if record.code == "no_entropy" and record.temperature_k is None:
            return ("with that \u0394S it would leave the hull at the min liquidus temp, not "
                    "come onto it")
        if record.code == "no_entropy":
            return ("with that \u0394S it would come onto the hull at the max liquidus temp, "
                    "not leave it")
    if record.solidus_k is not None and record.code == "below_solidus":
        return ("no liquid reaches its composition at that min liquidus temp; "
                + nearest("the lowest that works is {:.0f} \u00b0C", "no higher one works either"))
    if record.solidus_k is not None and record.code == "no_entropy" and "0 K hull" not in record.reason:
        return ("no entropy keeps it on the hull only between those liquidus temps; "
                + nearest("the highest min liquidus temp that works is {:.0f} \u00b0C",
                          "its min liquidus temp cannot be moved"))
    return REFUSALS.get(record.code, record.reason)


def too_stable(ti, record, units):
    """Why an accepted ``record`` of ``ti`` is not applied for its entropy, with the energies in
    ``units``: a line phase whose entropy keeps it below the liquid at its composition up to
    the top of TEMPERATURE_LIMITS; None for one that melts below it. Names the entropy it
    must stay below and the one at and above which it never melts (below the model's
    PRESCREEN_CEILING_K)."""
    if not record.accepted or not record.entropy > 0.0:
        return None
    x = [record.fractions[element] for element in ti.components[1:]]
    melts, never = ((record.enthalpy - ti.liquid_gibbs(x, top)) / top
                    for top in (TEMPERATURE_LIMITS[1] + 273.15, PRESCREEN_CEILING_K))
    if record.entropy < melts:
        return None
    return (f"it would not melt below {TEMPERATURE_LIMITS[1]:.0f} °C; give a ΔS below "
            f"{shown_amount('entropy', melts, units)} (at "
            f"{shown_amount('entropy', never, units)} or more it never melts)")


def displacement_note(record, edit, units):
    """What the sidebar says, with the energies in ``units``, of an applied ``record`` that
    pushes phases off the 0 K hull, or whose enthalpy was held where it keeps them; None for
    any other. ``edit`` is keyed as ``edit_phases`` takes it."""
    if not record.displaces:
        return None
    names = ", ".join(record.displaces)
    them = "it" if len(record.displaces) == 1 else "them"
    top = None if record.temperature_k is None else f"{record.temperature_k - 273.15:.0f} \u00b0C"
    if record.enthalpy != record.inverted_enthalpy:
        return (f"{record.name} held at \u0394H {shown_amount('enthalpy', record.enthalpy, units)}, "
                f"\u0394S {shown_amount('entropy', record.entropy, units)} to keep {names} on the "
                f"0 K hull.")
    note = f"{names} pushed off the 0 K hull by {record.name}."
    limit = record.hull_limit_enthalpy
    if limit is None:
        return f"{note} No \u0394H of {record.name} keeps {them}."
    keeps = f"\u0394H \u2265 {shown_amount('enthalpy', limit, units)} keeps {them}"
    if "enthalpy" in edit or top is None:
        return f"{note} {keeps}."
    floor = (limit - record.enthalpy) / record.temperature_k + record.entropy
    if "entropy" in edit and floor < (record.support_entropy or 0.0):
        return f"{note} \u0394S \u2265 {shown_amount('entropy', floor, units)} keeps {them} at {top}."
    return f"{note} {keeps}, but then {top} is out of reach."


def apply_edits(ti, edits, units=UNITS[0]):
    """Apply a session's ``edits`` to ``ti`` and return what the sidebar says of them, with the
    energies in ``units``: one note per phase that pushes others off the 0 K hull or was held
    at the enthalpy that keeps them. A phase given a max liquidus temp alone is held where an
    entropy still reaches that temperature; any other is placed as asked. Raises EditStopped
    for an edit that cannot be applied."""
    asked = edit_list(edits)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            records = ti.edit_phases(asked, screen=False, write_cache=False, hold_hull=True,
                                     force=True)
    except ValueError as error:
        raise EditStopped(f"Not applied: {error}") from error
    placed = [edit for edit in asked if "remove" not in edit]
    refused = [f"{record.name}: {why}." for record, edit in zip(records, placed)
               if (why := refusal(record, edit, units) or too_stable(ti, record, units))]
    if refused:
        raise EditStopped("Not applied. " + " ".join(refused))
    return [note for record, edit in zip(records, placed)
            if (note := displacement_note(record, edit, units))]


def phase_rows(plotter, figure, edits, default=()):
    """The editor's rows for a processed ``plotter`` whose 'tx' ``figure`` is drawn: one per
    solid phase, in legend order, each ``{'name', 'label', 'tag', 'color', 'fractions',
    'solidus', 'liquidus', 'enthalpy', 'entropy', 'shown', 'solidus_set', 'editable', 'dft',
    'kind', 'changed'}``.

    The label is the legend's name for the phase and the tag what the legend adds to it for a
    modified, imputed or user-defined phase, and the color its legend marker's (None for a
    phase the legend does not list); the fractions follow the sorted elements, and
    the four values are those of FIELDS (a temperature is None for a phase with no liquidus
    field), with ``shown`` holding them as shown_values shows them. ``solidus_set`` marks a phase
    given a solidus; a compound on the 0 K hull, a changed phase and an added one are editable;
    ``dft`` marks a compound whose hull entry is a DFT one. A changed or added phase shows the
    temperatures asked for where its field ends within ASKED_TOLERANCE_K of them. ``changed``
    lists the FIELDS an edit left at a value other than the one shown in ``default``, the rows
    of the system without edits."""
    ti = plotter._ti
    ranges = plotter.liquidus_ranges()
    tags = {**dict.fromkeys(ti.user_phase_names(), USER_DEFINED_SUFFIX),
            **dict.fromkeys(ti.imputed_phase_names(), IMPUTED_SUFFIX),
            **dict.fromkeys(ti.modified_phase_names(), MODIFIED_SUFFIX)}
    stable = [entry for entry in ti.solid_hull().stable_entries
              if len(entry.composition.elements) > 1]
    compounds = {hull_entry_name(entry) for entry in stable}
    dft = {hull_entry_name(entry) for entry in stable
           if non_dft_entry_kind(entry) is None and not (entry.data or {}).get("imputed")}
    colors = {trace.meta["gliquid_phase"]: trace.marker.color for trace in figure.data
              if isinstance(trace.meta, dict) and trace.meta.get("gliquid_role") == "legend"}
    legend = list(colors)
    solids = ti.proc_df[ti.proc_df["Phase"] != "L"].drop_duplicates("Phase")
    added = {phase["name"] for phase in edits["added"]}
    names = list(solids["Phase"])
    asked = {**edits["targets"], **{phase["name"]: phase["target"] for phase in edits["added"]}}
    original = {row["name"]: row for row in default}
    rows = []
    for name, x0, x1, enthalpy, entropy in zip(names, solids["x0"], solids["x1"], solids["H"],
                                               solids["S"]):
        low, high = ranges.get(name, (None, None))
        target = asked.get(name, {})
        if high is not None and abs(high - target.get("liquidus", math.inf)) <= ASKED_TOLERANCE_K:
            high = target["liquidus"]
        solidus_set = target.get("solidus") is not None
        if low is not None and solidus_set and abs(low - target["solidus"]) <= ASKED_TOLERANCE_K:
            low = target["solidus"]
        kind = "added" if name in added else "changed" if name in edits["targets"] else "default"
        values = {"solidus": low, "liquidus": high,
                  "enthalpy": float(enthalpy) / EV_ATOM_TO_J_MOL, "entropy": float(entropy)}
        shown = shown_values(values)
        was = original.get(name, {}).get("shown")
        if kind == "default":
            changed = []
        elif was is None:
            changed = [which for which in FIELDS if which in target]
        else:
            changed = [which for which in FIELDS
                       if any(shown[units][which] != was[units][which] for units in UNITS)]
        rows.append({"name": str(name),
                     "label": abbreviate_phase_name(
                         name, names, None if tags.get(name) == USER_DEFINED_SUFFIX
                         else plotter.components),
                     "tag": tags.get(name, ""), "color": colors.get(str(name)),
                     "fractions": [float(1 - x0 - x1), float(x0), float(x1)],
                     **values, "shown": shown, "solidus_set": solidus_set,
                     "editable": name in compounds or name in asked, "dft": name in dft,
                     "kind": kind, "changed": changed})
    rows.sort(key=lambda row: legend.index(row["name"]) if row["name"] in legend
              else len(legend))
    return rows


def check_locks(rows, locks):
    """Raise EditStopped if a locked temperature of ``locks`` (``(name, 'solidus' or
    'liquidus', degrees C)``) lies more than LOCK_TOLERANCE_K from its value in ``rows``."""
    by_name = {row["name"]: row for row in rows}
    for name, which, value in locks:
        now = by_name.get(name, {}).get(which)
        if now is None:
            raise EditStopped(f"Not applied: the update would leave {name}, whose "
                              f"{FIELD_NAMES[which]} is locked, without a liquidus field.")
        if abs(now - value) > LOCK_TOLERANCE_K:
            raise EditStopped(f"Not applied: the update would move the locked "
                              f"{FIELD_NAMES[which]} of {name} from {value:.0f} to {now:.0f} "
                              f"\u00b0C.")


def ternary_model(components, binary_L_dict, fit_or_pred):
    """The ``TernaryLiquidInterpolation`` over ``components`` with the store's phases;
    ``binary_L_dict`` and ``fit_or_pred`` as build_ternary_plotter takes them."""
    return TernaryLiquidInterpolation(
        sorted(components),
        CACHE_STORE,
        order="alphabetical",
        delta=DELTA,
        interp_scheme=INTERP_TYPE,
        param_format=PARAM_FORMAT,
        xs_mix=binary_L_dict or {},
        fit_or_pred=fit_or_pred or {},
        hull_mode=HULL_MODE,
        include_imputed=True,
    )


def uploaded_edits(model, entries):
    """``(edits, notes)``: the edits (see no_edits) that give ``model``, a system with the
    store's phases, the phases of an uploaded list of ``entries``, and what the user should
    know about them."""
    found = model.entry_list_edits(entries)
    edits = no_edits()
    for edit in found.edits:
        if "remove" in edit:
            edits["removed"].append(edit["remove"])
            continue
        target = {"enthalpy": edit["enthalpy"] / EV_ATOM_TO_J_MOL, "entropy": edit["entropy"],
                  "force": True}
        if "phase" in edit:
            edits["targets"][edit["phase"]] = target
        else:
            edits["added"].append({"name": edit["name"], "fractions": edit["composition"],
                                   "target": target})
    notes = []
    if found.reference_shifts:
        shifts = ", ".join(f"{el} by {shift:+.4f} eV/atom"
                           for el, shift in found.reference_shifts.items())
        notes.append(f"The list's reference energies differ from the app's ({shifts}); its "
                     f"phases keep their formation enthalpies.")
    counts = [f"{len(names)} {word}" for names, word in
              ((found.changed, "changed"), (found.added, "added"), (found.removed, "removed"))
              if names]
    notes.append(("The list has " + ", ".join(counts) + f" and {found.unchanged} unchanged "
                  "phases.") if counts else "The list holds the phases already shown.")
    elements = {hull_entry_name(entry): len(entry.composition.elements)
                for entry in model.solid_hull().all_entries}
    binary = [name for name in found.removed if elements.get(name) == 2]
    if binary:
        notes.append(f"It leaves out {', '.join(binary)}: removing a binary phase can change "
                     f"how well that binary's fit matches its measured liquidus.")
    return edits, notes


def build_ternary_plotter(components, binary_L_dict, fit_or_pred, progress=None, edits=None,
                          units=UNITS[0], notes=None):
    """A processed ``TLIPlotter`` over ``components``, ready for ``get_plot('tx')``.

    ``binary_L_dict`` is keyed by the cyclic edges of the sorted system
    (``gliquid.ternary.ordered_binary_systems``), each ``[L0_a, L0_b, L1_a, L1_b]`` oriented to
    its edge, and is passed through as ``xs_mix``. Every edge with a digitized MPDS liquidus
    draws it with its MAPE chip, predicted edges included. ``progress`` is called as
    ``progress(stage, done, total)`` after the interpolation and then by
    ``TLIPlotter.process_data``. ``edits`` (see ``no_edits``) are applied to the store's phases
    first, and what ``apply_edits`` says of them in ``units`` is added to the list ``notes``.
    """
    ti = ternary_model(components, binary_L_dict, fit_or_pred)
    if edits and any(edits.values()):
        said = apply_edits(ti, edits, units)
        if notes is not None:
            notes += said
    plotter = TLIPlotter(
        ti,
        order="alphabetical",
        edge_mape=True,
        pred_liquidus=True,
    )
    print(f"Rendering {'-'.join(ti.components)}: hull_mode={ti.hull_mode}")
    ti.interpolate()
    if progress is not None:
        progress("interpolate", 1, 1)
    plotter.process_data(progress=progress)
    return plotter


#: CSS class of the sidebar buttons: the look of the binary map's buttons.
BUTTON_CLASS = "gl-btn"
#: Browser-tab title, and the one shown for the whole of a plot generation.
PAGE_TITLE, BUSY_TITLE = "Ternary Plotter", "Updating..."
#: How often a tab polls a running generation, in ms.
POLL_MS = 250


# Fraction of a plot, from the press until it is on screen, done when each stage ends, in stage
# order. The binary edge figures share the span from "model" to "edges" evenly, and the span
# after "done" is the plots' transfer to the browser and their drawing.
PROGRESS_AT = {"start": 0.00, "lookup": 0.00, "interpolate": 0.03, "model": 0.07,
               "edges": 0.14, "hull": 0.31, "ternary": 0.60, "done": 0.60}
# Typical seconds of the step that ends with each stage.
TYPICAL_S = {"lookup": 0.1, "interpolate": 0.2, "model": 0.2, "edges": 0.3,
             "hull": 0.8, "ternary": 1.4, "done": 0.1}
# Binary edge figures drawn between "model" and "edges".
EDGE_COUNT = 3
# The step that runs after each stage, shown under the progress bar.
NEXT_STEP = {"start": "Reading binary parameters",
             "lookup": "Interpolating the ternary liquidus",
             "interpolate": "Preparing the ternary model",
             "model": "Drawing the binary edges",
             "edges": "Evaluating the ternary hull",
             "hull": "Drawing the ternary liquidus surface",
             "ternary": "Styling the binary plots",
             "done": "Sending the plots to the browser"}


def generation_progress(stage, done=1, total=1):
    """``(fraction, label)`` of the progress bar once ``stage`` has finished ``done`` of ``total``.

    ``stage`` is a key of :data:`PROGRESS_AT`; ``"edges"`` counts binary edge figures.
    """
    if stage == "edges" and done < total:
        lo, hi = PROGRESS_AT["model"], PROGRESS_AT["edges"]
        return lo + (hi - lo) * done / total, f"Drawing binary edge {done + 1} of {total}"
    return PROGRESS_AT[stage], NEXT_STEP[stage]


def creep(lo, hi, seconds, typical):
    """The fraction between ``lo`` and ``hi`` shown ``seconds`` into a step that typically takes
    ``typical`` seconds: half the span after ``typical``, and never ``hi``."""
    seconds = max(seconds, 0.0)
    return lo + (hi - lo) * seconds / (seconds + typical)


def stage_progress(stage, done, total, seconds):
    """``(fraction, label)`` of the progress bar ``seconds`` after ``stage`` reported ``done`` of
    ``total``: the bar creeps from that report's fraction toward the next one's."""
    lo, label = generation_progress(stage, done, total)
    if stage == "done":
        return lo, label
    if stage == "model" or (stage == "edges" and done < total):
        count = total if stage == "edges" else EDGE_COUNT
        span = (PROGRESS_AT["edges"] - PROGRESS_AT["model"]) / count
        return creep(lo, lo + span, seconds, TYPICAL_S["edges"] / count), label
    stages = list(PROGRESS_AT)
    after = stages[stages.index(stage) + 1]
    return creep(lo, PROGRESS_AT[after], seconds, TYPICAL_S[after]), label


# The liquidus-surface step (from "hull" to "ternary") is mostly gliquid's
# liquidus_fields.partition_fields, which reports no progress of its own. Each poll reads the
# worker thread's stack instead: SURFACE_STEPS are its sub-steps in call order, each with its
# share of the span and its typical seconds, and PARTITION_STEP maps the function
# partition_fields is calling to one of them. A function gliquid renames or adds keeps the
# sub-step shown last.
SURFACE_STEPS = (("Solving the liquidus sheets", 9, 0.11),
                 ("Placing the edge points", 15, 0.17),
                 ("Grading the phase-field sides", 7, 0.09),
                 ("Refining the phase-field junctions", 36, 0.54),
                 ("Locating the triple points", 3, 0.03),
                 ("Drawing the two-liquid walls", 1, 0.01),
                 ("Finishing the phase fields", 5, 0.06),
                 ("Assembling the figure", 25, 0.26))
PARTITION_STEP = {"_add_edge_points": 1, "_grade_sides": 2, "_refine_junctions": 3,
                  "_triple_points": 4, "_wall_setbacks": 5, "_grade_walls": 5, "_wound": 6,
                  "_cap_added": 6, "_merge_splinters": 6, "_fold_shallow": 6, "_piece": 6,
                  "_outline": 6}
# The gliquid function that assembles the figure once the phase fields are built.
FIGURE_STEP = "render_tx_surface"


def surface_step(thread_id):
    """Index into SURFACE_STEPS of what thread ``thread_id`` is computing, or None if unknown."""
    frame = sys._current_frames().get(thread_id)
    names = []
    while frame is not None:
        names.append(frame.f_code.co_name)
        frame = frame.f_back
    names.reverse()  # outermost call first
    if "partition_fields" in names:
        below = names[names.index("partition_fields") + 1:]
        return PARTITION_STEP.get(below[0]) if below else None
    if "_build_partition" in names:
        return 0
    return len(SURFACE_STEPS) - 1 if FIGURE_STEP in names else None


def surface_progress(state, step, now):
    """``(fraction, label, state)`` of the progress bar during the liquidus-surface step.

    ``state`` is ``(step index or None, when it began, fraction shown last)`` from the previous
    poll, ``step`` the sub-step the worker is in now (None: unknown, keep the last). The
    fraction never goes back.
    """
    index, since, shown = state
    if step is not None and step != index:
        index, since = step, now
    lo, hi, typical = PROGRESS_AT["hull"], PROGRESS_AT["ternary"], TYPICAL_S["ternary"]
    if index is not None:
        weights = [weight for _, weight, _ in SURFACE_STEPS]
        unit = (hi - lo) / sum(weights)
        lo += unit * sum(weights[:index])
        hi = lo + unit * weights[index]
        typical = SURFACE_STEPS[index][2]
    fraction = max(creep(lo, hi, now - since, typical), shown)
    label = SURFACE_STEPS[index][0] if index is not None else NEXT_STEP["hull"]
    return fraction, label, (index, since, fraction)


def format_elapsed(seconds):
    """``seconds`` as m:ss."""
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def gliquid_name():
    """The project name as the binary map writes it: an italic G and a subscript Liquid."""
    return [html.I("G"), html.Sub("Liquid")]


def progress_bar(fraction, label, elapsed=None):
    """The sidebar progress bar filled to ``fraction`` (0-1), with ``label`` under it and,
    while a plot generates, the ``elapsed`` seconds. The bar moves in tenths of a percent; the
    percentage is shown to the nearest whole percent."""
    pct = 100 * min(max(fraction, 0.0), 1.0)
    shown = f"{int(pct + 0.5)}%"
    detail = shown if elapsed is None else f"{shown}, {format_elapsed(elapsed)}"
    return html.Div([
        html.Div(html.Div(className="gl-progress-fill", style={'width': f"{pct:.1f}%"}),
                 className="gl-progress", role="progressbar",
                 **{'aria-valuenow': round(pct), 'aria-valuemin': 0, 'aria-valuemax': 100}),
        html.Div(f"{label} ({detail})", className="gl-progress-label"),
    ])


#: The sidebar's outbound links, as on the binary map. The Paper button points at the group site
#: until the paper has a page of its own.
GITHUB_URL = "https://github.com/willwerj/gliquid_python"
GROUP_URL = "https://whsunresearch.group"
PAPER_URL = GROUP_URL
MAP_URL = "/gliquid/interactive-matrix.html"
#: The binary map's thumbnail, an asset of this package directory.
MAP_THUMB = "binary_map_thumb.webp"
#: Link-button icon colours, as on the binary map; the muted one is the Paper button's.
LINK_COLOR, MUTED_LINK_COLOR = "#18191d", "#808080"

# The GitHub mark and a document outline, 16x16, and the badge's arrow, 24x24.
_GITHUB_MARK = (
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82'
    '-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 '
    '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87'
    '.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09'
    ' 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75'
    '-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8'
    'c0-4.42-3.58-8-8-8z"/>')
_DOCUMENT_ICON = ('<path d="M4 1.5h5.5L13 5v9.5H4z"/><path d="M9.5 1.5V5H13"/>'
                  '<path d="M6 8.5h5M6 11h5"/>')
_ARROW_ICON = '<path d="M6 18 18 6M9 6h9v9"/>'


def svg_icon(body, size, **paint):
    """A 16x16 ``<svg>`` of ``body`` as an ``html.Img`` of ``size`` px. ``paint`` gives the svg
    element's fill and stroke attributes, underscores for dashes; an image cannot use
    ``currentColor``."""
    attrs = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in paint.items())
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" width="{size}" '
           f'height="{size}" {attrs}>{body}</svg>')
    return html.Img(src="data:image/svg+xml," + quote(svg), width=size, height=size, alt="",
                    style={'display': 'block'})


_DOWNLOAD_ICON = '<path d="M8 2.5v8M4.75 7.5 8 10.75 11.25 7.5M3 13.5h10"/>'
_UPLOAD_ICON = '<path d="M8 10.75v-8M4.75 5.75 8 2.5l3.25 3.25M3 13.5h10"/>'
_RESET_ICON = '<path d="M3 8a5 5 0 1 0 1.6-3.7M3 3v2.6h2.6"/>'
_INFO_ICON = '<circle cx="8" cy="8" r="6"/><path d="M8 7.25v3.5M8 5.1v.1"/>'
_PLUS_ICON = '<path d="M8 3.5v9M3.5 8h9"/>'


def line_icon(body, color=LINK_COLOR):
    """A 16x16 outline icon of ``body`` in ``color``, 14 px."""
    return svg_icon(body, 14, fill="none", stroke=color, stroke_width=1.6,
                    stroke_linecap="round", stroke_linejoin="round")


def link_tiles(thumb_url):
    """The sidebar's outbound links, side by side: GitHub, the paper and the binary map, each
    a picture over its name."""
    def tile(picture, label, href):
        return html.A([html.Div(picture, className="gl-tile-img"), html.Span(label)],
                      href=href, target="_blank", rel="noopener", className="gl-tile")

    return html.Div([
        tile(svg_icon(_GITHUB_MARK, 30, fill=LINK_COLOR), "GitHub", GITHUB_URL),
        tile(svg_icon(_DOCUMENT_ICON, 30, fill="none", stroke=MUTED_LINK_COLOR,
                      stroke_width=1.1, stroke_linejoin="round", stroke_linecap="round"),
             "Paper", PAPER_URL),
        tile(html.Img(src=thumb_url, alt="", className="gl-tile-photo"), "Binary map", MAP_URL),
    ], className="gl-tiles")


def liquidus_legend():
    """The key to the binary sub-plots' liquidus lines (assessed, dashed; fitted; predicted): one
    row above the binary column, its ends on the left and right edges of the plot frames."""
    def segment(width, color=None):
        return html.Span(style={'width': f'{width}px', 'height': '3px', 'flex': 'none',
                                'backgroundColor': color or 'transparent'})

    rows = [([segment(8, '#B82E2E'), segment(4), segment(8, '#B82E2E')], "Assessed liquidus"),
            ([segment(20, 'cornflowerblue')], "Fitted liquidus"),
            ([segment(20, '#117733')], "Predicted liquidus")]
    return html.Div([html.Span([html.Span(swatch, className="gl-legend-swatch"), html.Span(label)],
                               className="gl-legend-row")
                     for swatch, label in rows],
                    className="gl-legend-bar",
                    style={'width': f'{SUB_WIDTH}px',
                           'height': f'{LEGEND_BAR_TOP + LEGEND_BAR_HEIGHT}px',
                           'paddingTop': f'{LEGEND_BAR_TOP}px',
                           'paddingLeft': f"{SUB_MARGIN['l']}px",
                           'paddingRight': f"{SUB_MARGIN['r']}px"})


_FORMULA = re.compile(r"(?<![\w.])(?:[A-Z][a-z]?\d*(?:\.\d+)?)+(?!\w)")
_COUNT = re.compile(r"(\d+(?:\.\d+)?)")


def ordered_formula(token):
    """``token`` with its elements in alphabetical order, the order of a system's components
    here, where it is a formula of two or more elements written with a digit or a lower-case
    letter; any other ``token`` unchanged."""
    elements = re.findall(r"[A-Z][a-z]?", token)
    if (len(set(elements)) < 2 or elements == sorted(elements)
            or not any(c.islower() or c.isdigit() for c in token)):
        return token
    try:
        return standard_formula(token, sorted(set(elements)))
    except Exception:
        return token


def formula_text(text):
    """``text`` as Dash children with every chemical formula in it written by ordered_formula
    and its counts subscripted."""
    parts, start = [], 0
    for match in _FORMULA.finditer(text):
        formula = ordered_formula(match.group())
        if formula == match.group() and not _COUNT.search(formula):
            continue
        parts.append(text[start:match.start()])
        parts += [html.Sub(piece) if index % 2 else piece
                  for index, piece in enumerate(_COUNT.split(formula))]
        start = match.end()
    parts.append(text[start:])
    return [part for part in parts if part != ""]


def phase_label(label):
    """A legend label, with ``<sub>`` markup, as Dash children."""
    return [html.Sub(piece) if index % 2 else piece
            for index, piece in enumerate(re.split(r"</?sub>", label)) if piece != ""]


def error_message(text):
    """``text`` as a sidebar message in red."""
    return html.Span(formula_text(text), className="gl-error")


def warning_message(text):
    """``text`` as a sidebar warning, its formulas subscripted."""
    return html.Span(formula_text(text), className="gl-warn")


def info_message(text):
    """``text`` as a sidebar message that is not an error."""
    return html.Span(formula_text(text), className="gl-info")


#: The column heads of the phase list and the placeholders of the add form, by field, and the
#: hover text that spells each out.
FIELD_HEADS = {"solidus": "Liq. min \u00b0C", "liquidus": "Liq. max \u00b0C"}
ENERGY_HEADS = {units: {"enthalpy": "\u0394H " + labels["enthalpy"].replace("atom", "at"),
                        "entropy": "\u0394S " + labels["entropy"].replace("atom", "at")}
                for units, labels in UNIT_LABELS.items()}
FIELD_TITLES = {"solidus": "Min liquidus temp: the lowest temperature of the phase's liquidus "
                           "field, \u00b0C",
                "liquidus": "Max liquidus temp: the highest temperature of the phase's liquidus "
                            "field, \u00b0C",
                "enthalpy": "Formation enthalpy, per atom or per mole of atoms",
                "entropy": "Formation entropy, per atom or per mole of atoms, per K"}
#: The values the add form takes: the entropy is set once the phase is drawn.
ADD_FIELDS = ("solidus", "liquidus", "enthalpy")


def field_head(which, units):
    """The column head of field ``which`` with the energies in ``units``."""
    return {**FIELD_HEADS, **ENERGY_HEADS[units]}[which]


def phase_editor():
    """The phase editor, one card: its title row with the units of the energies, the list of
    the phases on screen with their FIELDS, which render_phase_list fills, and at its foot the
    folded form that adds a phase and the buttons that save, load and reset the system's
    entries."""
    def amount(i):
        return html.Div([html.Span(id=f'comp-label-{i}', className='gl-ph-el'),
                         dcc.Input(id=f'comp-{i}', type='number', min=0, step='any',
                                   className='gl-ph-input')], className='gl-ph-amount')

    return html.Div([
        html.Div([html.Span("Phases", className='gl-section'),
                  dcc.RadioItems(id='energy-units', options=list(UNITS), value=UNITS[0],
                                 inline=True, className='gl-seg', labelClassName='gl-seg-item',
                                 inputClassName='gl-seg-box')],
                 className='gl-ph-title'),
        html.Div([html.Span(field_head(which, UNITS[0]), id=f'head-{which}',
                            title=FIELD_TITLES[which]) for which in FIELDS],
                 className='gl-ph-head'),
        html.Div(id='phase-list'),
        html.Div([
            html.Details([
                html.Summary([line_icon(_PLUS_ICON), "Add"], className='gl-add-summary',
                             title="Add a phase to the system shown"),
                html.Div("Input composition and one or more known properties",
                         className='gl-ph-note'),
                html.Div([amount(0), amount(1), amount(2)], className='gl-ph-amounts'),
                html.Div(id='comp-preview', className='gl-ph-note'),
                html.Div([dcc.Input(id=f'new-{which}', type='number', step='any',
                                    placeholder=field_head(which, UNITS[0]),
                                    className='gl-ph-input')
                          for which in ADD_FIELDS],
                         className='gl-ph-new'),
                dcc.Store(id='new-auto'),
                html.Div([dcc.Input(id='new-name', type='text', placeholder="Name (optional)",
                                    maxLength=30, className='gl-ph-input gl-ph-wide'),
                          html.Button("Add", id='add-phase', n_clicks=0,
                                      className='gl-small-btn')],
                         className='gl-ph-add'),
                html.Div(id='add-message', className='gl-ph-note'),
            ], className='gl-add'),
            html.Div([
                html.Button(line_icon(_DOWNLOAD_ICON), id='download-entries', n_clicks=0,
                            className='gl-tool-btn',
                            title="Download the phases as a JSON list of pymatgen entries"),
                dcc.Upload(html.Button(line_icon(_UPLOAD_ICON), className='gl-tool-btn',
                                       title="Upload a JSON list of pymatgen entries for "
                                             "the system shown"),
                           id='upload-entries', accept='.json,application/json',
                           multiple=False),
                html.Button(line_icon(_RESET_ICON), id='reset-phases', n_clicks=0,
                            disabled=True, className='gl-tool-btn',
                            title="Restore the phases of the store"),
            ], className='gl-foot-tools'),
        ], className='gl-foot'),
        dcc.Download(id='entries-file'),
        html.Div(id='entries-message', className='gl-ph-note'),
    ], className='gl-editor gl-card')


def phase_list_row(row, units, typed=None, locked=(), removed=False, restore=False,
                   drawn=True):
    """One row of the phase list: the phase's name with the button that removes the phase or
    takes that back, and under it its FIELDS with the energies in ``units``, each with a
    padlock once the phase is ``drawn`` and can be edited. A value the phase does not have
    (a temperature of a phase with no liquidus field) shows as "--". An added or changed phase
    is named in bold. ``typed`` holds typed values to keep, by field, ``locked`` the fields whose
    padlock is closed. A phase the plots were drawn without (``restore``) shows its name
    only."""
    name = row["name"]

    def padlocked(which):
        return row["shown"][units][which] is not None and drawn and row["editable"]

    closed = {which for which in locked if padlocked(which)} if not removed else set()

    def changed(which, kept, value):
        blank = kept is None or kept == ""
        typed_here = not blank and (not drawn or value is None or float(kept) != value)
        return typed_here or which in row.get("changed", ())

    def cell(which):
        value = row["shown"][units][which]
        if row["editable"] and not removed:
            kept = (typed or {}).get(which, value)
            field = dcc.Input(id={'type': 'ph-field', 'name': name, 'which': which},
                              type='number', step='any', value=kept,
                              className='gl-ph-input' + (
                                  ' gl-ph-changed' if changed(which, kept, value) else ''),
                              disabled=which in closed or len(closed) >= FREE_FIELDS,
                              placeholder="--" if value is None else None)
        else:
            text = "--" if value is None else f"{value:.4f}".rstrip("0").rstrip(".")
            field = html.Span(text, className='gl-ph-input gl-ph-fixed')
        if value is None or removed or not drawn or not row["editable"]:
            return [field, html.Span()]
        lock = dcc.Checklist(id={'type': 'ph-lock', 'name': name, 'which': which},
                             options=[{'label': '', 'value': 'on',
                                       'title': f"Hold the {FIELD_NAMES[which]} of {name}"}],
                             value=['on'] if which in locked else [],
                             className='gl-ph-lock' + (
                                 ' gl-ph-off' if which not in closed
                                 and len(closed) >= FREE_FIELDS else ''),
                             inputClassName='gl-ph-lock-box')
        return [field, lock]

    if row["editable"] or restore:
        button = html.Button("\u21ba" if removed else "\u00d7",
                             id={'type': 'ph-remove', 'name': name}, n_clicks=0,
                             className='gl-icon-btn' + (' gl-restore' if removed else ''),
                             title="Keep this phase" if removed else "Remove this phase")
    else:
        button = html.Span()
    label = row.get("label", name)
    plain = re.sub(r"</?sub>", "", label)
    tag = [html.Span(row["tag"], className='gl-ph-tag')] if row.get("tag") else []
    dot = html.Span(className='gl-ph-dot', style={'backgroundColor': row.get("color") or 'transparent'})
    head = html.Div([dot, html.Span([*phase_label(label), *tag], title=plain, className='gl-ph-name' + (
                         '' if row["kind"] == "default" else ' gl-ph-changed')), button],
                    className='gl-ph-row-head')
    values = [] if restore else [html.Div([part for which in FIELDS for part in cell(which)],
                                          className='gl-ph-values')]
    return html.Div([head, *values],
                    className='gl-ph-row' + (' gl-ph-removed' if removed else ''))


def ternary_figure(plotter):
    """The app's ternary liquidus figure from a processed ``plotter``.

    A legend click on a solid phase highlights its liquidus field, and the composition labels
    turn with the camera, once the page's ``dash_clientside_function`` callback has attached
    to the graph.
    """
    ternary_plot = plotter.get_plot("tx", interactive_legend=True, solidus=True)
    aspect = ternary_plot.layout.scene.aspectratio
    ternary_plot.update_layout(scene_aspectratio=dict(
        x=aspect.x * TERNARY_XY_STRETCH, y=aspect.y * TERNARY_XY_STRETCH, z=aspect.z))
    ternary_plot.update_layout(
        title=f"<b>Interpolated {plotter.sys_name} Ternary Phase Diagram</b>",
        showlegend=True,
        width=TERNARY_WIDTH,
        height=TERNARY_HEIGHT,
        margin=dict(t=TERNARY_TITLE_MARGIN, b=0),
        font=dict(family=FONT, size=14, color='black'),
        paper_bgcolor='white',
    )
    return ternary_plot


def render_landing_figures(system=None):
    """Render ``system`` from the store: the figures shown before any user input.

    Called once per app, at startup, and returns the ternary figure, the binary figures, the
    phase editor's rows, the system's entries as a list and its processed model: blank figures,
    no rows, no entries and no model if the render fails.
    """
    system = system or LANDING_SYSTEM
    # Outside the try below: an uncovered landing system stops the boot.
    require_landing_system_covered(system)

    text_input = sorted(system.split('-'))

    try:
        binary_L_dict, fitorpred = lookup_binary_parameters(text_input)

        plotter = build_ternary_plotter(text_input, binary_L_dict, fitorpred)

        ternary_plot = ternary_figure(plotter)

        binary_plots = [style_binary_subplot(f) for f in plotter.bin_fig_list]

        print(f"Successfully generated landing system: {plotter.sys_name}")
        return (ternary_plot, binary_plots, phase_rows(plotter, ternary_plot, no_edits()),
                plotter._ti.export_entries(), plotter._ti)

    except Exception as e:
        print(f"Error generating landing system: {e}")
        traceback.print_exc()
        return go.Figure(), [go.Figure(), go.Figure(), go.Figure()], [], [], None


# --- per-tab sessions ---------------------------------------------------------------
# Each browser tab has its own session (see serve_layout). The server keeps at most
# MAX_SESSIONS, drops one SESSION_TTL seconds after its last request and, when full, evicts
# the least recently used; a tab whose session was dropped shows the landing figures.
MAX_SESSIONS = 100
SESSION_TTL = 2 * 3600
# A plot still generating after GENERATION_TIMEOUT seconds is stopped and reported as failed.
# The slowest system measured, Bi-Sn-Th, takes 511 s with hull_mode 'gtx' (667 s with 'hsx').
GENERATION_TIMEOUT = 800
# A generation whose tab has not polled for ABANDON_AFTER seconds (the tab was closed) is
# stopped. Browsers poll a hidden tab as rarely as once a minute, so this stays well above that.
ABANDON_AFTER = 180
# How often, in seconds, a generation's watchdog checks the two limits above.
WATCH_S = 2.0
IDLE_MESSAGE = ""
TIMEOUT_MESSAGE = (f"Plot generation was stopped after {GENERATION_TIMEOUT // 60} min. "
                   "This system takes too long to draw.")
CANCELLED_MESSAGE = "Plot generation cancelled."
# The button's label: it starts a generation, and cancels the one running.
GENERATE_LABEL, CANCEL_LABEL, UPDATE_LABEL = "Generate Plot", "Cancel", "Update Plot"
# The note under the ternary plot on what its legend does.
LEGEND_HINT = "Click a legend phase to highlight its field"
# plot-token value that has deliver_plot send the landing figures; otherwise it is a generation.
LANDING_TOKEN = "landing"


class GenerationCancelled(BaseException):
    """Raised inside a generation's worker thread to stop it. A BaseException, so that no
    ``except Exception`` in the app or in gliquid catches it."""


_default_excepthook = threading.excepthook


def _ignore_cancelled(args):
    """threading.excepthook that drops a GenerationCancelled landing after a worker's handler."""
    if not issubclass(args.exc_type, GenerationCancelled):
        _default_excepthook(args)


threading.excepthook = _ignore_cancelled


class Generation:
    """One plot generation's worker thread, which :meth:`cancel` stops.

    A thread cannot be killed; ``cancel`` raises GenerationCancelled in it instead, which
    CPython delivers before the thread's next Python bytecode, so a long NumPy call finishes
    first. gliquid's locks are all ``with`` blocks, which the exception releases.
    """

    def __init__(self, target, args):
        self._lock = threading.Lock()
        self._running = True
        self.thread = threading.Thread(target=self._run, args=(target, args), daemon=True)

    def start(self):
        self.thread.start()

    def _run(self, target, args):
        try:
            target(*args)
        except GenerationCancelled:
            pass
        finally:
            with self._lock:
                self._running = False

    def cancel(self):
        """Stop the thread if it still runs. At most once: a finished thread's id can be reused."""
        with self._lock:
            if self._running:
                self._running = False
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(self.thread.ident), ctypes.py_object(GenerationCancelled))


class SessionStore:
    """Per-tab session dicts keyed by session id, bounded in number and in idle time."""

    def __init__(self, new_session, max_sessions, ttl, clock=time.monotonic):
        self._new_session = new_session
        self._max_sessions = max_sessions
        self._ttl = ttl
        self._clock = clock
        self._sessions = OrderedDict()  # session id -> (time of last request, session), oldest first
        self._lock = threading.Lock()

    def get(self, session_id, create=False):
        """Return the session for ``session_id`` and mark it as just used.

        An unknown id returns None, or a new session if ``create``.
        """
        with self._lock:
            now = self._clock()
            while self._sessions:
                oldest, (last_used, _) = next(iter(self._sessions.items()))
                if now - last_used <= self._ttl:
                    break
                del self._sessions[oldest]
            _, session = self._sessions.pop(session_id, (None, None))
            if session is None:
                if not create:
                    return None
                session = self._new_session()
            self._sessions[session_id] = (now, session)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
            return session

    def __len__(self):
        return len(self._sessions)


def create_gliqtern_app(requests_pathname_prefix):
    gliq_app = flask.Flask(__name__)
    # update_title=None: the page title is set by the busy-title callback below instead.
    app = dash.Dash(__name__, server=gliq_app, requests_pathname_prefix=requests_pathname_prefix,
                    assets_folder=str(SITE_DIR), update_title=None)
    app.title = PAGE_TITLE

    # The landing figures, rendered once at startup.
    (LANDING_TERNARY, LANDING_BINARIES, LANDING_ROWS, LANDING_ENTRIES,
     LANDING_MODEL) = render_landing_figures(LANDING_SYSTEM)
    LANDING_COMPONENTS = parse_system(LANDING_SYSTEM)
    # The landing system's elements as written, the order of the add-phase boxes.
    LANDING_ORDER = [part.strip() for part in LANDING_SYSTEM.split("-")]
    # The system a tab shows (see the shown-plot Store): None when the landing render failed.
    LANDING_SHOWN = "-".join(LANDING_COMPONENTS) if LANDING_TERNARY.data else None

    def new_session():
        """A session showing the landing figures."""
        return {
            # Shared by every session: never modify a stored figure in place.
            'ternary_plot': LANDING_TERNARY,
            'binary_plots': LANDING_BINARIES,
            'plot_ready': True,
            'error_occurred': False,
            'error_message': "",
            'system': "-".join(LANDING_COMPONENTS),
            # Bumped by each generation started; a generation saves its result only while current.
            'generation': 0,
            'started_at': 0.0,
            # When the tab last polled the generation running.
            'polled_at': 0.0,
            # The last report of the generation running, for the progress bar: the stage, how
            # many of how many, and when.
            'reported': ("start", 1, 1, 0.0),
            # surface_progress state of the liquidus-surface step.
            'surface': (None, 0.0, 0.0),
            # The Generation running, or the last one, and the system it draws.
            'job': None,
            'system_running': "",
            # Shown, not as an error, when deliver_plot next sends the session's plots.
            'notice': "",
            # The phase editor's rows for the plots, and the elements as the user typed them.
            'rows': LANDING_ROWS,
            'order': LANDING_ORDER,
            # The entries of the system as plotted, the list Download entries saves.
            'entries': LANDING_ENTRIES,
            # The processed model the plots were drawn from, which the add form solves against.
            'model': LANDING_MODEL,
            # The phase edits the plots were drawn with (see no_edits).
            'edits': no_edits(),
            # The system's plots and rows without edits, restored by Reset and by a reload.
            'default': {'ternary_plot': LANDING_TERNARY, 'binary_plots': LANDING_BINARIES,
                        'rows': LANDING_ROWS, 'entries': LANDING_ENTRIES,
                        'model': LANDING_MODEL, 'system': "-".join(LANDING_COMPONENTS)},
        }

    sessions = SessionStore(new_session, MAX_SESSIONS, SESSION_TTL)
    # Serializes a generation's end: its result, a cancel, a timeout.
    generation_lock = threading.Lock()

    def cancel_generation(session, message, generation=None, notice=""):
        """Stop the session's running generation (only ``generation``, if given) and show
        ``message`` as its error, or ``notice`` under the plots kept. True if one was stopped."""
        with generation_lock:
            if session['plot_ready'] or generation not in (None, session['generation']):
                return False
            session['generation'] += 1
            session.update(plot_ready=True, error_occurred=bool(message), error_message=message,
                           notice=notice)
            job = session['job']
        if job is not None:
            job.cancel()
        return True

    def watch(session, generation):
        """Stop ``generation`` once it runs past GENERATION_TIMEOUT or its tab stops polling."""
        while session['generation'] == generation and not session['plot_ready']:
            time.sleep(WATCH_S)
            now = time.monotonic()
            if now - session['started_at'] > GENERATION_TIMEOUT:
                if cancel_generation(session, TIMEOUT_MESSAGE, generation):
                    print(f"Stopped {session['system_running']}: timed out after "
                          f"{GENERATION_TIMEOUT} s", flush=True)
            elif now - session['polled_at'] > ABANDON_AFTER:
                if cancel_generation(session, "", generation):
                    print(f"Stopped {session['system_running']}: its tab stopped polling",
                          flush=True)

    # CSS for the sidebar buttons and the progress bar
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                .gl-btn {
                    display: block;
                    width: 100%;
                    box-sizing: border-box;
                    margin-top: 8px;
                    height: 28px;
                    padding: 6px 12px;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 12px;
                    color: #18191d;
                    background-color: #fff;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .gl-btn:hover { background-color: #e6e6e6; border-color: #adadad; }
                .gl-btn:disabled { opacity: 0.65; cursor: not-allowed; }
                .gl-btn:disabled:hover { background-color: #fff; border-color: #ccc; }

                .gl-progress {
                    width: 100%;
                    box-sizing: border-box;
                    height: 12px;
                    margin-top: 8px;
                    background-color: #fff;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    overflow: hidden;
                }
                .gl-progress-fill {
                    height: 100%;
                    background-color: #3276b1;
                    transition: width 0.25s linear;
                }
                .gl-progress-label {
                    margin-top: 4px;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 12px;
                    color: #18191d;
                }

                .gl-error { color: #cc0000; }
                .gl-info { color: #1f4e79; }
                .gl-warn { color: #8a5a00; }
                .gl-btn.gl-idle { opacity: 0.5; cursor: default; pointer-events: none; }
                .gl-editor {
                    margin-top: 12px;
                    padding-top: 8px;
                    border-top: 1px solid #dddddd;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 12px;
                    color: #18191d;
                }
                .gl-ph-title {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    font-size: 14px;
                }
                .gl-ph-note { margin-top: 4px; color: #555555; line-height: 1.3; }
                .gl-ph-note:empty { display: none; }
                .gl-ph-amounts { display: flex; column-gap: 6px; margin-top: 6px; }
                .gl-ph-amount { flex: 1 1 0; display: flex; align-items: center; column-gap: 4px; }
                .gl-ph-el { flex: none; font-weight: 600; }
                .gl-ph-add { display: flex; column-gap: 6px; margin-top: 6px; }
                .gl-ph-input {
                    box-sizing: border-box;
                    width: 100%;
                    min-width: 0;
                    height: 22px;
                    padding: 1px 4px;
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                    background: #ffffff;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 12px;
                    color: #18191d;
                    -moz-appearance: textfield;
                }
                .gl-ph-input::-webkit-outer-spin-button,
                .gl-ph-input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
                .gl-ph-wide { flex: 1 1 0; }
                .gl-ph-fixed {
                    display: flex;
                    align-items: center;
                    background: #eeeeee;
                    color: #777777;
                }
                .gl-ph-files { display: flex; column-gap: 6px; margin-top: 6px; }
                .gl-ph-new {
                    display: grid;
                    grid-template-columns: repeat(4, minmax(0, 1fr));
                    column-gap: 4px;
                    margin-top: 6px;
                }
                .gl-ph-new .gl-ph-input::placeholder { font-size: 10px; }
                .gl-ph-values .gl-ph-input::placeholder { color: #555555; opacity: 1; }
                .gl-ph-row { margin-top: 6px; }
                .gl-ph-row-head {
                    display: grid;
                    grid-template-columns: 10px minmax(0, 1fr) 20px;
                    column-gap: 4px;
                    align-items: center;
                }
                .gl-ph-values, .gl-ph-head {
                    display: grid;
                    grid-template-columns: repeat(4, minmax(0, 1fr) 16px);
                    column-gap: 3px;
                    align-items: center;
                    margin-top: 2px;
                }
                .gl-ph-head { margin-top: 10px; color: #555555; font-size: 10.5px; }
                .gl-ph-head span { display: flex; justify-content: center; white-space: nowrap; }
                .gl-ph-head span:nth-child(1) { grid-column: 1; }
                .gl-ph-head span:nth-child(2) { grid-column: 3; }
                .gl-ph-head span:nth-child(3) { grid-column: 5; }
                .gl-ph-head span:nth-child(4) { grid-column: 7; }
                .gl-ph-lock.gl-ph-off { visibility: hidden; }
                .gl-ph-unit-row {
                    display: flex;
                    align-items: center;
                    column-gap: 8px;
                    margin-top: 10px;
                    color: #555555;
                }
                .gl-ph-units label { margin: 0 8px 0 0; color: #18191d; }
                .gl-ph-unit-box { margin: 0 3px 0 0; vertical-align: -1px; }
                .gl-ph-head { margin-top: 4px; }
                .gl-plot-tools { display: flex; justify-content: center; margin-top: 4px; }
                .gl-ph-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
                .gl-ph-removed .gl-ph-name { text-decoration: line-through; color: #999999; }
                .gl-ph-tag { font-weight: 400; color: #555555; }
                .gl-ph-dot { width: 10px; height: 10px; border-radius: 50%; }
                .gl-ph-removed .gl-ph-dot { opacity: 0.4; }
                .gl-ph-changed, .gl-ph-row:has(.gl-ph-changed) .gl-ph-name { font-weight: 700; }
                .gl-ph-input:disabled { background: #eeeeee; color: #777777; cursor: not-allowed; }
                .gl-ph-lock, .gl-ph-lock label { display: flex; align-items: center; margin: 0; }
                .gl-ph-lock-box {
                    -webkit-appearance: none;
                    appearance: none;
                    width: 16px;
                    height: 18px;
                    margin: 0;
                    border: 0;
                    border-radius: 3px;
                    background: center / 16px 16px no-repeat url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='%23666b73' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='7.25' width='10' height='6.75' rx='1.5'/%3E%3Cpath d='M5.25 7V4.6a2.75 2.75 0 0 1 5.4-.75'/%3E%3C/svg%3E");
                    cursor: pointer;
                }
                .gl-ph-lock-box:checked {
                    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='none' stroke='%2318191d' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='7.25' width='10' height='6.75' rx='1.5' fill='%2318191d'/%3E%3Cpath d='M5.25 7V4.75a2.75 2.75 0 0 1 5.5 0V7'/%3E%3C/svg%3E");
                }
                .gl-ph-lock-box:hover { background-color: #e2e4e8; }
                .gl-ph-lock-box:focus-visible { outline: 2px solid #3276b1; outline-offset: 1px; }
                .gl-small-btn {
                    box-sizing: border-box;
                    height: 22px;
                    padding: 0 6px;
                    border: 1px solid #cccccc;
                    border-radius: 3px;
                    background: #ffffff;
                    color: #18191d;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 12px;
                    cursor: pointer;
                }
                .gl-icon-btn {
                    box-sizing: border-box;
                    width: 20px;
                    height: 22px;
                    padding: 0;
                    border: 0;
                    background: none;
                    color: #c00000;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 22px;
                    font-weight: 700;
                    line-height: 20px;
                    cursor: pointer;
                }
                .gl-icon-btn:hover { color: #7a0000; }
                .gl-icon-btn.gl-restore { color: #1f4e79; font-size: 18px; }
                .gl-icon-btn.gl-restore:hover { color: #0f2940; }
                .gl-icon-btn:focus-visible { outline: 2px solid #3276b1; outline-offset: 1px; }
                .gl-small-btn:hover { background: #e6e6e6; }
                .gl-small-btn:disabled { opacity: 0.5; cursor: default; pointer-events: none; }

                .gl-legend-bar {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    box-sizing: border-box;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 13px;
                    color: #18191d;
                }
                .gl-legend-row {
                    display: flex;
                    align-items: center;
                    column-gap: 5px;
                    white-space: nowrap;
                }
                .gl-legend-swatch { display: flex; flex: none; width: 20px; }

                html, body { margin: 0; background: #ffffff; color-scheme: light; }
                body, button, input { font-family: Arial, Helvetica, sans-serif; }
                .gl-title { margin: 4px 0 8px; font-size: 26px; font-weight: 700;
                            letter-spacing: -0.3px; line-height: 1.15; }
                .gl-title sub { font-size: 15px; }
                .gl-authors { margin: 0; font-size: 13px; line-height: 1.35; }
                .gl-about { margin: 6px 0 0; font-size: 11.5px; line-height: 1.4; color: #4a505a; }
                .gl-section { font-size: 16px; font-weight: 700; color: #18191d; }
                .gl-after-cards { margin-top: 16px; }
                .gl-card { margin-top: 12px; padding: 10px 7px; border: 1px solid #dfe2e7;
                           border-radius: 6px; background: #ffffff; }
                .gl-tiles { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
                            column-gap: 6px; margin-top: 12px; }
                .gl-tile {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    row-gap: 5px;
                    padding: 7px 4px 6px;
                    border: 1px solid #c9ced6;
                    border-radius: 5px;
                    background: #ffffff;
                    color: #18191d;
                    font-size: 12px;
                    font-weight: 700;
                    text-decoration: none;
                }
                .gl-tile:hover, .gl-tile:focus-visible { background: #eef1f5; border-color: #9aa3af; }
                .gl-tile-img { display: flex; align-items: center; justify-content: center;
                               height: 40px; }
                .gl-tile-photo { display: block; height: 40px; width: auto; border-radius: 3px; }
                .gl-system-row { display: flex; column-gap: 6px; margin-top: 8px; }
                .gl-system {
                    flex: 1 1 0;
                    min-width: 0;
                    height: 28px;
                    box-sizing: border-box;
                    padding: 0 8px;
                    border: 1px solid #c9ced6;
                    border-radius: 4px;
                    font-size: 13px;
                }
                .gl-btn {
                    display: block;
                    flex: none;
                    width: 106px;
                    margin-top: 0;
                    padding: 0 8px;
                    font-size: 12px;
                    font-weight: 700;
                    color: #ffffff;
                    background-color: #1f4e79;
                    border-color: #1f4e79;
                }
                .gl-btn:hover { background-color: #173b5c; border-color: #173b5c; }
                .gl-btn:disabled:hover { background-color: #1f4e79; border-color: #1f4e79; }
                .gl-status { margin-top: 6px; font-size: 12px; color: #4a505a; }
                .gl-status:empty { display: none; }
                .gl-editor { font-family: inherit; }
                .gl-foot { position: relative; margin-top: 10px; padding-top: 8px;
                           border-top: 1px solid #dfe2e7; }
                .gl-foot-tools { position: absolute; right: 0; top: 8px; display: flex;
                                 column-gap: 6px; }
                .gl-ph-title { column-gap: 6px; justify-content: flex-start; }
                .gl-ph-title .gl-section { flex: 1 1 0; }
                .gl-seg { display: inline-flex; border: 1px solid #c9ced6; border-radius: 4px;
                          overflow: hidden; background: #ffffff; }
                .gl-seg-item { display: block !important; margin: 0; padding: 5px 7px;
                               font-size: 11px; color: #5b6270; cursor: pointer; }
                .gl-seg-box { position: absolute; opacity: 0; pointer-events: none; }
                .gl-seg-item:has(input:checked) { background: #1f4e79; color: #ffffff;
                                                  font-weight: 700; }
                .gl-tool-btn {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-sizing: border-box;
                    width: 26px;
                    height: 26px;
                    padding: 0;
                    border: 1px solid #c9ced6;
                    border-radius: 4px;
                    background: #ffffff;
                    cursor: pointer;
                }
                .gl-tool-btn:hover { background: #e9ecf0; }
                .gl-tool-btn:disabled { opacity: 0.45; cursor: default; pointer-events: none; }
                .gl-add-summary {
                    display: inline-flex;
                    align-items: center;
                    column-gap: 5px;
                    box-sizing: border-box;
                    height: 26px;
                    padding: 0 10px;
                    border: 1px solid #c9ced6;
                    border-radius: 4px;
                    background: #ffffff;
                    font-size: 12px;
                    cursor: pointer;
                    list-style: none;
                }
                .gl-add-summary::-webkit-details-marker { display: none; }
                .gl-add-summary:hover, .gl-add[open] .gl-add-summary { background: #e9ecf0; }
                .gl-add[open] .gl-add-summary { margin-bottom: 4px; }
                .gl-ph-input, .gl-small-btn, .gl-progress-label, .gl-legend-bar {
                    font-family: inherit; }
                .gl-plot-tools { align-items: center; column-gap: 8px; margin-top: 6px; }
                .gl-pill {
                    display: inline-flex;
                    align-items: center;
                    column-gap: 6px;
                    box-sizing: border-box;
                    height: 28px;
                    padding: 0 12px;
                    border: 1px solid #dfe2e7;
                    border-radius: 14px;
                    background: #f6f7f9;
                    font-size: 12.5px;
                    color: #3c424d;
                    white-space: nowrap;
                }
                button.gl-pill { cursor: pointer; }
                button.gl-pill:hover { background: #e9ecf0; border-color: #c9ced6; }
                button.gl-pill:focus-visible { outline: 2px solid #3276b1; outline-offset: 1px; }
                .gl-ph-head { margin-top: 10px; }
                .gl-ph-values, .gl-ph-head {
                    grid-template-columns: repeat(4, minmax(0, 1fr) 16px);
                    column-gap: 2px;
                }                .gl-ph-values .gl-ph-input { padding: 1px 3px; }
                .gl-ph-new { grid-template-columns: repeat(3, minmax(0, 1fr)); }
                .gl-ph-new .gl-ph-input::placeholder { font-size: 11px; }
                .gl-ph-new .gl-ph-input:disabled { font-weight: 700; }
                [id^="binary-plot"] g.annotation:has(rect.bg[style*="fill-opacity: 0;"]) text {
                    letter-spacing: 0.5px; }
            </style>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    document.addEventListener('keydown', function(event) {
                        if (event.key === 'Enter') {
                            // Find the generate button and click it
                            var button = document.getElementById('submit-val');
                            if (button && !button.disabled) {
                                button.click();
                            }
                        }
                    });
                });
            </script>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''

    def generate_plot(session, generation, components, edits, units, locks, order):
        """Plot sorted ``components`` with the phase ``edits`` into ``session`` unless
        ``generation`` is no longer its current one. A failure, an edit that cannot be applied
        and a moved lock of ``locks`` keep the session's plots. What the edits push off the
        0 K hull or leave without a liquidus field becomes the session's notice, with the
        energies in ``units``. ``order`` is the elements as typed."""
        def report(stage, done=1, total=1):
            if session['generation'] == generation:
                now = time.monotonic()
                session['reported'] = (stage, done, total, now)
                if stage == "hull":
                    session['surface'] = (None, now, PROGRESS_AT["hull"])

        try:
            print(f"Generating plot for: {components} with interpolation type: {INTERP_TYPE}")
            binary_L_dict, fitorpred = lookup_binary_parameters(components)
            report("lookup")

            notes = []
            plotter = build_ternary_plotter(components, binary_L_dict, fitorpred,
                                            progress=report, edits=edits, units=units,
                                            notes=notes)
            ternary_plot = ternary_figure(plotter)
            default = session.get('default') or {}
            rows = phase_rows(plotter, ternary_plot, edits,
                              default['rows'] if default.get('system') == "-".join(components)
                              else ())
            if any(edits.values()) and all(row["liquidus"] is None for row in rows):
                raise EditStopped("Not applied: the phase fields cannot be drawn with these "
                                  "temperatures. Try lower liquidus temps.")
            check_locks(rows, locks)
            shown = session['rows'] if session.get('system') == "-".join(components) else ()
            had = {row["name"] for row in shown if row["liquidus"] is not None}
            lost = [row["name"] for row in rows if row["liquidus"] is None and row["name"] in had]
            if lost:
                notes.insert(0, f"{', '.join(lost)} no longer "
                                f"{'has' if len(lost) == 1 else 'have'} a liquidus field.")
            asked = {**edits["targets"], **{p["name"]: p["target"] for p in edits["added"]}}
            unmet = [row["name"] for row in rows
                     if row["liquidus"] is None and row["name"] not in lost
                     and {"solidus", "liquidus"} & set(asked.get(row["name"], ()))]
            if unmet:
                notes.append(f"{', '.join(unmet)} {'has' if len(unmet) == 1 else 'have'} no "
                             f"liquidus field: the liquidus temps asked for are not met.")
            report("ternary")

            assert len(plotter.bin_fig_list) == 3, \
                f"Expected 3 binary figures, got {len(plotter.bin_fig_list)}"
            binary_plots = [style_binary_subplot(fig) for fig in plotter.bin_fig_list]
            report("done")

            result = {
                'ternary_plot': ternary_plot,
                'binary_plots': binary_plots,
                'system': "-".join(components),
                'error_occurred': False,
                'notice': warning_message(" ".join(notes)) if notes else "",
                'rows': rows,
                'entries': plotter._ti.export_entries(),
                'model': plotter._ti,
                'order': order,
                'edits': edits,
            }
            if not any(edits.values()):
                result['default'] = {key: result[key] for key in
                                     ('ternary_plot', 'binary_plots', 'rows', 'entries', 'model',
                                      'system')}
        except EditStopped as stopped:
            # The message holds characters a console's code page may lack.
            print(ascii(f"Phase edit stopped: {stopped}"))
            result = {'error_occurred': True, 'error_message': str(stopped)}
        except Exception as e:
            print(f"Error occurred during plot generation: {str(e)}")
            traceback.print_exc()
            result = {'error_occurred': True,
                      'error_message': "Invalid or unsupported system, please try again."}

        # A generation that was cancelled, timed out or was superseded leaves the session alone.
        with generation_lock:
            if session['generation'] == generation:
                session.update(result, plot_ready=True)

    # app.layout must stay a function: a static layout gives every visitor the same session id.
    def serve_layout():
        """The page, built on each load so that each load gets a new session id.

        The session-id Store keeps its id in the tab's sessionStorage, which overrides the new
        id on mount: a reload keeps its session, while another tab or browser gets its own.
        """
        return html.Div(
            [
                # The tab's session id
                dcc.Store(id='session-id', storage_type='session', data=str(uuid.uuid4())),

                # Left panel for description and input fields
                html.Div(
                    [
                        html.H1([*gliquid_name(), " Ternary Plotter"], className='gl-title'),
                        html.P(html.B("By Abrar Rauf (arauf@umich.edu), Joshua Willwerth, Shibo Tan, and Wenhao Sun (whsun@umich.edu)"), className='gl-authors'),
                        link_tiles(app.get_asset_url(MAP_THUMB)),
                        html.Div([
                            html.Div("System", className='gl-section'),
                            html.Div([dcc.Input(id='text-input', type='text', value='',
                                                placeholder="e.g. Bi-Cd-Sn",
                                                className='gl-system'),
                                      html.Button(GENERATE_LABEL, id='submit-val', n_clicks=0,
                                                  className=BUTTON_CLASS)],
                                     className='gl-system-row'),
                            html.Div(id='loading-message', children=IDLE_MESSAGE,
                                     className='gl-status'),
                        ], className='gl-card'),
                        phase_editor(),
                        html.Div("About", className='gl-section gl-after-cards'),
                        html.P([
                            "This web app generates an interpolated ternary liquidus for the specified ternary system using fitted or predicted binary interaction parameters from the ",
                            *gliquid_name(),
                            " project (",
                            html.A("Sun Research Group", href=GROUP_URL, target="_blank"),
                            ")"
                        ], className='gl-about'),
                        html.P(["This project is made possible by funding from the U.S. Department of Energy (DOE) Office of Science, Basic Energy Sciences Award No. DE-SC0021130 and the National Science Foundation (NSF) Award No. OAC-2209423"], className='gl-about'),
                        html.P(html.I("Note: The accuracy of the ternary liquidus reconstruction is a work-in-progress and is not guaranteed to work as intended for all ternary systems."), className='gl-about')
                    ],
                    style={
                        'width': f'{SIDEBAR_WIDTH}px', 'boxSizing': 'border-box',
                        'height': '100vh', 'padding': f'{SIDEBAR_PADDING}px',
                        'position': 'sticky', 'top': 0, 'flex': 'none', 'backgroundColor': '#eef0f3',
                        'fontFamily': FONT, 'color': '#18191d',
                        'boxShadow': '2px 0 5px rgba(0,0,0,0.1)', 'overflowY': 'auto',
                        # The text keeps its width whether or not the scrollbar shows.
                        'scrollbarGutter': 'stable',
                    }
                ),

                # Plot area: fixed-size plots at fixed offsets; a narrow window scrolls instead.
                html.Div(
                    [
                        # Left column for the liquidus key and the binary plots
                        html.Div(
                            [liquidus_legend(),
                             *[dcc.Graph(id=f'binary-plot-{i}',
                                         style={'width': f'{SUB_WIDTH}px', 'height': f'{SUB_HEIGHT}px'})
                               for i in (1, 2, 3)]],
                            style={'flex': 'none', 'width': f'{SUB_WIDTH}px'},
                        ),

                        # Right column for the ternary plot and the note on its legend
                        html.Div(
                            [dcc.Graph(id='ternary-plot',
                                       style={'width': f'{TERNARY_WIDTH}px', 'height': f'{TERNARY_HEIGHT}px'}),
                             html.Div([html.Span([line_icon(_INFO_ICON, color="#3c424d"),
                                                  LEGEND_HINT],
                                                 id='legend-hint', className='gl-pill'),
                                       html.Button([line_icon(_DOWNLOAD_ICON, color="#3c424d"),
                                                    "Download HTML"],
                                                   id='download-html', n_clicks=0,
                                                   className='gl-pill',
                                                   title="Save this plot as an interactive "
                                                         "HTML file"),
                                       dcc.Download(id='html-file')],
                                      className='gl-plot-tools')],
                            style={'flex': 'none', 'width': f'{TERNARY_WIDTH}px',
                                   'marginLeft': f'{PLOT_GAP}px', 'paddingTop': f'{TERNARY_TOP}px'},
                        ),
                    ],
                    style={
                        'display': 'flex',
                        'alignItems': 'flex-start',
                        'flex': 'none',
                        'marginLeft': f'{PLOT_GAP}px',
                    }
                ),

                # Polls a running generation for its progress and result
                dcc.Interval(id='interval-component', interval=POLL_MS, n_intervals=0, disabled=True),
                # The generation whose plots deliver_plot sends, or LANDING_TOKEN
                dcc.Store(id='plot-token'),
                # The system on screen, set by deliver_plot
                dcc.Store(id='shown-plot'),
                # The Generate button's label and whether it is disabled, as the server sets
                # them; the button itself is drawn from these by a clientside callback.
                dcc.Store(id='button-label', data=GENERATE_LABEL),
                dcc.Store(id='button-disabled', data=False),
                # The phase editor's rows, the elements as typed and the names of the phases
                # removed from the plots on screen, set by deliver_plot.
                dcc.Store(id='phase-rows'),
                # Phase edits made since the plots on screen were drawn
                dcc.Store(id='pending-edits', data=NO_PENDING),
                # An uploaded list of entries not yet drawn: the system it was read for and
                # the edits (see no_edits) that turn the store's phases into the list's.
                dcc.Store(id='uploaded-edits'),
                # The value the third composition box was last filled in with
                dcc.Store(id='comp-auto'),
                # The ternary figure deliver_plot made, before it takes the view on screen
                dcc.Store(id='ternary-figure'),
                # Outputs of the busy-title and ternary-plot clientside callbacks
                dcc.Store(id='title-sink'),
                dcc.Store(id='ternary-js-sink'),
            ],
            # The sidebar and the plot area side by side, scrolled sideways as one.
            style={'display': 'flex', 'alignItems': 'flex-start', 'width': 'max-content',
                   'minWidth': '100%'},
        )

    app.layout = serve_layout

    def restore_default(session):
        """Show the session's system without phase edits again. Returns the new generation."""
        with generation_lock:
            session['generation'] += 1
            session.update(session['default'], edits=no_edits(), plot_ready=True,
                           error_occurred=False, error_message="", notice="")
            return session['generation']

    def requested_edits(applied, rows, pending, typed, locked, units):
        """``(edits, problem)``: the phase edits an update asks for, from the ``applied`` ones,
        the ``pending`` additions, removals and restorals, the values ``typed`` into the
        ``rows`` on screen (name -> field -> value, the energies in ``units``) and the fields
        ``locked`` (name -> fields).

        A phase with a typed value gets a new target: its typed values and its locked ones,
        FREE_FIELDS in all at most. Where they are fewer, a typed solidus keeps the liquidus, a
        typed liquidus keeps a solidus the phase was given or else a nonzero entropy, and a
        typed enthalpy or entropy keeps the other. A locked solidus the phase was not given is
        not part of the target. An emptied solidus box takes a phase's solidus away again."""
        def blank(value):
            return value in (None, "")

        def retargeted(row):
            """``(target, problem)``: the target the typed values ask for the phase of ``row``,
            None where they ask for no change."""
            name, boxes = row["name"], typed.get(row["name"], {})
            values = {}
            for which in FIELDS:
                value, shown = boxes.get(which), row["shown"][units][which]
                if blank(value) or (shown is not None and float(value) == shown):
                    continue
                values[which], problem = read_value(which, value, units)
                if problem:
                    return None, f"{name}: {problem}"
            cleared = row["solidus_set"] and "solidus" in boxes and blank(boxes["solidus"])
            if not values and not cleared:
                return None, None
            held = [which for which in FIELDS if which in locked.get(name, ())
                    and which not in values and row[which] is not None]
            if len(values) + len(held) > FREE_FIELDS:
                names = " and ".join(FIELD_NAMES[which] for which in held)
                return None, (f"{name}: only {FREE_FIELDS} of its min and max liquidus temp, "
                              f"enthalpy and entropy can be set at once"
                              + (f", and its {names} {'is' if len(held) == 1 else 'are'} "
                                 f"locked" if held else "") + ".")
            target = {**values, **{which: row[which] for which in held
                                   if which != "solidus" or row["solidus_set"]}}
            if cleared:
                target.pop("solidus", None)
            if len(target) < FREE_FIELDS:
                if "solidus" in target:
                    keep = "liquidus"
                elif "liquidus" in target:
                    keep = ("solidus" if row["solidus_set"] and not cleared
                            else "entropy" if abs(row["entropy"]) > 1e-9 and not cleared
                            else None)
                elif "enthalpy" in target:
                    keep = "entropy"
                else:
                    keep = "enthalpy" if "entropy" in target else "liquidus"
                if keep is not None and row[keep] is not None:
                    target[keep] = row[keep]
            if {"solidus", "liquidus"} <= set(target) and not target["solidus"] < target["liquidus"]:
                return None, f"{name}: its min liquidus temp must lie below its max."
            return target, None

        edits = {"added": [{**phase, "target": dict(phase["target"])}
                           for phase in applied["added"]],
                 "targets": {name: dict(target) for name, target in applied["targets"].items()},
                 "removed": list(applied["removed"])}
        dropped = set(pending["removed"])
        edits["removed"] = [n for n in edits["removed"] if n not in pending["restored"]]
        added_names = {phase["name"] for phase in edits["added"]}
        edits["added"] = [p for p in edits["added"] if p["name"] not in dropped]
        for name in pending["removed"]:
            if name not in added_names and name not in edits["removed"]:
                edits["removed"].append(name)
                edits["targets"].pop(name, None)
        taken = {row["name"] for row in rows}
        for phase in pending["added"]:
            if phase.get("name") in dropped:
                continue
            fractions = phase.get("fractions") or {}
            elements = list(fractions)
            target = phase.get("target") or {}
            given = {**{which: shown_value(which, target.get(which), units) for which in target
                        if which in FIELDS},
                     **typed.get(phase.get("name"), {})}
            checked, problem = read_added_phase(
                [fractions[el] for el in elements], elements, phase.get("name"), given, units,
                rows, taken | {p["name"] for p in edits["added"]})
            if problem:
                return None, problem
            edits["added"].append(checked)
        if sum(not p["target"].get("force") for p in edits["added"]) > MAX_ADDED_PHASES:
            return None, f"At most {MAX_ADDED_PHASES} phases can be added to a system."
        for row in rows:
            name = row["name"]
            if not row["editable"] or name in dropped:
                continue
            target, problem = retargeted(row)
            if problem:
                return None, problem
            if target is None:
                continue
            mine = [p for p in edits["added"] if p["name"] == name]
            if mine:
                mine[0]["target"] = target
            else:
                edits["targets"][name] = target
        return edits, None

    # Starts generations, polls them and cancels them. The figures are sent by deliver_plot,
    # once per plot-token value, so a poll never carries a figure.
    @app.callback(
        [Output('loading-message', 'children'),
            Output('interval-component', 'disabled'),
            Output('button-disabled', 'data'),
            Output('button-label', 'data'),
            Output('plot-token', 'data')],
        [Input('submit-val', 'n_clicks'),
            Input('interval-component', 'n_intervals'),
            Input('reset-phases', 'n_clicks')],
        [State('text-input', 'value'),
         State('session-id', 'data'),
         State('shown-plot', 'data'),
         State('pending-edits', 'data'),
         State('uploaded-edits', 'data'),
         State('energy-units', 'value'),
         State({'type': 'ph-field', 'name': ALL, 'which': ALL}, 'id'),
         State({'type': 'ph-field', 'name': ALL, 'which': ALL}, 'value'),
         State({'type': 'ph-lock', 'name': ALL, 'which': ALL}, 'id'),
         State({'type': 'ph-lock', 'name': ALL, 'which': ALL}, 'value')]
    )
    def trigger_and_update_plot(n_clicks, n_intervals, n_resets, text_input, session_id, shown,
                                pending, uploaded, units, field_ids, field_values, lock_ids,
                                lock_values):
        # Identify what triggered the callback; a page load has no trigger
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else ""

        # A tab's first generation starts its session; until then, and once the session is
        # dropped, the tab shows the landing figures.
        session = sessions.get(session_id)

        def unchanged(message):
            """Outputs that leave the plots on screen and the Generate button enabled."""
            return (message, True, False, GENERATE_LABEL, dash.no_update)

        def running():
            """The progress of the generation running; its button cancels it."""
            now = time.monotonic()
            session['polled_at'] = now
            stage, done, total, since = session['reported']
            if stage == "hull":
                fraction, label, session['surface'] = surface_progress(
                    session['surface'], surface_step(session['job'].thread.ident), now)
            else:
                fraction, label = stage_progress(stage, done, total, now - since)
            return (progress_bar(fraction, label, now - session['started_at']), False, False,
                    CANCEL_LABEL, dash.no_update)

        def start(components, edits, locks, order):
            """Generate ``components`` with the phase ``edits`` in a separate thread, watched by
            another."""
            session['generation'] += 1
            now = time.monotonic()
            session.update(plot_ready=False, error_occurred=False, error_message="",
                           started_at=now, polled_at=now, reported=("start", 1, 1, now),
                           system_running="-".join(components), notice="")
            session['job'] = Generation(
                generate_plot,
                (session, session['generation'], components, edits, units, locks, order))
            session['job'].start()
            threading.Thread(target=watch, args=(session, session['generation']),
                             daemon=True).start()
            return running()

        # Reset shows the system on screen without phase edits again.
        if 'reset-phases' in trigger:
            if session is None or not session['plot_ready'] or not any(session['edits'].values()):
                raise PreventUpdate
            return ("", True, True, GENERATE_LABEL, restore_default(session))

        # The button cancels the generation running. Otherwise it plots the typed system, or
        # the one on screen with the phase edits asked for; a system that cannot be plotted,
        # and the one on screen with no new edit, redraw nothing.
        if 'submit-val' in trigger:
            if session is not None and cancel_generation(session, "", notice=CANCELLED_MESSAGE):
                print(f"Cancelled {session['system_running']}", flush=True)
                return unchanged(CANCELLED_MESSAGE)
            order = [part.strip() for part in (text_input or "").split("-") if part.strip()]
            components = sorted(order) or (shown.split("-") if shown else [])
            problem = describe_system_problem(components)
            if problem:
                return unchanged(error_message(problem))
            if shown != "-".join(components):
                session = sessions.get(session_id, create=True)
                return start(components, no_edits(), [], order)
            applied = session['edits'] if session is not None else no_edits()
            rows = session['rows'] if session is not None else LANDING_ROWS
            if uploaded and uploaded.get('system') == shown:
                # An uploaded list is drawn as it is: no typed value and no lock applies.
                edits = uploaded['edits']
                if edits == applied:
                    return unchanged("")
                if not any(edits.values()):
                    return ("", True, True, GENERATE_LABEL, restore_default(session))
                session = sessions.get(session_id, create=True)
                return start(components, edits, [], session['order'])
            typed, locked = {}, {}
            for item, value in zip(field_ids, field_values):
                typed.setdefault(item['name'], {})[item['which']] = value
            for item, value in zip(lock_ids, lock_values):
                if value:
                    locked.setdefault(item['name'], set()).add(item['which'])
            edits, problem = requested_edits(applied, rows, pending or NO_PENDING, typed, locked,
                                             units)
            if problem:
                return unchanged(error_message(problem))
            if edits == applied:
                return unchanged("")
            by_name = {row["name"]: row for row in rows}
            locks = [(name, which, by_name[name][which])
                     for name, fields in locked.items() for which in ("solidus", "liquidus")
                     if which in fields and by_name.get(name, {}).get(which) is not None]
            session = sessions.get(session_id, create=True)
            return start(components, edits, locks, session['order'])

        if session is None:
            return (IDLE_MESSAGE, True, False, GENERATE_LABEL, LANDING_TOKEN)

        if not trigger:
            # A page load (a reload) cancels the generation running and shows the session's
            # last plots, without phase edits, rather than an error already shown.
            if cancel_generation(session, "", notice=CANCELLED_MESSAGE):
                print(f"Cancelled {session['system_running']}: its tab reloaded", flush=True)
            else:
                session['notice'] = ""
            session['error_occurred'] = False
            if any(session['edits'].values()):
                restore_default(session)

        # A page load or poll of a finished generation stops polling and has deliver_plot send
        # its plots; the button stays disabled until they arrive. A page load shows the plots
        # as they are.
        if session['plot_ready']:
            if trigger:
                return (progress_bar(*generation_progress("done")), True, True, GENERATE_LABEL,
                        session['generation'])
            return ("", True, False, GENERATE_LABEL, session['generation'])

        return running()

    @app.callback(
        [Output('ternary-figure', 'data'),
            Output('binary-plot-1', 'figure'),
            Output('binary-plot-2', 'figure'),
            Output('binary-plot-3', 'figure'),
            Output('loading-message', 'children', allow_duplicate=True),
            Output('button-disabled', 'data', allow_duplicate=True),
            Output('shown-plot', 'data'),
            Output('phase-rows', 'data'),
            Output('pending-edits', 'data'),
            Output('uploaded-edits', 'data'),
            Output('entries-message', 'children', allow_duplicate=True)],
        Input('plot-token', 'data'),
        State('session-id', 'data'),
        prevent_initial_call=True,
    )
    def deliver_plot(token, session_id):
        """The plots of generation ``token`` of the tab's session, or the landing plots, with
        the sidebar message, the button re-enabled, the system now on screen and the phase
        editor's rows; an uploaded list and its note are dropped once it is drawn. A failed
        generation sends its error in red and leaves the plots, the pending edits and an
        uploaded list as they are."""
        session = sessions.get(session_id)
        if session is None or token == LANDING_TOKEN:
            return (LANDING_TERNARY, *LANDING_BINARIES, IDLE_MESSAGE, False, LANDING_SHOWN,
                    {'rows': LANDING_ROWS, 'order': LANDING_ORDER, 'removed': []}, NO_PENDING,
                    None, "")
        if token != session['generation'] or not session['plot_ready']:
            # A newer generation started; it sends its own plots.
            raise PreventUpdate
        if session['error_occurred']:
            return (*[dash.no_update] * 4, error_message(session['error_message']), False,
                    *[dash.no_update] * 5)
        return (session['ternary_plot'], *session['binary_plots'], session['notice'], False,
                session['system'],
                {'rows': session['rows'], 'order': session['order'],
                 'removed': session['edits']['removed']}, NO_PENDING, None, "")

    @app.callback(
        [Output('phase-list', 'children'),
            Output('comp-label-0', 'children'),
            Output('comp-label-1', 'children'),
            Output('comp-label-2', 'children'),
            Output('reset-phases', 'disabled')],
        [Input('phase-rows', 'data'),
            Input('pending-edits', 'data'),
            Input('energy-units', 'value')],
        [State({'type': 'ph-field', 'name': ALL, 'which': ALL}, 'id'),
         State({'type': 'ph-field', 'name': ALL, 'which': ALL}, 'value'),
         State({'type': 'ph-lock', 'name': ALL, 'which': ALL}, 'id'),
         State({'type': 'ph-lock', 'name': ALL, 'which': ALL}, 'value')],
    )
    def render_phase_list(shown, pending, units, field_ids, field_values, lock_ids,
                          lock_values):
        """The phase list, with the energies in ``units``: the phases added since the plots
        were drawn, then the phases on screen in legend order, then the removed ones, which can
        be kept again. Typed values survive a pending edit and a change of units, which
        converts them, and are dropped when new plots arrive. A padlock stays as the user left
        it; one not yet on screen is closed on the enthalpy of a DFT phase."""
        if not shown:
            raise PreventUpdate
        pending = pending or NO_PENDING
        triggers = [item['prop_id'] for item in dash.callback_context.triggered]
        fresh = any(prop.startswith('phase-rows') for prop in triggers)
        typed = {}
        if not fresh:
            for item, value in zip(field_ids, field_values):
                typed.setdefault(item['name'], {})[item['which']] = value
        if typed and any(prop.startswith('energy-units') for prop in triggers):
            before = UNITS[1 - UNITS.index(units)]
            was = {row["name"]: row["shown"][before] for row in shown['rows']}
            for name, boxes in typed.items():
                for which in ("enthalpy", "entropy"):
                    value = boxes.get(which)
                    if value in (None, ""):
                        continue
                    if float(value) == was.get(name, {}).get(which):
                        del boxes[which]
                    else:
                        value, problem = read_value(which, value, before)
                        boxes[which] = None if problem else shown_value(which, value, units)
        padlocks = {(item['name'], item['which']): bool(value)
                    for item, value in zip(lock_ids, lock_values)}

        def locked(row):
            return {which for which in FIELDS
                    if padlocks.get((row["name"], which), which == "enthalpy" and row["dft"])}

        def unlisted(name, target=None):
            """The row of a phase the plots on screen do not show."""
            values = {which: (target or {}).get(which) for which in FIELDS}
            return {"name": name, "fractions": [], **values, "shown": shown_values(values),
                    "solidus_set": False, "editable": target is not None, "dft": False,
                    "kind": "default" if target is None else "added"}

        rows = shown['rows']
        new = [unlisted(phase["name"], phase["target"]) for phase in pending["added"]]
        gone = [unlisted(name) for name in shown['removed']]
        children = [phase_list_row(row, units, typed.get(row["name"]), drawn=False)
                    for row in new]
        children += [phase_list_row(row, units, typed.get(row["name"]), locked(row),
                                    removed=row["name"] in pending["removed"]) for row in rows]
        children += [phase_list_row(row, units, removed=row["name"] not in pending["restored"],
                                    restore=True) for row in gone]
        edited = bool(shown['removed']) or any(row["kind"] != "default" for row in rows)
        return (children, *shown['order'], not edited)

    @app.callback(
        [Output('pending-edits', 'data', allow_duplicate=True),
            Output('add-message', 'children'),
            Output('comp-0', 'value'),
            Output('comp-1', 'value'),
            Output('comp-2', 'value'),
            Output('new-name', 'value'),
            *[Output(f'new-{which}', 'value') for which in ADD_FIELDS]],
        Input('add-phase', 'n_clicks'),
        [State('comp-0', 'value'),
         State('comp-1', 'value'),
         State('comp-2', 'value'),
         State('new-name', 'value'),
         *[State(f'new-{which}', 'value') for which in ADD_FIELDS],
         State('new-auto', 'data'),
         State('energy-units', 'value'),
         State('phase-rows', 'data'),
         State('pending-edits', 'data')],
        prevent_initial_call=True,
    )
    def add_phase(n_clicks, a, b, c, name, solidus, liquidus, enthalpy, auto, units, shown,
                  pending):
        """Add the phase the form describes to the pending edits and clear the form, or say
        why it cannot be added. The value fill_added_value filled in is not read."""
        if not n_clicks or not shown:
            raise PreventUpdate
        pending = pending or NO_PENDING
        rows = shown['rows']
        taken = {row["name"] for row in rows} | {phase["name"] for phase in pending["added"]}
        added = sum(row["kind"] == "added" for row in rows) + len(pending["added"])
        if added >= MAX_ADDED_PHASES:
            problem = f"At most {MAX_ADDED_PHASES} phases can be added to a system."
        else:
            given = dict(zip(ADD_FIELDS, (solidus, liquidus, enthalpy)))
            given.pop((auto or {}).get('field'), None)
            phase, problem = read_added_phase([a, b, c], shown['order'], name, given, units,
                                              rows, taken)
        if problem:
            return (dash.no_update, error_message(problem), *[dash.no_update] * 7)
        return ({**pending, "added": [phase, *pending["added"]]}, "", None, None, None, "",
                None, None, None)

    @app.callback(
        [Output('head-enthalpy', 'children'),
            Output('head-entropy', 'children')],
        Input('energy-units', 'value'),
        prevent_initial_call=True,
    )
    def show_units(units):
        """Head the energy columns with ``units``."""
        return tuple(field_head(which, units) for which in ("enthalpy", "entropy"))

    def solved_enthalpy(model, fractions, solidus, liquidus):
        """``(enthalpy in eV/atom, problem)``: the formation enthalpy ``model`` gives a phase at
        element ``fractions`` that joins the hull at ``solidus`` and leaves it at ``liquidus``
        (degrees C), or why there is none."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            record = model.impute_phase(fractions, liquidus + 273.15, solidus=solidus + 273.15,
                                        apply=False, screen=False, write_cache=False)
        if not record.accepted or record.inverted_enthalpy is None:
            return None, refusal(record, {}, UNITS[0])
        return record.inverted_enthalpy / EV_ATOM_TO_J_MOL, None

    @app.callback(
        [*[Output(f'new-{which}', 'value', allow_duplicate=True) for which in ADD_FIELDS],
            *[Output(f'new-{which}', 'disabled') for which in ADD_FIELDS],
            *[Output(f'new-{which}', 'placeholder') for which in ADD_FIELDS],
            Output('new-auto', 'data'),
            Output('add-message', 'children', allow_duplicate=True)],
        [*[Input(f'new-{which}', 'value') for which in ADD_FIELDS],
            Input('comp-0', 'value'),
            Input('comp-1', 'value'),
            Input('comp-2', 'value'),
            Input('energy-units', 'value'),
            Input('phase-rows', 'data')],
        [State('new-auto', 'data'),
         State('session-id', 'data')],
        prevent_initial_call=True,
    )
    def fill_added_value(solidus, liquidus, enthalpy, a, b, c, units, shown, auto, session_id):
        """With two of the add form's ADD_FIELDS typed, grey out the third and fill it: the
        enthalpy from a solidus and a liquidus, a dash for a temperature, which only the drawn
        surface gives. A typed enthalpy follows a change of ``units``."""
        values = dict(zip(ADD_FIELDS, (solidus, liquidus, enthalpy)))
        field, before = (auto or {}).get('field'), (auto or {}).get('units', units)
        if before != units and field != "enthalpy" and values["enthalpy"] not in (None, ""):
            value, problem = read_value("enthalpy", values["enthalpy"], before)
            values["enthalpy"] = None if problem else shown_value("enthalpy", value, units)
        typed = [which for which in ADD_FIELDS
                 if which != field and values[which] not in (None, "")]
        note = ""
        if len(typed) == FREE_FIELDS:
            field = next(which for which in ADD_FIELDS if which not in typed)
            values[field] = None
            session = sessions.get(session_id)
            model = session['model'] if session is not None else LANDING_MODEL
            fractions, problem = read_fractions([a, b, c], (shown or {}).get('order', []))
            if field == "enthalpy" and model is not None and not problem:
                target, problem = read_target({w: values[w] for w in typed}, units)
                if not problem:
                    try:
                        enthalpy, problem = solved_enthalpy(model, fractions,
                                                            target["solidus"],
                                                            target["liquidus"])
                    except ValueError as error:
                        problem = str(error)
                if problem:
                    note = info_message(f"No ΔH fits: {problem}.")
                else:
                    values[field] = shown_value("enthalpy", enthalpy, units)
        else:
            if field is not None:
                values[field] = None
            field = None
        placeholders = [("–" if which == field and values[which] is None
                         else field_head(which, units)) for which in ADD_FIELDS]
        return (*[values[which] for which in ADD_FIELDS],
                *[which == field for which in ADD_FIELDS], *placeholders,
                {'field': field, 'units': units}, note)

    @app.callback(
        Output('html-file', 'data'),
        Input('download-html', 'n_clicks'),
        State('session-id', 'data'),
        prevent_initial_call=True,
    )
    def download_html(n_clicks, session_id):
        """The ternary plot on screen as an HTML file that needs no network."""
        session = sessions.get(session_id)
        figure = session['ternary_plot'] if session is not None else LANDING_TERNARY
        system = session['system'] if session is not None else "-".join(LANDING_COMPONENTS)
        if not n_clicks or not figure.data:
            raise PreventUpdate
        return dict(content=figure.to_html(include_plotlyjs=True, full_html=True),
                    filename=f"{system}_liquidus.html", type="text/html")

    @app.callback(
        Output('entries-file', 'data'),
        Input('download-entries', 'n_clicks'),
        State('session-id', 'data'),
        prevent_initial_call=True,
    )
    def download_entries(n_clicks, session_id):
        """The entries of the system as plotted, phase edits included, as a JSON file."""
        session = sessions.get(session_id)
        entries = session['entries'] if session is not None else LANDING_ENTRIES
        system = session['system'] if session is not None else "-".join(LANDING_COMPONENTS)
        if not n_clicks or not entries:
            raise PreventUpdate
        return dict(content=json.dumps(entries, indent=1), filename=f"{system}_entries.json",
                    type="application/json")

    @app.callback(
        [Output('uploaded-edits', 'data', allow_duplicate=True),
            Output('entries-message', 'children'),
            Output('upload-entries', 'contents')],
        Input('upload-entries', 'contents'),
        State('shown-plot', 'data'),
        prevent_initial_call=True,
    )
    def read_upload(contents, shown):
        """Read an uploaded list of entries for the system on screen: the edits that give the
        store's phases the list's, kept until the next update draws them, and what the user
        should know about them."""
        if not contents:
            raise PreventUpdate
        if not shown:
            return (None, error_message("Generate a system before uploading its entries."), None)
        try:
            entries = json.loads(base64.b64decode(contents.split(",", 1)[1]).decode("utf-8"))
            if not isinstance(entries, list):
                raise ValueError("the file does not hold a list")
            components = shown.split("-")
            model = ternary_model(components, *lookup_binary_parameters(components))
            edits, notes = uploaded_edits(model, entries)
        except Exception as error:
            return (None, error_message(f"The file is not a list of pymatgen entries: {error}"),
                    None)
        return ({'system': shown, 'edits': edits},
                info_message(" ".join(notes) + " Press Update Plot to draw them."), None)

    @app.callback(
        Output('pending-edits', 'data', allow_duplicate=True),
        Input({'type': 'ph-remove', 'name': ALL}, 'n_clicks'),
        [State('phase-rows', 'data'),
         State('pending-edits', 'data')],
        prevent_initial_call=True,
    )
    def toggle_phase(n_clicks, shown, pending):
        """A row's button: drop a phase not yet drawn, mark a drawn one for removal or take
        that back, and keep or drop again a phase the plots were drawn without."""
        clicked = dash.callback_context.triggered_id
        if not clicked or not any(n_clicks) or not dash.callback_context.triggered[0]['value']:
            raise PreventUpdate
        pending = {key: list(value) for key, value in (pending or NO_PENDING).items()}
        name = clicked['name']
        if name in {phase["name"] for phase in pending["added"]}:
            pending["added"] = [p for p in pending["added"] if p["name"] != name]
            return pending
        key = "restored" if name in (shown or {}).get('removed', []) else "removed"
        if name in pending[key]:
            pending[key].remove(name)
        else:
            pending[key].append(name)
        return pending

    @app.callback(
        [Output('pending-edits', 'data', allow_duplicate=True),
            Output('uploaded-edits', 'data', allow_duplicate=True),
            Output('entries-message', 'children', allow_duplicate=True)],
        Input('reset-phases', 'n_clicks'),
        prevent_initial_call=True,
    )
    def clear_pending(n_clicks):
        """Reset also drops the edits not yet drawn and an uploaded list."""
        return NO_PENDING, None, ""

    @app.callback(
        Output('comp-preview', 'children'),
        [Input('comp-0', 'value'),
            Input('comp-1', 'value'),
            Input('comp-2', 'value')],
        State('phase-rows', 'data'),
    )
    def preview_composition(a, b, c, shown):
        """The phase the three composition boxes stand for: its name and element fractions."""
        if not shown or all(value in (None, "") for value in (a, b, c)):
            return ""
        fractions, problem = read_fractions([a, b, c], shown['order'])
        if problem:
            return problem
        parts = ", ".join(f"{el} {f:.3g}" for el, f in fractions.items() if f > 0)
        return formula_text(f"{phase_name(fractions)}: {parts}")

    # Fills the third composition box with what two fractions leave of 1, unless the user has
    # typed their own value there since it was last filled.
    app.clientside_callback(
        """function(a, b, c, last) {
            const none = window.dash_clientside.no_update;
            const fraction = (v) => typeof v === 'number' && v >= 0 && v < 1;
            const untouched = c === null || c === undefined || c === '' || c === last;
            if (!fraction(a) || !fraction(b) || a + b >= 1 || a + b === 0 || !untouched) {
                return [none, none];
            }
            const rest = Math.round((1 - a - b) * 1e6) / 1e6;
            return [rest, rest];
        }""",
        [Output('comp-2', 'value', allow_duplicate=True),
            Output('comp-auto', 'data')],
        [Input('comp-0', 'value'),
            Input('comp-1', 'value')],
        [State('comp-2', 'value'),
         State('comp-auto', 'data')],
        prevent_initial_call=True,
    )

    # A box is bold where its value is not the phase's own, whether drawn or only typed, and
    # disabled while its padlock is closed or FREE_FIELDS of the phase's padlocks are; an open
    # padlock is hidden then.
    app.clientside_callback(
        f"""function(rows, values, locks, units) {{
            const FREE = {FREE_FIELDS};
            const byName = {{}};
            ((rows || {{}}).rows || []).forEach(row => {{ byName[row.name] = row; }});
            const inputs = window.dash_clientside.callback_context.inputs_list;
            const closed = {{}};
            (inputs[2] || []).forEach(item => {{
                if ((item.value || []).length) {{
                    (closed[item.id.name] = closed[item.id.name] || new Set()).add(item.id.which);
                }}
            }});
            const of = name => closed[name] || new Set();
            const boxes = (inputs[1] || []).map(item => {{
                const row = byName[item.id.name], which = item.id.which, value = item.value;
                const blank = value === null || value === undefined || value === '';
                const typed = !blank && (!row || row.shown[units][which] === null
                                         || Number(value) !== row.shown[units][which]);
                const changed = typed || (!!row && (row.changed || []).includes(which));
                return [changed ? 'gl-ph-input gl-ph-changed' : 'gl-ph-input',
                        of(item.id.name).has(which) || of(item.id.name).size >= FREE];
            }});
            const padlocks = (inputs[2] || []).map(item =>
                (!(item.value || []).length && of(item.id.name).size >= FREE)
                    ? 'gl-ph-lock gl-ph-off' : 'gl-ph-lock');
            return [boxes.map(box => box[0]), boxes.map(box => box[1]), padlocks];
        }}""",
        [Output({'type': 'ph-field', 'name': ALL, 'which': ALL}, 'className'),
            Output({'type': 'ph-field', 'name': ALL, 'which': ALL}, 'disabled'),
            Output({'type': 'ph-lock', 'name': ALL, 'which': ALL}, 'className')],
        [Input('phase-rows', 'data'),
            Input({'type': 'ph-field', 'name': ALL, 'which': ALL}, 'value'),
            Input({'type': 'ph-lock', 'name': ALL, 'which': ALL}, 'value')],
        State('energy-units', 'value'),
    )

    # Draws the Generate button: Cancel while a plot generates, disabled while its plots are
    # sent, Update once the system on screen has a phase edit or an uploaded list not yet
    # drawn, and greyed out where a press would redraw nothing.
    app.clientside_callback(
        f"""function(label, sending, typed, shown, pending, rows, values, uploaded, units) {{
            const GENERATE = {json.dumps(GENERATE_LABEL)}, CANCEL = {json.dumps(CANCEL_LABEL)};
            const BUTTON = {json.dumps(BUTTON_CLASS)};
            if (label === CANCEL) return [CANCEL, false, BUTTON];
            if (sending) return [GENERATE, true, BUTTON];
            const parts = (typed || '').split('-').map(s => s.trim()).filter(Boolean).sort();
            if (parts.length && parts.join('-') !== shown) return [GENERATE, false, BUTTON];
            let edited = !!pending && ['added', 'removed', 'restored'].some(
                key => (pending[key] || []).length > 0);
            if (uploaded && uploaded.system === shown) edited = true;
            const byName = {{}};
            ((rows || {{}}).rows || []).forEach(row => {{ byName[row.name] = row; }});
            const inputs = window.dash_clientside.callback_context.inputs_list;
            (inputs[6] || []).forEach(item => {{
                const row = byName[item.id.name], which = item.id.which, value = item.value;
                if (!row) return;
                if (value === null || value === undefined || value === '') {{
                    if (which === 'solidus' && row.solidus_set) edited = true;
                    return;
                }}
                if (row.shown[units][which] === null
                        || Number(value) !== row.shown[units][which]) {{
                    edited = true;
                }}
            }});
            return edited ? [{json.dumps(UPDATE_LABEL)}, false, BUTTON]
                          : [GENERATE, true, BUTTON + ' gl-idle'];
        }}""",
        [Output('submit-val', 'children'),
            Output('submit-val', 'disabled'),
            Output('submit-val', 'className')],
        [Input('button-label', 'data'),
            Input('button-disabled', 'data'),
            Input('text-input', 'value'),
            Input('shown-plot', 'data'),
            Input('pending-edits', 'data'),
            Input('phase-rows', 'data'),
            Input({'type': 'ph-field', 'name': ALL, 'which': ALL}, 'value'),
            Input('uploaded-edits', 'data')],
        State('energy-units', 'value'),
    )

    # The button reads Cancel while a plot generates and is disabled while its plots are sent;
    # the tab title shows both as busy.
    app.clientside_callback(
        f"""function(disabled, label) {{
            const busy = disabled || label === {json.dumps(CANCEL_LABEL)};
            document.title = busy ? {json.dumps(BUSY_TITLE)} : {json.dumps(PAGE_TITLE)};
            return busy;
        }}""",
        Output('title-sink', 'data'),
        Input('button-disabled', 'data'),
        Input('button-label', 'data'),
    )

    # Attaches the ternary plot's legend-click phase highlighting and camera-following labels.
    app.clientside_callback(
        dash_clientside_function(),
        Output('ternary-js-sink', 'data'),
        Input('ternary-plot', 'figure'),
        State('ternary-plot', 'id'),
    )

    # Draws deliver_plot's ternary figure; a redraw of the system on screen keeps its camera and
    # orthographic zoom (the aspect ratio).
    app.clientside_callback(
        """function (figure) {
            const gd = document.querySelector('#ternary-plot .js-plotly-plot');
            const scene = gd && gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
            const title = layout => layout && layout.title && layout.title.text;
            if (figure && scene && title(gd.layout) === title(figure.layout)) {
                figure = JSON.parse(JSON.stringify(figure));
                figure.layout.scene.camera = Object.assign({}, figure.layout.scene.camera,
                                                           scene.getCamera());
                figure.layout.scene.aspectratio = scene.glplot.getAspectratio();
            }
            return figure;
        }""",
        Output('ternary-plot', 'figure'),
        Input('ternary-figure', 'data'),
        prevent_initial_call=True,
    )

    return app


if __name__ == '__main__':
    app = create_gliqtern_app(requests_pathname_prefix="/gliquid/ternary-interpolation/")
    app.run_server(debug=True)

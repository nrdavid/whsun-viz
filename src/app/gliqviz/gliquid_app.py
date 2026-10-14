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
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import threading
import flask
import gliquid.config as gliquid_config
from gliquid.cache import resolve_backend
from gliquid.plotting.binary_tx import CORNER_NOTE_NAME
from gliquid.plotting.ternary_surface import dash_clientside_function
from pymatgen.core import Element
from gliquid.ternary import TernaryLiquidInterpolation, TLIPlotter
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
T_INCR = 10.0
DELTA = 0.025
# Hull evaluation backend of TernaryLiquidInterpolation.
HULL_MODE = 'hsx'

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
TERNARY_WIDTH, TERNARY_HEIGHT = 700, 900

# Page layout, in px. The sidebar's width includes its padding and scrollbar gutter and holds
# IDLE_MESSAGE on one line. PLOT_GAP separates the sidebar, the binary column and the ternary
# plot, which sits TERNARY_TOP below the top of the binary column. Nothing scales with the window.
SIDEBAR_WIDTH, SIDEBAR_PADDING = 330, 10
PLOT_GAP = 20
TERNARY_TOP = 30


def style_binary_subplot(fig):
    """Shrink a package binary phase diagram to a ternary-app sub-plot.

    ``plot_tx`` sets explicit font sizes on its titles and labels, which a figure-level
    ``font`` does not override, so each is resized here.
    """
    fig.update_layout(
        showlegend=False,
        width=SUB_WIDTH,
        height=SUB_HEIGHT,
        font=dict(size=10, color='black'),
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


def build_ternary_plotter(components, binary_L_dict, fit_or_pred, progress=None):
    """A processed ``TLIPlotter`` over ``components``, ready for ``get_plot('tx')``.

    ``binary_L_dict`` is keyed by the cyclic edges of the sorted system
    (``gliquid.ternary.ordered_binary_systems``), each ``[L0_a, L0_b, L1_a, L1_b]`` oriented to
    its edge, and is passed through as ``xs_mix``. Every edge with a digitized MPDS liquidus
    draws it with its MAPE chip, predicted edges included. ``progress`` is called as
    ``progress(stage, done, total)`` after the interpolation and then by
    ``TLIPlotter.process_data``.
    """
    ti = TernaryLiquidInterpolation(
        sorted(components),
        CACHE_STORE,
        order="alphabetical",
        delta=DELTA,
        interp_scheme=INTERP_TYPE,
        param_format=PARAM_FORMAT,
        xs_mix=binary_L_dict or {},
        fit_or_pred=fit_or_pred or {},
        T_incr=T_INCR,
        hull_mode=HULL_MODE,
    )
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


# Fraction of a plot, from the press until it is on screen, done when each stage ends. The
# binary edge figures share the span from "model" to "edges" evenly, and
# the span after "done" is the plots' transfer to the browser.
PROGRESS_AT = {"start": 0.0, "lookup": 0.06, "interpolate": 0.10, "model": 0.12,
               "edges": 0.45, "hull": 0.50, "ternary": 0.78, "done": 0.82}
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


def gliquid_name():
    """The project name as the binary map writes it: an italic G and a subscript Liquid."""
    return [html.I("G"), html.Sub("Liquid")]


def progress_bar(fraction, label):
    """The sidebar progress bar filled to ``fraction`` (0-1), with ``label`` under it."""
    pct = round(100 * min(max(fraction, 0.0), 1.0))
    return html.Div([
        html.Div(html.Div(className="gl-progress-fill", style={'width': f"{pct}%"}),
                 className="gl-progress", role="progressbar",
                 **{'aria-valuenow': pct, 'aria-valuemin': 0, 'aria-valuemax': 100}),
        html.Div(f"{label} ({pct}%)", className="gl-progress-label"),
    ])


#: The sidebar's outbound links, as on the binary map. The Paper button points at the group site
#: until the paper has a page of its own.
GITHUB_URL = "https://github.com/willwerj/gliquid_python"
GROUP_URL = "https://whsunresearch.group"
PAPER_URL = GROUP_URL
MAP_URL = "/gliquid/interactive-matrix.html"
#: The binary map's thumbnail, an asset of this package directory.
MAP_THUMB = "binary_map_thumb.webp"
#: Font size of the sidebar's text, in px. The binary liquidus legend uses it too: the size at
#: which the legend's longest line would fill the sidebar is larger.
TEXT_PX = 14
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


def svg_icon(body, size, view_box="0 0 16 16", **paint):
    """An ``<svg>`` of ``body`` as an ``html.Img`` of ``size`` px. ``paint`` gives the svg
    element's fill and stroke attributes, underscores for dashes; an image cannot use
    ``currentColor``."""
    attrs = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in paint.items())
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}" width="{size}" '
           f'height="{size}" {attrs}>{body}</svg>')
    return html.Img(src="data:image/svg+xml," + quote(svg), width=size, height=size, alt="",
                    style={'display': 'block'})


def link_buttons():
    """The GitHub and Paper buttons side by side, as on the binary map's sidebar."""
    github = html.A([svg_icon(_GITHUB_MARK, 14, fill=LINK_COLOR), html.Span("GitHub")],
                    href=GITHUB_URL, target="_blank", rel="noopener", className="gl-link-btn")
    paper = html.A([svg_icon(_DOCUMENT_ICON, 14, fill="none", stroke=MUTED_LINK_COLOR,
                             stroke_width=1.4, stroke_linejoin="round", stroke_linecap="round"),
                    html.Span("Paper")],
                   href=PAPER_URL, target="_blank", rel="noopener",
                   className="gl-link-btn gl-muted")
    return html.Div([github, paper], className="gl-link-row")


def map_card(thumb_url):
    """The link to the binary map: its thumbnail with an "App" badge, titled over a scrim on
    hover, as the binary map's link to this app."""
    return html.A(
        [html.Img(src=thumb_url, alt="Binary Phase Diagram Map preview", className="gl-map-img"),
         html.Span([svg_icon(_ARROW_ICON, 9, view_box="0 0 24 24", fill="none",
                             stroke="#171a1c", stroke_width=3), "App"],
                   className="gl-map-badge"),
         html.Span(html.Span("Binary Phase Diagram Map \u2197", className="gl-map-title"),
                   className="gl-map-scrim")],
        href=MAP_URL, target="_blank", rel="noopener", className="gl-map-card")


def liquidus_legend():
    """The key to the binary sub-plots' liquidus lines: assessed (dashed), fitted, predicted."""
    def segment(width, color=None):
        return html.Span(style={'width': f'{width}px', 'height': '3px', 'flex': 'none',
                                'backgroundColor': color or 'transparent'})

    rows = [([segment(8, '#B82E2E'), segment(4), segment(8, '#B82E2E')],
             "Assessed binary liquidus"),
            ([segment(20, 'cornflowerblue')], "Fitted binary liquidus"),
            ([segment(20, '#117733')], "Predicted binary liquidus")]
    return html.Div([html.Div([html.Span(swatch, className="gl-legend-swatch"), html.Span(label)],
                              className="gl-legend-row", style={'fontSize': f'{TEXT_PX}px'})
                     for swatch, label in rows])


def error_message(text):
    """``text`` as a sidebar message in red."""
    return html.Span(text, className="gl-error")


def ternary_figure(plotter):
    """The app's ternary liquidus figure from a processed ``plotter``.

    A legend click on a solid phase highlights its liquidus field, and the composition labels
    turn with the camera, once the page's ``dash_clientside_function`` callback has attached
    to the graph.
    """
    ternary_plot = plotter.get_plot("tx", interactive_legend=True)
    ternary_plot.update_layout(
        title=f"<b>Interpolated {plotter.sys_name} Ternary Phase Diagram</b>",
        showlegend=True,
        width=TERNARY_WIDTH,
        height=TERNARY_HEIGHT,
        font=dict(size=14, color='black')
    )
    return ternary_plot


def render_landing_figures(system=None):
    """Render ``system`` from the store: the figures shown before any user input.

    Called once per app, at startup. Returns blank figures if the render fails.
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
        return ternary_plot, binary_plots

    except Exception as e:
        print(f"Error generating landing system: {e}")
        traceback.print_exc()
        return go.Figure(), [go.Figure(), go.Figure(), go.Figure()]


# --- per-tab sessions ---------------------------------------------------------------
# Each browser tab has its own session (see serve_layout). The server keeps at most
# MAX_SESSIONS, drops one SESSION_TTL seconds after its last request and, when full, evicts
# the least recently used; a tab whose session was dropped shows the landing figures.
MAX_SESSIONS = 100
SESSION_TTL = 2 * 3600
# A plot still generating after this many seconds is reported as failed and the buttons are
# re-enabled, so a stuck generation cannot lock its tab out.
GENERATION_TIMEOUT = 300
IDLE_MESSAGE = "Enter input and click 'Generate Plot' to see the result."
TIMEOUT_MESSAGE = "Plot generation timed out, please try again."
# The note under the ternary plot on what its legend does.
LEGEND_HINT = ("Click a phase on the legend to view its solid-liquid coexistence region on the "
               "liquidus surface")
# plot-token value that has deliver_plot send the landing figures; otherwise it is a generation.
LANDING_TOKEN = "landing"


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
    LANDING_TERNARY, LANDING_BINARIES = render_landing_figures(LANDING_SYSTEM)
    LANDING_COMPONENTS = parse_system(LANDING_SYSTEM)
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
            # (fraction, label) of the generation running, for the progress bar.
            'progress': generation_progress("start"),
        }

    sessions = SessionStore(new_session, MAX_SESSIONS, SESSION_TTL)

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

                .gl-plot-hint {
                    width: fit-content;
                    max-width: 100%;
                    box-sizing: border-box;
                    margin: 8px auto 0;
                    padding: 6px 10px;
                    border: 1px solid #dddddd;
                    border-radius: 4px;
                    background: #f8f9fa;
                    color: #444444;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 13px;
                    text-align: center;
                }

                .gl-link-row { display: flex; column-gap: 6px; margin: 14px 0; }
                .gl-link-btn {
                    flex: 1 1 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    column-gap: 6px;
                    box-sizing: border-box;
                    height: 28px;
                    padding: 6px 12px;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    background: #ffffff;
                    color: #18191d;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 12px;
                    text-decoration: none;
                    cursor: pointer;
                }
                .gl-link-btn:hover { background: #f9f9f9; }
                .gl-link-btn.gl-muted { color: #808080; }

                .gl-legend-row {
                    display: flex;
                    align-items: center;
                    column-gap: 5px;
                    margin-bottom: 5px;
                    white-space: nowrap;
                }
                .gl-legend-swatch { display: flex; flex: none; width: 20px; }

                .gl-map-card {
                    position: relative;
                    display: block;
                    box-sizing: border-box;
                    width: 100%;
                    margin: 14px 0;
                    padding-block: 5px;
                    border: 1px solid rgba(229, 229, 229, 0.5);
                    background: #ffffff;
                    overflow: hidden;
                    text-decoration: none;
                    cursor: pointer;
                }
                .gl-map-img { display: block; width: 100%; height: auto; }
                .gl-map-badge {
                    position: absolute;
                    top: 6px;
                    right: 6px;
                    display: flex;
                    align-items: center;
                    gap: 3px;
                    padding: 2px 6px 2px 5px;
                    border-radius: 999px;
                    background: rgba(255, 255, 255, 0.92);
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
                    color: #171a1c;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 10px;
                    font-weight: 600;
                    letter-spacing: 0.03em;
                }
                .gl-map-scrim {
                    position: absolute;
                    inset: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 10px;
                    background: rgba(10, 14, 16, 0.62);
                    text-align: center;
                    opacity: 0;
                    transition: opacity 0.22s ease;
                }
                .gl-map-title {
                    color: #fff;
                    font-family: Helvetica, Arial, sans-serif;
                    font-size: 12.5px;
                    font-weight: 600;
                    line-height: 1.35;
                }
                .gl-map-card:hover .gl-map-scrim,
                .gl-map-card:focus-visible .gl-map-scrim { opacity: 1; }
                .gl-map-card:hover .gl-map-img,
                .gl-map-card:focus-visible .gl-map-img { filter: saturate(0.85) brightness(0.94); }
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

    def generate_plot(session, generation, components):
        """Plot sorted ``components`` into ``session`` unless ``generation`` is no longer its
        current one. A failure keeps the session's plots."""
        def report(stage, done=1, total=1):
            if session['generation'] == generation:
                session['progress'] = generation_progress(stage, done, total)

        try:
            print(f"Generating plot for: {components} with interpolation type: {INTERP_TYPE}")
            binary_L_dict, fitorpred = lookup_binary_parameters(components)
            report("lookup")

            plotter = build_ternary_plotter(components, binary_L_dict, fitorpred, progress=report)
            ternary_plot = ternary_figure(plotter)
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
            }
        except Exception as e:
            print(f"Error occurred during plot generation: {str(e)}")
            traceback.print_exc()
            result = {'error_occurred': True,
                      'error_message': "Invalid or unsupported system, please try again."}

        # A generation that timed out or was superseded leaves the session alone.
        if session['generation'] == generation:
            session.update(result, plot_ready=True)

    # app.layout must stay a function: a static layout gives every visitor the same session id.
    def serve_layout():
        """The page, built on each load so that each load gets a new session id.

        The session-id Store keeps its id in the tab's sessionStorage, which overrides the new
        id on mount: a reload keeps its session, while another tab or browser gets its own.
        """
        text = {'fontSize': f'{TEXT_PX}px'}
        return html.Div(
            [
                # The tab's session id
                dcc.Store(id='session-id', storage_type='session', data=str(uuid.uuid4())),

                # Left panel for description and input fields
                html.Div(
                    [
                        html.H2([*gliquid_name(), " Ternary Plotter"]),
                        html.P(html.B("By Abrar Rauf (arauf@umich.edu), Joshua Willwerth, Shibo Tan, and Wenhao Sun (whsun@umich.edu)"), style=text),
                        html.P([
                            "This web app generates an interpolated ternary liquidus for the specified ternary system using fitted or predicted binary interaction parameters from the ",
                            *gliquid_name(),
                            " project (",
                            html.A("Sun Research Group", href=GROUP_URL, target="_blank"),
                            ")"
                        ], style=text),
                        link_buttons(),
                        html.P(html.B("Usage Instructions:"), style=text),
                        html.P("Specify the system to generate the ternary and corresponding binary phase diagrams.", style=text),
                        html.Label("Ternary system:", htmlFor='text-input', style=text),
                        dcc.Input(id='text-input', type='text', value='', placeholder="e.g., Bi-Cd-Sn",
                                  style={**text, 'display': 'block', 'width': '100%',
                                         'boxSizing': 'border-box', 'marginTop': '4px'}),
                        html.Button('Generate Plot', id='submit-val', n_clicks=0,
                                    className=BUTTON_CLASS, style={'marginTop': '12px'}),
                        html.Div(id='loading-message', children=IDLE_MESSAGE,
                                 style={'fontSize': '13px', 'marginTop': '8px'}),
                        html.Div(liquidus_legend(), style={'marginTop': '14px'}),
                        map_card(app.get_asset_url(MAP_THUMB)),
                        html.P(["This project is made possible by funding from the U.S. Department of Energy (DOE) Office of Science, Basic Energy Sciences Award No. DE-SC0021130 and the National Science Foundation (NSF) Award No. OAC-2209423"], style=text),
                        html.P(html.I("Note: The accuracy of the ternary liquidus reconstruction is a work-in-progress and is not guaranteed to work as intended for all ternary systems."), style={'fontSize': '12px'})
                    ],
                    style={
                        'width': f'{SIDEBAR_WIDTH}px', 'boxSizing': 'border-box',
                        'height': '100vh', 'padding': f'{SIDEBAR_PADDING}px',
                        'position': 'fixed', 'left': 0, 'top': 0, 'backgroundColor': '#f8f9fa',
                        'boxShadow': '2px 0 5px rgba(0,0,0,0.1)', 'overflowY': 'auto',
                        # The text keeps its width whether or not the scrollbar shows.
                        'scrollbarGutter': 'stable',
                    }
                ),

                # Plot area: fixed-size plots at fixed offsets; a narrow window scrolls instead.
                html.Div(
                    [
                        # Left column for binary plots
                        html.Div(
                            [dcc.Graph(id=f'binary-plot-{i}',
                                       style={'width': f'{SUB_WIDTH}px', 'height': f'{SUB_HEIGHT}px'})
                             for i in (1, 2, 3)],
                            style={'flex': 'none', 'width': f'{SUB_WIDTH}px'},
                        ),

                        # Right column for the ternary plot and the note on its legend
                        html.Div(
                            [dcc.Graph(id='ternary-plot',
                                       style={'width': f'{TERNARY_WIDTH}px', 'height': f'{TERNARY_HEIGHT}px'}),
                             html.Div(LEGEND_HINT, id='legend-hint', className='gl-plot-hint')],
                            style={'flex': 'none', 'width': f'{TERNARY_WIDTH}px',
                                   'marginLeft': f'{PLOT_GAP}px', 'paddingTop': f'{TERNARY_TOP}px'},
                        ),
                    ],
                    style={
                        'display': 'flex',
                        'alignItems': 'flex-start',
                        'width': 'max-content',
                        'marginLeft': f'{SIDEBAR_WIDTH + PLOT_GAP}px',
                    }
                ),

                # Polls a running generation for its progress and result
                dcc.Interval(id='interval-component', interval=POLL_MS, n_intervals=0, disabled=True),
                # The generation whose plots deliver_plot sends, or LANDING_TOKEN
                dcc.Store(id='plot-token'),
                # The system on screen, set by deliver_plot
                dcc.Store(id='shown-plot'),
                # Outputs of the busy-title and ternary-plot clientside callbacks
                dcc.Store(id='title-sink'),
                dcc.Store(id='ternary-js-sink'),
            ]
        )

    app.layout = serve_layout

    # Starts generations and polls them. The figures are sent by deliver_plot, once per
    # plot-token value, so a poll never carries a figure.
    @app.callback(
        [Output('loading-message', 'children'),
            Output('interval-component', 'disabled'),
            Output('submit-val', 'disabled'),
            Output('plot-token', 'data')],
        [Input('submit-val', 'n_clicks'),
            Input('interval-component', 'n_intervals')],
        [State('text-input', 'value'),
         State('session-id', 'data'),
         State('shown-plot', 'data')]
    )
    def trigger_and_update_plot(n_clicks, n_intervals, text_input, session_id, shown):
        # Identify what triggered the callback; a page load has no trigger
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else ""

        # A tab's first generation starts its session; until then, and once the session is
        # dropped, the tab shows the landing figures.
        session = sessions.get(session_id)

        def unchanged(message):
            """Outputs that leave the plots on screen and the button enabled."""
            return (message, True, False, dash.no_update)

        def start(components):
            """Generate ``components`` in a separate thread; the button is disabled meanwhile."""
            session['generation'] += 1
            session.update(plot_ready=False, error_occurred=False, error_message="",
                           started_at=time.monotonic(), progress=generation_progress("start"))
            threading.Thread(target=generate_plot, daemon=True,
                             args=(session, session['generation'], components)).start()
            return (progress_bar(*session['progress']), False, True, dash.no_update)

        # The Generate button plots the typed system. A system that cannot be plotted, or the
        # one on screen, redraws nothing.
        if 'submit-val' in trigger:
            components = parse_system(text_input)
            problem = describe_system_problem(components)
            if problem:
                return unchanged(error_message(problem))
            if shown == "-".join(components):
                return unchanged("")
            session = sessions.get(session_id, create=True)
            return start(components)

        if session is None:
            return (IDLE_MESSAGE, True, False, LANDING_TOKEN)

        if (not session['plot_ready']
                and time.monotonic() - session['started_at'] > GENERATION_TIMEOUT):
            session['generation'] += 1
            session.update(plot_ready=True, error_occurred=True, error_message=TIMEOUT_MESSAGE)
        elif not trigger:
            # A page load shows the session's last plots rather than an error already shown.
            session['error_occurred'] = False

        # A page load or poll of a finished generation stops polling and has deliver_plot send
        # its plots; the button stays disabled until they arrive. A page load shows the plots
        # as they are.
        if session['plot_ready']:
            if trigger:
                return (progress_bar(*generation_progress("done")), True, True,
                        session['generation'])
            return ("", True, False, session['generation'])

        return (progress_bar(*session['progress']), False, True, dash.no_update)

    @app.callback(
        [Output('ternary-plot', 'figure'),
            Output('binary-plot-1', 'figure'),
            Output('binary-plot-2', 'figure'),
            Output('binary-plot-3', 'figure'),
            Output('loading-message', 'children', allow_duplicate=True),
            Output('submit-val', 'disabled', allow_duplicate=True),
            Output('shown-plot', 'data')],
        Input('plot-token', 'data'),
        State('session-id', 'data'),
        prevent_initial_call=True,
    )
    def deliver_plot(token, session_id):
        """The plots of generation ``token`` of the tab's session, or the landing plots, with
        the sidebar message, the button re-enabled and the system now on screen. A failed
        generation sends its error in red and leaves the plots on screen."""
        session = sessions.get(session_id)
        if session is None or token == LANDING_TOKEN:
            return (LANDING_TERNARY, *LANDING_BINARIES, IDLE_MESSAGE, False, LANDING_SHOWN)
        if token != session['generation'] or not session['plot_ready']:
            # A newer generation started; it sends its own plots.
            raise PreventUpdate
        if session['error_occurred']:
            return (*[dash.no_update] * 4, error_message(session['error_message']), False,
                    dash.no_update)
        return (session['ternary_plot'], *session['binary_plots'], "", False, session['system'])

    # The Generate button is disabled exactly while a plot generates; the tab title follows it.
    app.clientside_callback(
        f"""function(busy) {{
            document.title = busy ? {json.dumps(BUSY_TITLE)} : {json.dumps(PAGE_TITLE)};
            return busy;
        }}""",
        Output('title-sink', 'data'),
        Input('submit-val', 'disabled'),
    )

    # Attaches the ternary plot's legend-click phase highlighting and camera-following labels.
    app.clientside_callback(
        dash_clientside_function(),
        Output('ternary-js-sink', 'data'),
        Input('ternary-plot', 'figure'),
        State('ternary-plot', 'id'),
    )

    return app


if __name__ == '__main__':
    app = create_gliqtern_app(requests_pathname_prefix="/gliquid/ternary-interpolation/")
    app.run_server(debug=True)

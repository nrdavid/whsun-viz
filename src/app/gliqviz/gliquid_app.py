'''
Authors: Abrar Rauf & Joshua Willwerth

This script generates a Dash web app for the G-Liquid Ternary Interpolation project.
The app allows users to input a ternary system to generate the interpolated
ternary liquidus and corresponding binary phase diagrams.

Neither this package nor the data directory may be named ``gliquid``: /app is the
container WORKDIR and therefore first on ``sys.path``, so that name would shadow the
pip-installed package.
'''
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import threading
import flask
import gliquid.config as gliquid_config
from gliquid.cache import resolve_backend
from pymatgen.core import Element
from gliquid.ternary import TernaryLiquidInterpolation, TLIPlotter
import pandas as pd
import json
import uuid
import traceback
from pathlib import Path
import copy

# --- app paths -----------------------------------------------------------------------
# Data and code share this package directory, which is what main.py mounts at /gliquid/.
SITE_DIR = Path(__file__).resolve().parent
if not (SITE_DIR / "params").is_dir():
    raise RuntimeError(
        f"The /gliquid/ data is missing from {SITE_DIR}: no params/ directory. "
        f"main.py mounts this package as StaticFiles(directory='gliqviz')."
    )

# The whole DFT + digitized-MPDS corpus as one read-only SQLite file, opened
# `mode=ro&immutable=1`; nothing in this container writes to it.
CACHE_STORE = SITE_DIR / "gliquid_cache.sqlite"
# Fitted / ML-predicted Redlich-Kister parameters -- app inputs, not cache records.
PARAMS_DIR = SITE_DIR / "params"

# --- gliquid configuration -----------------------------------------------------------
gliquid_config.set_cache_dir(CACHE_STORE)  # a .sqlite path also sets cache_mode='sqlite'
gliquid_config.set_offline(True)

# --- app-wide model settings ---------------------------------------------------------
# The landing figures shown before the user has typed anything. Rendered LIVE from the
# store, once, at app startup (see create_gliqtern_app) 
LANDING_SYSTEM = "Ce-Fe-Si"

INTERP_TYPE = 'linear'
PARAM_FORMAT = 'combined'
T_INCR = 10.0
DELTA = 0.025

# The DFT functional the ternary path asks the store for, fixed by
# gliquid.ternary.TernaryLiquidInterpolation.get_ternary_form_en. Named here so the
# startup coverage check asks the store the same question the render will.
TERNARY_DFT_TYPE = "GGA"


class UnsupportedSystemError(ValueError):
    """The typed system cannot be plotted, with a reason worth showing the user."""


def describe_system_problem(components):
    """Return a user-facing reason ``components`` cannot be plotted, or None if it can.

    Checked here rather than left to fail deeper: an uncovered ternary otherwise surfaces
    as an OfflineError from get_dft_convexhull, several frames down, and reads identically
    to a typo.
    """
    if len(components) != 3:
        return (f"Enter three elements separated by dashes, e.g. Bi-Cd-Sn. "
                f"Got {len(components)}: {'-'.join(components)}")
    # Membership, not just shape: 'Xx' is well-formed but is not an element, and would
    # otherwise fall through and be reported as merely uncovered by the dataset.
    bad = [e for e in components if not Element.is_valid_symbol(e)]
    if bad:
        return f"Not valid element symbols: {', '.join(bad)}. Use e.g. Bi-Cd-Sn."
    if len(set(components)) != 3:
        return f"Three DIFFERENT elements are required; got {'-'.join(components)}."

    sys_name = "-".join(sorted(components))
    backend = resolve_backend(CACHE_STORE)
    pool_covers = getattr(backend, "pool_covers", None)
    if pool_covers is None or pool_covers(sys_name, TERNARY_DFT_TYPE):
        return None
    n = len(backend.pool_systems())
    return (f"{sys_name} is not in this server's ternary DFT dataset")


def require_landing_system_covered(system):
    """Raise at boot, by name, if ``system`` is not in this store's ternary entry pool.

    Ternary DFT hulls are served EXCLUSIVELY from the pooled entry store: this cache
    carries binary ``dft_entries`` records and zero ternary ones, so an uncovered
    ternary has nowhere to come from. Offline -- which this container is by
    construction -- that surfaces as an ``OfflineError`` thrown part-way through hull
    evaluation while the app is still starting up, naming the network rather than the
    missing coverage. Checking the precondition here turns that into one sentence that
    names the system, the store, and the fix.

    Deliberately NOT caught by the landing renderer's fallback: a misconfigured
    LANDING_SYSTEM is a deployment error that should stop the boot, not degrade
    silently into three blank figures.
    """
    sys_name = "-".join(sorted(system.split("-")))
    backend = resolve_backend(CACHE_STORE)
    pool_covers = getattr(backend, "pool_covers", None)
    if pool_covers is None:
        # A backend with no entry-pool concept at all (e.g. a directory tree). There is
        # no precondition to check; let the render report whatever it finds.
        return
    if pool_covers(sys_name, TERNARY_DFT_TYPE):
        return

    covered = [row[0] for row in backend.pool_systems()]
    raise RuntimeError(
        f"LANDING_SYSTEM {system!r} (resolved to {sys_name!r}) is NOT covered by the "
        f"ternary entry pool in {CACHE_STORE}.\n"
        f"The landing figures are rendered live at app startup and this container is "
        f"offline by construction, so there is no way to fetch the missing entries: "
        f"the render would raise OfflineError part-way through hull evaluation.\n"
        f"The pool covers {len(covered)} ternary system(s)"
        + (f", e.g. {', '.join(covered[:5])}" if covered else "")
        + ".\n"
        f"Fix by EITHER setting LANDING_SYSTEM to one of those, OR extending the pool "
        f"with dev/scripts/build_ternary_entry_pool.py in the G_liquid workspace and "
        f"rebuilding this store."
    )


# --- Redlich-Kister parameter lookup -------------------------------------------------
# The intersection both parameter files must satisfy. The fitted table carries additional
# provenance columns, which nothing here reads.
_REQUIRED_PARAM_COLS = ('system', 'L0_a', 'L0_b', 'L1_a', 'L1_b')


def lookup_binary_parameters(components):
    """``(binary_L_dict, fit_or_pred)`` for the three binary edges of ``components``.

    THE SINGLE SOURCE OF TRUTH for this lookup, called by BOTH render paths: the landing
    figures rendered at startup and the figures a user gets by typing a system. It was
    previously copied into each of them. That duplication predates the 08b refactor but
    got sharper in it -- the landing figures stopped being loaded from pre-rendered JSON
    and are now produced by this same code at every boot, so if the two copies ever
    drifted, the landing diagram and the diagram for that *same* system typed into the
    box would disagree with nothing in the app comparing them.

    ``components`` must already be sorted. Keys are the cyclic edges of that sorted
    system -- ``A-B``, ``B-C``, ``C-A`` -- which is ``build_ternary_plotter``'s and hence
    the package's ``xs_mix`` convention. The third edge is deliberately NOT alphabetical:
    its parameters are stored under the inverse pair, so ``L1`` is negated to re-orient
    them onto the edge as keyed. ``L0`` is symmetric and is not.

    Fitted parameters WIN over predicted ones -- the fitted table is searched first and a
    hit there ends the search, so a system present in both is never served from the ML
    prediction.
    """
    # The user types this string, so it is the one input here not under the app's
    # control. (Landing systems reach require_landing_system_covered first, which is a
    # strictly stronger check; this one is what catches a typo in the input box.)
    assert all(len(e) <= 2 and e.isalpha() for e in components), \
        f"Invalid element symbols: {components}"

    binary_param_df = pd.read_excel(PARAMS_DIR / "fitted_params.xlsx")
    binary_param_pred_df = pd.read_excel(PARAMS_DIR / "predicted_params.xlsx")

    # A params file that lost a column would otherwise surface as a bare KeyError from
    # inside the loop, naming one row rather than the file.
    for label, df in (("Fitted", binary_param_df), ("Predicted", binary_param_pred_df)):
        missing = set(_REQUIRED_PARAM_COLS) - set(df.columns)
        assert not missing, f"{label} params missing columns: {missing}"

    binary_sys_labels = [
        f"{components[0]}-{components[1]}",
        f"{components[1]}-{components[2]}",
        f"{components[2]}-{components[0]}"
    ]
    print("Binary System Labels: ", binary_sys_labels)

    fitted_systems = binary_param_df['system'].tolist()
    pred_systems = binary_param_pred_df['system'].tolist()

    binary_L_dict = {}
    fit_or_pred = {}

    for bin_sys in binary_sys_labels:
        flipped_sys = "-".join(sorted(bin_sys.split('-')))

        # Prioritize fitted params over predicted - check fitted dataframe first
        if bin_sys in fitted_systems or flipped_sys in fitted_systems:
            key = bin_sys if bin_sys in fitted_systems else flipped_sys
            params = binary_param_df[binary_param_df['system'] == key].iloc[0]
            fit_or_pred[bin_sys] = "fit"
        elif bin_sys in pred_systems or flipped_sys in pred_systems:
            key = bin_sys if bin_sys in pred_systems else flipped_sys
            params = binary_param_pred_df[binary_param_pred_df['system'] == key].iloc[0]
            fit_or_pred[bin_sys] = "pred"
        else:
            raise ValueError(f"Binary system {bin_sys} not found in the parameter dataframe.")

        L0_a, L0_b = float(params["L0_a"]), float(params["L0_b"])
        L1_a, L1_b = float(params["L1_a"]), float(params["L1_b"])

        # L1 is odd under component swap, so negate iff the row MATCHED is stored in the
        # opposite ordering to the edge asked for. Key on the matched row, not the
        # requested name: that is correct under either file convention.
        if key != bin_sys:
            L1_a, L1_b = -L1_a, -L1_b

        binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]

    print(fit_or_pred)
    print("Binary Interaction Parameters: ", binary_L_dict)

    # The only count check here that can fire: a degenerate input like "Fe-Fe-Fe"
    # collapses all three edge labels onto one key, and the missing edges would
    # otherwise reach the plotter as a silently short xs_mix.
    assert len(binary_L_dict) == 3, \
        f"Expected 3 binary systems in L_dict, got {len(binary_L_dict)}: {list(binary_L_dict)}"

    return binary_L_dict, fit_or_pred


SUB_WIDTH, SUB_HEIGHT = 400, 300
SUB_MARGIN = dict(t=10, b=50, l=52, r=12)
# Fonts for the 400x300 panel: on-plot phase/polymorph/compound labels, the liquid region
# label, and the component names under the axis.
SUB_LABEL_FONT, SUB_L_FONT, SUB_COMPONENT_FONT = 9, 11, 11
# Paper-y of the component names, i.e. |y| * plot_height px below the axis (24px here).
SUB_COMPONENT_Y = -0.10


def style_binary_subplot(fig):
    """Shrink a package binary phase diagram to a ternary-app sub-plot.

    ``plot_tx`` pins explicit font sizes for a 750x600 standalone figure (title 18, axis
    titles 16, component names 13.5) which a figure-level ``font`` override does NOT reach,
    so at 400x300 everything but the tick labels renders about twice too large. The vendored
    plotter this replaced set only ``font_size``, so the old override cascaded.
    """
    fig.update_layout(
        showlegend=False,
        width=SUB_WIDTH,
        height=SUB_HEIGHT,
        font=dict(size=10, color='black'),
        title=None,
        # The package sizes margins for a titled 750x600 figure (t=72, b=72, default l=80).
        # At 300px tall that is ~half the panel; the title is gone and the legend is off.
        margin=SUB_MARGIN,
    )
    fig.update_xaxes(title_font=dict(size=10))
    fig.update_yaxes(title_font=dict(size=10))

    # Headroom above the liquidus, as the vendored plotter had (conds[1] + 100).
    rng = fig.layout.yaxis.range
    if rng is not None:
        fig.update_yaxes(range=[rng[0], rng[1] + 100])

    for a in fig.layout.annotations:
        if a.xref == "paper" and a.yref == "paper":
            # The component names. Re-anchor them centred under the 0 and 100 composition
            # ticks (the package offsets them outward to x=-0.05/1.05, which overlaps the
            # tick labels once the panel is this narrow) and drop them clear of the ticks.
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


def build_ternary_plotter(components, binary_L_dict, fit_or_pred, temp_slider):
    """A processed ``TLIPlotter`` over ``components``, ready for ``get_plot('tx')``.

    This is the whole of what the deleted ``ternary_hsx.ternary_gtx_plotter`` adapter
    did: translate this app's vocabulary into the package's. ``binary_L_dict`` is keyed
    by the cyclic edges of the *sorted* system -- exactly
    ``gliquid.ternary.ordered_binary_systems(sorted(components))`` -- with each edge's
    ``[L0_a, L0_b, L1_a, L1_b]`` already oriented to that edge (L1 negated when the
    stored pair is the inverse). That is the package's ``xs_mix`` convention verbatim,
    so the dict passes straight through.

    The construction ORDER is load-bearing and is the adapter's: build the model, wrap
    it in the plotter, then interpolate, then process. ``process_data()`` would
    interpolate lazily on its own, but doing it explicitly means an offline cache miss
    raises at a named step rather than part-way through hull evaluation. Both objects
    are framed 'alphabetical', so ``TLIPlotter`` re-frames onto the same order and holds
    this very interpolation rather than a copy of it.
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
    )
    plotter = TLIPlotter(
        ti,
        order="alphabetical",
        temp_slider=tuple(temp_slider),
        T_incr=T_INCR,
    )
    ti.interpolate()
    plotter.process_data()
    return plotter


def render_landing_figures(system=None):
    """Render ``system`` from the store: the figures shown before any user input.

    Called ONCE per app, at startup, so this live render costs nothing per request.
    """
    system = system or LANDING_SYSTEM
    # Precondition first, and OUTSIDE the try below: an uncovered landing system is a
    # deployment error, and the fallback must not turn it into three blank figures.
    require_landing_system_covered(system)

    text_input = sorted(system.split('-'))
    upper_increment = 0.0
    lower_increment = 0.0

    try:
        binary_L_dict, fitorpred = lookup_binary_parameters(text_input)

        temp_slider = [lower_increment, upper_increment]
        plotter = build_ternary_plotter(text_input, binary_L_dict, fitorpred, temp_slider)

        tern_width, tern_height = 700, 900

        ternary_plot = plotter.get_plot("tx")
        ternary_plot.update_layout(
            title=f"<b>Interpolated {plotter.sys_name} Ternary Phase Diagram</b>",
            showlegend=True,
            width=tern_width,
            height=tern_height,
            font=dict(size=14, color='black')
        )

        binary_plots = [style_binary_subplot(f) for f in plotter.bin_fig_list]

        print(f"Successfully generated landing system: {plotter.sys_name}")
        return ternary_plot, binary_plots

    except Exception as e:
        # Anything OTHER than the coverage precondition above: log loudly and fall back
        # to blank figures 
        print(f"Error generating landing system: {e}")
        traceback.print_exc()
        # Return empty figures as fallback
        return go.Figure(), [go.Figure(), go.Figure(), go.Figure()]

def create_gliqtern_app(requests_pathname_prefix):
    gliq_app = flask.Flask(__name__)
    app = dash.Dash(__name__, server=gliq_app, requests_pathname_prefix=requests_pathname_prefix, 
                    assets_folder=str(SITE_DIR))
    app.title = "Ternary Plotter"

    # Render the landing figures once at app startup, live from the store. Doing it here
    # rather than per request is why the live render costs nothing at serve time -- and
    # is what let the four pre-rendered ternary_cache/ JSONs go away.
    LANDING_TERNARY, LANDING_BINARIES = render_landing_figures(LANDING_SYSTEM)
    
    # Dictionary to store per-session data
    session_data = {}
    
    # Session cleanup - remove sessions older than 2 hours to prevent memory leaks
    import time
    session_timestamps = {}
    
    def cleanup_old_sessions():
        """Remove sessions older than 2 hours"""
        current_time = time.time()
        sessions_to_remove = [
            sid for sid, timestamp in session_timestamps.items() 
            if current_time - timestamp > 7200  # 2 hours
        ]
        for sid in sessions_to_remove:
            if sid in session_data:
                del session_data[sid]
            del session_timestamps[sid]
        if sessions_to_remove:
            print(f"Cleaned up {len(sessions_to_remove)} old sessions")

    # CSS for loading animation
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                .loading-spinner {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border: 2px solid #f3f3f3;
                    border-top: 2px solid #3498db;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
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

    def generate_plot(session_id, text_input, upper_increment, lower_increment):
        """Generate plot for a specific session"""
        if session_id not in session_data:
            session_data[session_id] = {
                'ternary_plot': copy.deepcopy(LANDING_TERNARY),
                'binary_plots': [copy.deepcopy(fig) for fig in LANDING_BINARIES],
                'plot_ready': False,
                'error_occurred': False,
                'error_message': "",
                'button_clicked': False
            }
        
        session = session_data[session_id]
        
        try:
            session['error_occurred'] = False
            session['error_message'] = ""
            
            text_input = text_input.split('-')

            text_input = sorted(text_input)
            print(f"Generating plot for: {text_input} with interpolation type: {INTERP_TYPE}")

            temp_slider = [lower_increment, upper_increment]

            # Tell a malformed system apart from one this offline store does not cover;
            # an uncovered ternary otherwise surfaces as a deep OfflineError.
            problem = describe_system_problem(text_input)
            if problem:
                raise UnsupportedSystemError(problem)

            # Shared with render_landing_figures -- see lookup_binary_parameters for what
            # this used to be inline here, and which of its asserts were kept.
            binary_L_dict, fitorpred = lookup_binary_parameters(text_input)

            plotter = build_ternary_plotter(text_input, binary_L_dict, fitorpred, temp_slider)

            tern_width = 700
            tern_height = 900
            # Generate the plots
            ternary_plot = plotter.get_plot("tx")
            ternary_plot.update_layout(title=f"<b>Interpolated {plotter.sys_name} Ternary Phase Diagram</b>", showlegend=True, width=tern_width, height=tern_height, font=dict(size=14, color='black'))

            # ASSERT: Plotter should have 3 binary figures
            assert hasattr(plotter, 'bin_fig_list'), "Plotter should have bin_fig_list attribute"
            assert len(plotter.bin_fig_list) == 3, f"Expected 3 binary figures, got {len(plotter.bin_fig_list)}"

            binary_plot_1 = style_binary_subplot(plotter.bin_fig_list[0])
            binary_plot_2 = style_binary_subplot(plotter.bin_fig_list[1])
            binary_plot_3 = style_binary_subplot(plotter.bin_fig_list[2])
            
            # ASSERT: All plots should be valid Plotly figures
            assert isinstance(ternary_plot, go.Figure), "Ternary plot should be a Plotly Figure"
            assert isinstance(binary_plot_1, go.Figure), "Binary plot 1 should be a Plotly Figure"
            assert isinstance(binary_plot_2, go.Figure), "Binary plot 2 should be a Plotly Figure"
            assert isinstance(binary_plot_3, go.Figure), "Binary plot 3 should be a Plotly Figure"

            # Store in session
            session['ternary_plot'] = ternary_plot
            session['binary_plots'] = [binary_plot_1, binary_plot_2, binary_plot_3]
            session['plot_ready'] = True
            
        except UnsupportedSystemError as e:
            # Diagnosed above, so show the specific reason rather than the catch-all.
            print(f"Unsupported system: {e}")
            session['error_occurred'] = True
            session['error_message'] = str(e)
            session['plot_ready'] = True
        except Exception as e:
            print(f"Error occurred during plot generation: {str(e)}")
            traceback.print_exc()
            session['error_occurred'] = True
            session['error_message'] = "Invalid or unsupported system, please try again."
            session['plot_ready'] = True


    app.layout = html.Div(
        [ 
            # Hidden div to store session ID
            dcc.Store(id='session-id', storage_type='session', data=str(uuid.uuid4())),
            
            # Left panel for description and input fields
            html.Div(
                [
                    html.H2("GLiquid Ternary Plotter", style={'fontsize': '14px'}),
                    html.P([
                        "This web app generates an interpolated ternary liquidus for the specified ternary system using fitted or predicted binary interaction parameters from the GLiquid project (",
                        html.A("Sun Research Group", href="https://whsunresearch.group", target="_blank"),
                        ")"
                    ], style={'fontSize': '14px'}),
                    html.P(["This project is made possible by funding from the U.S. Department of Energy (DOE) Office of Science, Basic Energy Sciences Award No.      DE-SC0021130 and the National Science Foundation (NSF) Award No. OAC-2209423"], style={'fontSize': '14px'}),
                    html.A("Binary Phase Diagram Map", href="/gliquid/interactive-matrix.html", target="_blank", style={'fontSize': '14px'}),
                    html.Br(),
                    html.Br(),
                    html.P(html.B("Usage Instructions:"), style={'fontSize': '14px'}),
                    html.P("Specify the system to generate the ternary and corresponding binary phase diagrams.", style={'fontSize': '14px'}),
                    html.Label("Ternary system: ", style={'fontSize': '14px'}),
                    dcc.Input(id='text-input', type='text', value='', placeholder="e.g., Bi-Cd-Sn", style={'fontSize': '14px'}),
                    html.Br(),
                    html.Br(),
                    html.Div([
                        html.Div([
                            html.Div(style={'width': '8px', 'height': '3px', 'backgroundColor': '#B82E2E', 'display': 'inline-block'}),
                            html.Div(style={'width': '4px', 'height': '3px', 'display': 'inline-block'}),  # gap
                            html.Div(style={'width': '8px', 'height': '3px', 'backgroundColor': '#B82E2E', 'display': 'inline-block'})
                        ], style={'display': 'inline-block', 'marginRight': '5px'}),
                        html.Span("Assessed binary liquidus", style={'fontSize': '13px'})
                    ], style={'marginBottom': '5px'}),
                    html.Div([
                        html.Div(style={'width': '20px', 'height': '3px', 'backgroundColor': 'cornflowerblue', 'display': 'inline-block', 'marginRight': '5px'}),
                        html.Span("Fitted binary liquidus", style={'fontSize': '13px'})
                    ], style={'marginBottom': '5px'}),
                    html.Div([
                        html.Div(style={'width': '20px', 'height': '3px', 'backgroundColor': '#117733', 'display': 'inline-block', 'marginRight': '5px'}),
                        html.Span("Predicted binary liquidus", style={'fontSize': '13px'})
                    ], style={'marginBottom': '5px'}),
                    html.Br(),
                    html.P("The default temperature range may not capture the entire liquidus. To extend this, adjust the 'Temperature Axis Slider' and regenerate the plot", style={'fontSize': '14px'}),
                    html.P(html.B("Temperature Axis Slider:"), style={'fontSize': '14px'}),
                    html.Label("Increment upper bound by:", style={'fontSize': '14px'}),
                    dcc.Input(id='upper_increment', type='number', value=0.0, style={'fontSize': '14px'}),
                    html.Br(),
                    html.Label("Decrement lower bound by:", style={'fontSize': '14px'}),
                    dcc.Input(id='lower_increment', type='number', value=0.0, style={'fontSize': '14px'}),
                    html.Br(),
                    html.Br(),
                    html.Button('Generate Plot', id='submit-val', n_clicks=0),
                    html.Br(),
                    html.Div(id='loading-message', children="Enter input and click 'Generate Plot' to see the result.", style={'fontSize': '13px'}),
                    html.Br(),
                    html.P(html.B("By Abrar Rauf (arauf@umich.edu), Joshua Willwerth, Shibo Tan, and Wenhao Sun (whsun@umich.edu)"), style={'fontSize': '14px'}),
                    html.P(html.I("Note: The accuracy of the ternary liquidus reconstruction is a work-in-progress and is not guaranteed to work as intended for all ternary systems."), style={'fontSize': '12px'})
                ],
                style={
                    'width': '15%', 'height': '100vh', 'padding': '10px',
                    'position': 'fixed', 'left': 0, 'top': 0, 'backgroundColor': '#f8f9fa',
                    'boxShadow': '2px 0 5px rgba(0,0,0,0.1)', 'overflowY': 'auto',
                    'display': 'inline-block',  
                    'verticalAlign': 'top'
                }
            ),

            # Right side main plot area
            html.Div(
                [
                    html.Div(
                        [
                            # Left column for binary plots
                            html.Div(
                                [
                                    dcc.Graph(id='binary-plot-1', style={'height': '100%', 'width': '100%'}),
                                    dcc.Graph(id='binary-plot-2', style={'height': '100%', 'width': '100%'}),
                                    dcc.Graph(id='binary-plot-3', style={'height': '100%', 'width': '100%'})
                                ],
                                style={
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'width': '30%', 
                                    'margin-right': '2%'  
                                }
                            ),

                            # Right column for the ternary plot
                            html.Div(
                                dcc.Graph(id='ternary-plot', style={'height': '100%', 'width': '100%'}),
                                style={
                                    'width': '65%',  # Increased from 65%
                                    'margin-left': '20px', # Changed from 'auto'
                                    'paddingTop': '30px'
                                }
                            )
                        ],
                        style={
                            'display': 'flex',
                            'flexDirection': 'row',
                            'margin-left': '20%',  
                            'boxSizing': 'border-box',
                            'height': '100vh'
                        }
                    )
                ]
            ),

            # Interval component to check if the plot is ready every 1 second
            dcc.Interval(id='interval-component', interval=1000, n_intervals=0, disabled=True)
        ]
    )

    # Combined callback for triggering the plot and updating the graph
    @app.callback(
        [Output('ternary-plot', 'figure'),
            Output('binary-plot-1', 'figure'),
            Output('binary-plot-2', 'figure'),
            Output('binary-plot-3', 'figure'),
            Output('loading-message', 'children'),
            Output('interval-component', 'disabled'),
            Output('submit-val', 'disabled')],
        [Input('submit-val', 'n_clicks'),
            Input('interval-component', 'n_intervals')],
        [State('text-input', 'value'), 
         State('upper_increment', 'value'), 
         State('lower_increment', 'value'),
         State('session-id', 'data')]
    )
    def trigger_and_update_plot(n_clicks, n_intervals, text_input, upper_increment, lower_increment, session_id):
        # Periodic session cleanup
        if n_intervals % 60 == 0:  # Every 60 seconds
            cleanup_old_sessions()
        
        # Initialize session if it doesn't exist (new user)
        if session_id not in session_data:
            session_timestamps[session_id] = time.time()
            session_data[session_id] = {
                'ternary_plot': copy.deepcopy(LANDING_TERNARY),
                'binary_plots': [copy.deepcopy(fig) for fig in LANDING_BINARIES],
                'plot_ready': True,  # Landing page is ready immediately
                'error_occurred': False,
                'error_message': "",
                'button_clicked': False
            }
        else:
            # Update timestamp for active session
            session_timestamps[session_id] = time.time()
        
        session = session_data[session_id]

        # Identify what triggered the callback
        ctx = dash.callback_context

        # If the button is clicked, start generating the plot in a separate thread
        if ctx.triggered and 'submit-val' in ctx.triggered[0]['prop_id']:
            session['button_clicked'] = True
            session['plot_ready'] = False
            session['error_occurred'] = False
            session['error_message'] = ""
            thread = threading.Thread(target=generate_plot, args=(session_id, text_input, upper_increment, lower_increment))
            thread.start()
            
            # Create animated loading message
            loading_message = html.Div([
                html.Span("Takes up to 2 minutes to generate plot"),
                html.Div(className="loading-spinner", style={'marginLeft': '8px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'fontSize': '13px'})
            
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, loading_message, False, True

        # If the interval triggered the callback, check if the plot is ready
        if session['plot_ready']:
            if session['error_occurred']:
                # Return empty plots and show error message
                empty_fig = go.Figure()
                return empty_fig, empty_fig, empty_fig, empty_fig, session['error_message'], True, False
            else:
                # Return the plots from session data
                binary_plots = session['binary_plots']
                return (session['ternary_plot'], binary_plots[0], binary_plots[1], binary_plots[2], 
                       "", True, False)

        # While waiting, show loading animation only if button was clicked
        if session['button_clicked']:
            loading_message = html.Div([
                html.Span("Takes up to 2 minutes to generate plot"),
                html.Div(className="loading-spinner", style={'marginLeft': '8px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'fontSize': '13px'})
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, loading_message, False, True
        else:
            # Initial state - return landing plots
            binary_plots = session['binary_plots']
            return (session['ternary_plot'], binary_plots[0], binary_plots[1], binary_plots[2], 
                   "Enter input and click 'Generate Plot' to see the result.", False, False)

    return app


if __name__ == '__main__':
    app = create_gliqtern_app(requests_pathname_prefix="/gliquid/ternary-interpolation/")
    app.run_server(debug=True)
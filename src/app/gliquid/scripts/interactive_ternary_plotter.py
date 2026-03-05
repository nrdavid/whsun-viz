'''
Authors: Abrar Rauf & Joshua Willwerth

This script generates a Dash web app for the G-Liquid Ternary Interpolation project.
The app allows users to input a ternary system to generate the interpolated 
ternary liquidus and corresponding binary phase diagrams. 
'''
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import threading
import flask
from gliquid.scripts.ternary_hsx import ternary_gtx_plotter
import gliquid.scripts.config as cfg
import pandas as pd
import json
import uuid
from pathlib import Path
import copy

def load_landing_figures():
    """
    Load pre-generated landing figures from the ternary_cache directory.
    Falls back to generating them if cache files don't exist.
    """
    cache_dir = Path(cfg.project_root) / "gliquid" / "ternary_cache"
    
    # Try to load from JSON files
    try:
        ternary_json_path = cache_dir / "landing_ternary.json"
        binary_paths = [cache_dir / f"landing_binary_{i}.json" for i in range(1, 4)]
        
        if not ternary_json_path.exists() or not all(p.exists() for p in binary_paths):
            raise FileNotFoundError("Landing cache JSON files not found")
        
        # Load ternary figure from JSON string
        with open(ternary_json_path, 'r', encoding='utf-8') as f:
            ternary_json_str = f.read()
        landing_ternary_dict = json.loads(ternary_json_str)
        landing_ternary = go.Figure(landing_ternary_dict)
        
        # Load binary figures from JSON
        landing_binaries = []
        for path in binary_paths:
            with open(path, 'r', encoding='utf-8') as f:
                binary_json_str = f.read()
            binary_dict = json.loads(binary_json_str)
            landing_binaries.append(go.Figure(binary_dict))
        
        print("Successfully loaded landing figures from JSON cache")
        return landing_ternary, landing_binaries
        
    except Exception as e:
        print(f"Could not load from cache ({e}), generating landing figures...")
        # Generate Fe-Ce-Si as the default landing system
        return generate_landing_system()

def generate_landing_system():
    """Generate the Fe-Ce-Si system as the landing page"""
    text_input = ['Ce', 'Fe', 'Si']
    upper_increment = 0.0
    lower_increment = 0.0
    interp_type = 'linear'
    param_format = 'combined'
    
    try:
        binary_param_df = pd.read_excel(f"{cfg.data_dir}/fitted_params.xlsx")
        binary_param_pred_df = pd.read_excel(f"{cfg.data_dir}/predicted_params.xlsx")
        
        binary_sys_labels = [
            f"{text_input[0]}-{text_input[1]}", 
            f"{text_input[1]}-{text_input[2]}", 
            f"{text_input[2]}-{text_input[0]}"
        ]
        
        binary_L_dict = {}
        fitorpred = {}
        
        for bin_sys in binary_sys_labels:
            flipped_sys = "-".join(sorted(bin_sys.split('-')))
            order_changed = (bin_sys != flipped_sys)
            
            if bin_sys in binary_param_df['system'].tolist() or flipped_sys in binary_param_df['system'].tolist():
                if bin_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == bin_sys].iloc[0]
                else:
                    params = binary_param_df[binary_param_df['system'] == flipped_sys].iloc[0]
                fitorpred[bin_sys] = "fit"
            elif bin_sys in binary_param_pred_df['system'].tolist() or flipped_sys in binary_param_pred_df['system'].tolist():
                if bin_sys in binary_param_pred_df['system'].tolist():
                    params = binary_param_pred_df[binary_param_pred_df['system'] == bin_sys].iloc[0]
                else:
                    params = binary_param_pred_df[binary_param_pred_df['system'] == flipped_sys].iloc[0]
                fitorpred[bin_sys] = "pred"
            else:
                raise ValueError(f"Binary system {bin_sys} not found")
            
            L0_a, L0_b = float(params["L0_a"]), float(params["L0_b"])
            L1_a, L1_b = float(params["L1_a"]), float(params["L1_b"])
            
            if order_changed:
                L1_a, L1_b = -L1_a, -L1_b
            
            binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]
        
        temp_slider = [lower_increment, upper_increment]
        plotter = ternary_gtx_plotter(
            text_input, cfg.data_dir, 
            interp_type=interp_type, 
            param_format=param_format,
            L_dict=binary_L_dict, 
            temp_slider=temp_slider, 
            T_incr=10.0, 
            delta=0.025, 
            fit_or_pred=fitorpred
        )
        
        plotter.interpolate()
        plotter.process_data()
        
        sub_width, sub_height = 400, 300
        tern_width, tern_height = 700, 900
        
        ternary_plot = plotter.plot_ternary()
        ternary_plot.update_layout(
            title=f"<b>Interpolated {plotter.tern_sys_name} Ternary Phase Diagram</b>",
            showlegend=True, 
            width=tern_width, 
            height=tern_height, 
            font=dict(size=14, color='black')
        )
        
        binary_plots = []
        for bin_fig in plotter.bin_fig_list:
            bin_fig.update_layout(showlegend=False, width=sub_width, height=sub_height, font=dict(size=10))
            binary_plots.append(bin_fig)
        
        print("Successfully generated landing system: Fe-Ce-Si")
        return ternary_plot, binary_plots
        
    except Exception as e:
        print(f"Error generating landing system: {e}")
        # Return empty figures as fallback
        return go.Figure(), [go.Figure(), go.Figure(), go.Figure()]

def create_gliqtern_app(requests_pathname_prefix):
    gliq_app = flask.Flask(__name__)
    app = dash.Dash(__name__, server=gliq_app, requests_pathname_prefix=requests_pathname_prefix, 
                    assets_folder=f"{cfg.project_root}/gliquid/")
    app.title = "Ternary Plotter"
    
    # Load landing figures once at app startup
    LANDING_TERNARY, LANDING_BINARIES = load_landing_figures()
    
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

    interp_type = 'linear'  
    param_format = 'combined'

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
            print(f"Generating plot for: {text_input} with interpolation type: {interp_type}")
            
            # ASSERT: Elements should be valid after sorting
            assert all(len(e) <= 2 and e.isalpha() for e in text_input), f"Invalid element symbols: {text_input}"
            
            temp_slider = [lower_increment, upper_increment]
            binary_param_df = pd.read_excel(f"{cfg.data_dir}/fitted_params.xlsx")
            binary_param_pred_df = pd.read_excel(f"{cfg.data_dir}/predicted_params.xlsx")
            
            # ASSERT: Parameter dataframes have required columns
            required_cols = ['system', 'L0_a', 'L0_b', 'L1_a', 'L1_b']
            assert all(col in binary_param_df.columns for col in required_cols), f"Fitted params missing columns: {set(required_cols) - set(binary_param_df.columns)}"
            assert all(col in binary_param_pred_df.columns for col in required_cols), f"Predicted params missing columns: {set(required_cols) - set(binary_param_pred_df.columns)}"
            
            binary_sys_labels = [
                f"{text_input[0]}-{text_input[1]}", f"{text_input[1]}-{text_input[2]}", f"{text_input[2]}-{text_input[0]}"
            ]
            print("Binary System Labels: ", binary_sys_labels)
            
            # ASSERT: Binary labels should have exactly 3 systems for ternary
            assert len(binary_sys_labels) == 3, f"Expected 3 binary systems, got {len(binary_sys_labels)}"
            binary_L_dict = {}
            fitorpred = {}
            for bin_sys in binary_sys_labels:
                flipped_sys = "-".join(sorted(bin_sys.split('-')))
                order_changed = (bin_sys != flipped_sys)
                
                # ASSERT: Flipped system should be alphabetically ordered
                components = flipped_sys.split('-')
                assert components == sorted(components), f"Flipped system {flipped_sys} is not alphabetically ordered"

                # Prioritize fitted params over predicted - check fitted dataframe first
                if bin_sys in binary_param_df['system'].tolist() or flipped_sys in binary_param_df['system'].tolist():
                    # Found in fitted params
                    if bin_sys in binary_param_df['system'].tolist():
                        params = binary_param_df[binary_param_df['system'] == bin_sys].iloc[0]
                    else:
                        params = binary_param_df[binary_param_df['system'] == flipped_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                    
                    # ASSERT: When found in fitted params, should be marked as 'fit'
                    assert fitorpred[bin_sys] == "fit", f"Binary {bin_sys} found in fitted but marked as {fitorpred[bin_sys]}"
                    
                elif bin_sys in binary_param_pred_df['system'].tolist() or flipped_sys in binary_param_pred_df['system'].tolist():
                    # Found in predicted params
                    if bin_sys in binary_param_pred_df['system'].tolist():
                        params = binary_param_pred_df[binary_param_pred_df['system'] == bin_sys].iloc[0]
                    else:
                        params = binary_param_pred_df[binary_param_pred_df['system'] == flipped_sys].iloc[0]
                    fitorpred[bin_sys] = "pred"
                    
                    # ASSERT: When found in predicted params, should be marked as 'pred'
                    assert fitorpred[bin_sys] == "pred", f"Binary {bin_sys} found in predicted but marked as {fitorpred[bin_sys]}"
                    
                else:
                    raise ValueError(f"Binary system {bin_sys} not found in the parameter dataframe.")

                
                L0_a = float(params["L0_a"])
                L0_b = float(params["L0_b"])
                L1_a = float(params["L1_a"])
                L1_b = float(params["L1_b"])
                
                # Store original values for assertion
                original_L1_a = L1_a
                original_L1_b = L1_b
                
                if order_changed:
                    L1_a = -L1_a
                    L1_b = -L1_b
                    
                    # ASSERT: L1 parameters should be negated when order changed
                    assert L1_a == -original_L1_a, f"L1_a not properly negated: {L1_a} != -{original_L1_a}"
                    assert L1_b == -original_L1_b, f"L1_b not properly negated: {L1_b} != -{original_L1_b}"

                binary_L_dict[bin_sys] = [L0_a, L0_b, L1_a, L1_b]
                
                # ASSERT: Parameter array should have exactly 4 elements
                assert len(binary_L_dict[bin_sys]) == 4, f"Parameter array for {bin_sys} should have 4 elements, got {len(binary_L_dict[bin_sys])}"
                
            print(fitorpred)
            print("Binary Interaction Parameters: ", binary_L_dict)
            
            # ASSERT: L_dict and fitorpred should have same keys
            assert set(binary_L_dict.keys()) == set(fitorpred.keys()), f"Key mismatch: L_dict keys {set(binary_L_dict.keys())} != fitorpred keys {set(fitorpred.keys())}"
            
            # ASSERT: All binary systems should be accounted for
            assert len(binary_L_dict) == 3, f"Expected 3 binary systems in L_dict, got {len(binary_L_dict)}"
            assert len(fitorpred) == 3, f"Expected 3 binary systems in fitorpred, got {len(fitorpred)}"
            
            plotter = ternary_gtx_plotter(text_input, cfg.data_dir, interp_type=interp_type, param_format=param_format, 
                                          L_dict=binary_L_dict, temp_slider=temp_slider, T_incr=10.0, delta = 0.025, fit_or_pred=fitorpred)
            plotter.interpolate()
            plotter.process_data()

            sub_width = 400
            sub_height = 300
            tern_width = 700
            tern_height = 900
            # Generate the plots
            ternary_plot = plotter.plot_ternary()
            ternary_plot.update_layout(title=f"<b>Interpolated {plotter.tern_sys_name} Ternary Phase Diagram</b>", showlegend=True, width=tern_width, height=tern_height, font=dict(size=14, color='black'))

            # ASSERT: Plotter should have 3 binary figures
            assert hasattr(plotter, 'bin_fig_list'), "Plotter should have bin_fig_list attribute"
            assert len(plotter.bin_fig_list) == 3, f"Expected 3 binary figures, got {len(plotter.bin_fig_list)}"

            binary_plot_1 = plotter.bin_fig_list[0]
            binary_plot_1.update_layout(showlegend=False, width=sub_width, height=sub_height, font=dict(size=10))

            binary_plot_2 = plotter.bin_fig_list[1]
            binary_plot_2.update_layout(showlegend=False, width=sub_width, height=sub_height, font=dict(size=10))

            binary_plot_3 = plotter.bin_fig_list[2]
            binary_plot_3.update_layout(showlegend=False, width=sub_width, height=sub_height, font=dict(size=10))
            
            # ASSERT: All plots should be valid Plotly figures
            assert isinstance(ternary_plot, go.Figure), "Ternary plot should be a Plotly Figure"
            assert isinstance(binary_plot_1, go.Figure), "Binary plot 1 should be a Plotly Figure"
            assert isinstance(binary_plot_2, go.Figure), "Binary plot 2 should be a Plotly Figure"
            assert isinstance(binary_plot_3, go.Figure), "Binary plot 3 should be a Plotly Figure"

            # Store in session
            session['ternary_plot'] = ternary_plot
            session['binary_plots'] = [binary_plot_1, binary_plot_2, binary_plot_3]
            session['plot_ready'] = True
            
        except Exception as e:
            print(f"Error occurred during plot generation: {str(e)}")
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
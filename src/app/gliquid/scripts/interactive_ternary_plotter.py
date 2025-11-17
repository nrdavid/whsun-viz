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

def create_gliqtern_app(requests_pathname_prefix):
    gliq_app = flask.Flask(__name__)
    app = dash.Dash(__name__, server=gliq_app, requests_pathname_prefix=requests_pathname_prefix, 
                    assets_folder=f"{cfg.project_root}/gliquid/")
    app.title = "Ternary Plotter"

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

    # Global variables to store figures and readiness
    ternary_plot = go.Figure()
    binary_plot_1 = go.Figure()
    binary_plot_2 = go.Figure()
    binary_plot_3 = go.Figure()
    plot_ready = False
    error_occurred = False
    error_message = ""
    button_clicked = False

    def generate_plot(text_input, upper_increment, lower_increment):
        nonlocal ternary_plot, binary_plot_1, binary_plot_2, binary_plot_3, plot_ready, error_occurred, error_message
        
        try:
            error_occurred = False
            error_message = ""
            
            text_input = text_input.split('-')
            text_input = sorted(text_input)
            print(f"Generating plot for: {text_input} with interpolation type: {interp_type}")
            temp_slider = [lower_increment, upper_increment]
            binary_param_df = pd.read_excel(f"{cfg.data_dir}/fitted_params.xlsx")
            binary_param_pred_df = pd.read_excel(f"{cfg.data_dir}/predicted_params.xlsx")
            binary_sys_labels = [
                f"{text_input[0]}-{text_input[1]}", f"{text_input[1]}-{text_input[2]}", f"{text_input[2]}-{text_input[0]}"
            ]
            print("Binary System Labels: ", binary_sys_labels)
            binary_L_dict = {}
            fitorpred = {}
            for bin_sys in binary_sys_labels:
                flipped_sys = "-".join(sorted(bin_sys.split('-')))

                if bin_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == bin_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                elif flipped_sys in binary_param_df['system'].tolist():
                    params = binary_param_df[binary_param_df['system'] == flipped_sys].iloc[0]
                    fitorpred[bin_sys] = "fit"
                elif bin_sys in binary_param_pred_df['system'].tolist():
                    params = binary_param_pred_df[binary_param_pred_df['system'] == bin_sys].iloc[0]
                    fitorpred[bin_sys] = "pred"
                elif flipped_sys in binary_param_pred_df['system'].tolist():
                    params = binary_param_pred_df[binary_param_pred_df['system'] == flipped_sys].iloc[0]
                    fitorpred[bin_sys] = "pred"
                else:
                    raise ValueError(f"Binary system {bin_sys} not found in the parameter dataframe.")

                binary_L_dict[bin_sys] = [
                    float(params["L0_a"]),
                    float(params["L0_b"]),
                    float(params["L1_a"]),
                    float(params["L1_b"])
                ]
                
            print(fitorpred)
            print("Binary Interaction Parameters: ", binary_L_dict)
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

            binary_plot_1 = plotter.bin_fig_list[0]
            binary_plot_1.update_layout(showlegend=False, width=sub_width, height=sub_height, font=dict(size=10))

            binary_plot_2 = plotter.bin_fig_list[1]
            binary_plot_2.update_layout(showlegend=False, width=sub_width, height=sub_height, font=dict(size=10))

            binary_plot_3 = plotter.bin_fig_list[2]
            binary_plot_3.update_layout(showlegend=False, width=sub_width, height=sub_height, font=dict(size=10))

            plot_ready = True
            
        except Exception as e:
            print(f"Error occurred during plot generation: {str(e)}")
            error_occurred = True
            error_message = "Invalid or unsupported system, please try again."
            plot_ready = True  


    app.layout = html.Div(
        [ 
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
                    html.A("Binary Phase Diagram Map", href="https://viz.whsunresearch.group/gliquid/interactive-matrix.html", target="_blank", style={'fontSize': '14px'}),
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
        [State('text-input', 'value'), State('upper_increment', 'value'), State('lower_increment', 'value')]
    )
    def trigger_and_update_plot(n_clicks, n_intervals, text_input, upper_increment, lower_increment):
        nonlocal ternary_plot, binary_plot_1, binary_plot_2, binary_plot_3, plot_ready, error_occurred, error_message, button_clicked

        # Identify what triggered the callback
        ctx = dash.callback_context

        # If the button is clicked, start generating the plot in a separate thread
        if ctx.triggered and 'submit-val' in ctx.triggered[0]['prop_id']:
            button_clicked = True
            plot_ready = False
            error_occurred = False
            error_message = ""
            thread = threading.Thread(target=generate_plot, args=(text_input, upper_increment, lower_increment))
            thread.start()
            
            # Create animated loading message
            loading_message = html.Div([
                html.Span("Takes up to 2 minutes to generate plot"),
                html.Div(className="loading-spinner", style={'marginLeft': '8px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'fontSize': '13px'})
            
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, loading_message, False, True

        # If the interval triggered the callback, check if the plot is ready
        if plot_ready:
            if error_occurred:
                # Return empty plots and show error message
                empty_fig = go.Figure()
                return empty_fig, empty_fig, empty_fig, empty_fig, error_message, True, False
            else:
                # Return successful plots and clear message
                return ternary_plot, binary_plot_1, binary_plot_2, binary_plot_3, "", True, False

        # While waiting, show loading animation only if button was clicked
        if button_clicked:
            loading_message = html.Div([
                html.Span("Takes up to 2 minutes to generate plot"),
                html.Div(className="loading-spinner", style={'marginLeft': '8px'})
            ], style={'display': 'flex', 'alignItems': 'center', 'fontSize': '13px'})
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, loading_message, False, True
        else:
            # Initial state - no loading message
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Enter input and click 'Generate Plot' to see the result.", False, False

    return app


if __name__ == '__main__':
    app = create_gliqtern_app(requests_pathname_prefix="/gliquid/ternary-interpolation/")
    app.run_server(debug=True)
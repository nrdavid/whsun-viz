'''
Author: Abrar Rauf & Joshua Willwerth

This script generates a Dash web app for the G-Liquid Ternary Interpolation project.
The app allows users to input a ternary system and select an interpolation type to generate the interpolated 
ternary liquidus and corresponding binary phase diagrams. 
'''
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import threading
import flask
from gliquid_ternary_interpolation.ternary_HSX import ternary_gtx_plotter
import pandas as pd
# from ternary_HSX import ternary_hsx_plotter

def create_gliqtern_app(requests_pathname_prefix):
    gliq_app = flask.Flask(__name__)
    app = dash.Dash(__name__, server=gliq_app, requests_pathname_prefix=requests_pathname_prefix)

    interp_type = 'linear'  # Default interpolation type
    param_format = 'combined'

    # Global variables to store figures and readiness
    ternary_plot = go.Figure()
    binary_plot_1 = go.Figure()
    binary_plot_2 = go.Figure()
    binary_plot_3 = go.Figure()
    plot_ready = False

    def generate_plot(text_input, upper_increment, lower_increment):
        nonlocal ternary_plot, binary_plot_1, binary_plot_2, binary_plot_3, plot_ready
        dir = "gliquid_ternary_interpolation/matrix_data_jsons/"
        text_input = text_input.split('-')
        text_input = sorted(text_input)
        print(f"Generating plot for: {text_input} with interpolation type: {interp_type}")
        temp_slider = [lower_increment, upper_increment]
        binary_param_df = pd.read_excel(dir + "multi_fit_no1S_nmae_lt_0.5.xlsx")
        binary_param_pred_df = pd.read_excel(dir + "v17_comb1S_tau10k_predictions_rf_optimized.xlsx")
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
        plotter = ternary_gtx_plotter(text_input, dir, interp_type=interp_type, param_format=param_format, L_dict=binary_L_dict, temp_slider=temp_slider, T_incr=5.0, delta = 0.025, fit_or_pred=fitorpred)
        plotter.interpolate()
        plotter.process_data()

        sub_width = 420
        sub_height = 350
        tern_width = 900
        tern_height = 1000
        # Generate the plots
        ternary_plot = plotter.plot_ternary()
        ternary_plot.update_layout(title=f"<b>Interpolated {plotter.tern_sys_name} Ternary Phase Diagram</b>", showlegend=True, width=tern_width, height=tern_height, font=dict(size=14))

        binary_plot_1 = plotter.bin_fig_list[0]
        binary_plot_1.update_layout(showlegend=False, width=sub_width, height=sub_height, font=dict(size=10))

        binary_plot_2 = plotter.bin_fig_list[1]
        binary_plot_2.update_layout(showlegend=False, width=sub_width, height=sub_height, font=dict(size=10))

        binary_plot_3 = plotter.bin_fig_list[2]
        binary_plot_3.update_layout(showlegend=False, width=sub_width, height=sub_height, font=dict(size=10))

        plot_ready = True


    app.layout = html.Div(
        [
            # Left panel for description and input fields
            html.Div(
                [
                    html.H2("G-Liquid Ternary Plotter", style={'fontsize': '14px'}),
                    html.P("This web app generates an interpolated ternary liquidus for the specified ternary system using fitted/predicted binary interaction parameters from the Sun Lab G-Liquid Project.", style={'fontSize': '14px'}),
                    html.A("Binary Interaction Map", href="https://viz.whsunresearch.group/gliquid/interactive_matrix.html", target="_blank", style={'fontSize': '14px'}),
                    html.Br(),
                    html.Br(),
                    html.P(html.B("Usage Instructions:"), style={'fontSize': '14px'}),
                    html.P("Specify the system and select the interpolation type to generate the ternary and corresponding binary phase diagrams.", style={'fontSize': '14px'}),
                    html.Label("System: ", style={'fontSize': '14px'}),
                    dcc.Input(id='text-input', type='text', value='', placeholder="e.g., Bi-Cd-Sn", style={'fontSize': '14px'}),
                    html.Br(),
                    html.Br(),
                    html.P("For some systems, the default temperature range may not capture the entire liquidus. Manually input values to the temperature slider to decrement or increment the lower and upper temperature bounds", style={'fontSize': '14px'}),
                    html.P("Temperature Axis Slider:", style={'fontSize': '14px'}),
                    html.Label("Increment Upper Bound:", style={'fontSize': '14px'}),
                    dcc.Input(id='upper_increment', type='number', value=0.0, style={'fontSize': '14px'}),
                    html.Br(),
                    html.Label("Decrement Lower Bound:", style={'fontSize': '14px'}),
                    dcc.Input(id='lower_increment', type='number', value=0.0, style={'fontSize': '14px'}),
                    html.Br(),
                    html.Br(),
                    html.Button('Generate Plot', id='submit-val', n_clicks=0),
                    html.Div(id='loading-message', children="Enter input and click 'Generate Plot' to see the result."),
                    html.Br(),
                    html.P(html.B("By Abrar Rauf (arauf@umich.edu), Joshua Willwerth, Shibo Tan, Wenhao Sun(whsun@umich.edu)"), style={'fontSize': '12px'}),
                    html.P(html.I("Note: The accuracy of the ternary liquidus reconstruction is a work-in-progress and is not guaranteed to work as intended for all ternary systems."), style={'fontSize': '12px'})
                ],
                style={
                    'width': '15%', 'height': '100vh', 'padding': '10px',
                    'position': 'fixed', 'left': 0, 'top': 0, 'backgroundColor': '#f8f9fa',
                    'boxShadow': '2px 0 5px rgba(0,0,0,0.1)', 'overflowY': 'auto',
                    'display': 'inline-block',  # Prevent overlap with the right section
                    'verticalAlign': 'top'
                }
            ),

            # Right side main plot area
            html.Div(
                [
                    # Two-column layout: one for binary plots (left) and one for the ternary plot (right)
                    html.Div(
                        [
                            # Left column for binary plots
                            html.Div(
                                [
                                    dcc.Graph(id='binary-plot-1', style={'height': '30vh', 'width': '100%'}),
                                    dcc.Graph(id='binary-plot-2', style={'height': '30vh', 'width': '100%'}),
                                    dcc.Graph(id='binary-plot-3', style={'height': '30vh', 'width': '100%'})
                                ],
                                style={
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'width': '30%',  # Left column width
                                    'margin-right': '2%'  # Spacing between columns
                                }
                            ),

                            # Right column for the ternary plot
                            html.Div(
                                dcc.Graph(id='ternary-plot', style={'height': '90vh', 'width': '100%'}),
                                style={
                                    'width': '65%',  # Right column width
                                    'margin-left': 'auto',
                                    'paddingTop': '30px'
                                }
                            )
                        ],
                        style={
                            'display': 'flex',
                            'flexDirection': 'row',
                            'margin-left': '20%',  # Adjust for left panel width
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
            Output('interval-component', 'disabled')],
        [Input('submit-val', 'n_clicks'),
            Input('interval-component', 'n_intervals')],
        [State('text-input', 'value'), State('upper_increment', 'value'), State('lower_increment', 'value')]
    )
    def trigger_and_update_plot(n_clicks, n_intervals, text_input, upper_increment, lower_increment):
        nonlocal ternary_plot, binary_plot_1, binary_plot_2, binary_plot_3, plot_ready

        # Identify what triggered the callback
        ctx = dash.callback_context

        # If the button is clicked, start generating the plot in a separate thread
        if ctx.triggered and 'submit-val' in ctx.triggered[0]['prop_id']:
            plot_ready = False
            thread = threading.Thread(target=generate_plot, args=(text_input, upper_increment, lower_increment))
            thread.start()
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Takes up to 2 minutes to generate plot...", False

        # If the interval triggered the callback, check if the plot is ready
        if plot_ready:
            return ternary_plot, binary_plot_1, binary_plot_2, binary_plot_3, "", True

        # While waiting, do not update the plot, keep the interval running
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, "Takes up to 2 minutes to generate plot...", False

    return app


if __name__ == '__main__':
    app = create_gliqtern_app(requests_pathname_prefix="/gliquid_ternary_interpolation/")
    app.run_server(debug=True)
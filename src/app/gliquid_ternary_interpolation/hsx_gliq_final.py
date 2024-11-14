'''___Imports___'''
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
from collections import defaultdict


def remove_duplicates(coordinates):
    coordinates.sort(key=lambda x: (x[0], x[1]))
    filtered_coordinates = []
    current_x = None
    for coord in coordinates:
        if coord[0] != current_x:
            filtered_coordinates.append(coord)
            current_x = coord[0]
    return filtered_coordinates


def gliq_lowerhull(points, liq_points, intermetallics):
    # Function to calculate the general lower convex hull of an N-dimensional Xi-S-H space
    # Input: points = array of coordinates of the points in the Xi-S-H space
    # Output: simplices = array of simplices that form the lower convex hull of the Xi-S-H space

    # determine the dimensionality of the points
    dim = points.shape[1]

    # initialize bounds for Xi 
    x_list = []
    for i in range(dim - 1):
        sublist = []
        for j in range(dim - 2):
            if i == 0:
                sublist.append(0)
            else:
                if j == i - 1:
                    sublist.append(1)
                else:
                    sublist.append(0)

        x_list.append(sublist)

    # initialize bounds for S and H
    s_index = dim - 2
    s_data = points[:, s_index]
    s_min = np.min(s_data)

    # entropy of liquid
    s_liq_data = liq_points[:, s_index]
    s_extr = np.max([s_liq_data[0], s_liq_data[-1]])

    h_index = dim - 1
    h_data = points[:, h_index]
    h_max = np.max(h_data)
    h_min = np.min(h_data)
    upper_bound = h_max + 2*abs(h_max - h_min) # alternative give to josh after his next new run to test improvements

    # create a list of liq fictitious points
    liq_fict_coords = []
    for point in liq_points:
        x_coord = point[0]
        s_coord = point[1]
        h_coord = upper_bound
        liq_fict_coords.append([x_coord, s_coord, h_coord])

    # create a list of fictitious points
    fict_coords = []
    for i in range(dim - 1):
        fict_coord = []
        for j in range(dim - 2):
            fict_coord.append(x_list[i][j])
        fict_coord.append(s_min)
        fict_coord.append(upper_bound)
        fict_coords.append(fict_coord)

    for i in range(dim - 1):
        fict_coord = []
        for j in range(dim - 2):
            fict_coord.append(x_list[i][j])
        fict_coord.append(s_extr)
        fict_coord.append(upper_bound)
        fict_coords.append(fict_coord)

    fict_coords = np.array(fict_coords)
    liq_fict_coords = np.array(liq_fict_coords)

    # add the fictitious points to the original distribution
    fict_points = np.array(fict_coords)
    fict_points = np.vstack((fict_points, liq_fict_coords))
    new_points = np.vstack((points, fict_points))

    # take the total convex hull
    new_hull = ConvexHull(new_points, qhull_options="Qt i")

    # iterate through these new simplices and delete all simplices that touch the fictitious points.
    # remaining simplices will belong to the lower hull
    lower_hull_filter1 = []
    lower_hull_filter2 = []
    lower_hull = []

    def check_common_rows(arr1, arr2):
        # print('here')
        return any((arr1 == row).all(axis=1).any() for row in arr2)

    for simplex in new_hull.simplices:
        if not check_common_rows(new_points[simplex], fict_points):
            lower_hull_filter1.append(simplex)

    for simplex in lower_hull_filter1:
        vertices = points[simplex]
        count = 0
        for vertex in vertices:
            for intermetallic in intermetallics:
                if (vertex == intermetallic).all():
                    count += 1
        if count < 3:
            lower_hull_filter2.append(simplex)

    for simplex in lower_hull_filter2:
        vertices = points[simplex]
        count = 0
        for vertex in vertices:
            for liq in liq_points:
                if (vertex == liq).all():
                    count += 1
        if count < 3:
            lower_hull.append(simplex)

    arr_lowerhull = np.array(lower_hull_filter2) # change to lower_hull to exclude misc gaps

    return arr_lowerhull


# dict = {data: {x: [], S: [], H: [], Phase Name: []}, phases: [], comps: []}

def dict_construct(filename):
    filename = filename
    df = pd.read_csv(filename)
    phasefile = f'{filename[:-4]}_phases.txt'
    compfile = f'{filename[:-4]}_comps.txt'
    phases = []
    comps = []
    with open(phasefile, 'r') as f:
        for line in f:
            phases.append(line.strip())

    with open(compfile, 'r') as f:
        for line in f:
            comps.append(line.strip())
    x_values = df['X'].tolist()
    s_values = df['S'].tolist()
    h_values = df['H'].tolist()
    phase_values = df['Phase Name'].tolist()

    data_dict = {'data': {'X': x_values, 'S': s_values, 'H': h_values, 'Phase Name': phase_values}, 'phases': phases,
                 'comps': comps}
    return data_dict


class HSX:
    def __init__(self, data_dict, conds):
        self.phases = data_dict['phases']
        self.comps = data_dict['comps']
        self.conds = conds
        self.df = pd.DataFrame(data_dict['data'])
        self.phase_color_remap = {}
        self.simplices = []
        self.final_phases = []
        self.df_tx = pd.DataFrame()
        self.df.columns = ['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]', 'Phase']
        s_scaler = 100
        h_scaler = 10000
        self.df['S [J/mol/K]'] = self.df['S [J/mol/K]']/s_scaler
        self.df['H [J/mol]'] = self.df['H [J/mol]']/h_scaler
        color_array = px.colors.qualitative.Pastel
        df_inter = self.df[self.df['Phase'] != 'L']
        df_liq = self.df[self.df['Phase'] == 'L']
        self.liq_points = np.array(df_liq[['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]']])
        self.inter_points = np.array(df_inter[['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]']])
        self.points = np.array(self.df[['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]']])
        inter_phases = self.phases.copy()
        if 'L' in inter_phases:
            inter_phases.remove('L')
        self.color_map = dict(zip(inter_phases, color_array))
        self.color_map['L'] = 'cornflowerblue'
        # self.color_map = dict(zip(self.phases, color_array))
        self.df['Colors'] = self.df['Phase'].map(self.color_map)
        self.scaler = h_scaler / s_scaler

    def hull(self):
        self.simplices = gliq_lowerhull(self.points, self.liq_points, self.inter_points)
        return self.simplices

    def compute_tx(self):
        temps = []
        self.simplices = self.hull()

        # print(self.simplices)

        for simplex in self.simplices:
            A = self.points[simplex[0]]
            B = self.points[simplex[1]]
            C = self.points[simplex[2]]

            # Calculate vectors AB and AC
            AB = B - A
            AC = C - A

            AB = AB.astype(float)
            AC = AC.astype(float)
    
            # Calculate normal vector to the plane
            n = np.cross(AB, AC)

            # Calculate the temperature at the plane
            T = (-n[1] / n[2])*(self.scaler)

            # Append the plane and temperature to the list
            temps.append(T)

        temps = np.array(temps)
        nan_indices = np.where(np.isnan(temps))[0]
        final_temps = temps[~np.isnan(temps)]
        # final_temps = final_temps - 273.15
        final_simplices = np.delete(self.simplices, nan_indices, axis=0)

        new_phases = []
        for simplex in self.simplices:
            phase1 = self.df.loc[simplex[0], 'Phase']
            phase2 = self.df.loc[simplex[1], 'Phase']
            phase3 = self.df.loc[simplex[2], 'Phase']

            phase_arr = np.array([phase1, phase2, phase3])

            new_phases.append(phase_arr)

        new_phases = np.array(new_phases)
        self.final_phases = new_phases[~np.isnan(temps)]

        data = []
        for i, simplex in enumerate(final_simplices):
            x_coords = [self.points[vertex][0] for vertex in simplex]
            t_value = final_temps[i]
            labels = self.final_phases[i]

            for j, x in enumerate(x_coords):
                label = labels[j]
                color = self.color_map.get(label)
                data.append([x, t_value, label, color])

        self.df_tx = pd.DataFrame(data, columns=['x', 't', 'label', 'color'])

        # drop all rows containing temps above the self.conds[1]
        # self.df_tx = self.df_tx[self.df_tx['t'] <= self.conds[1] + 273.15]

        phase_remap = {}
        for entry in data:
            label = entry[2]
            values = [entry[0], entry[1]]

            if label not in phase_remap:
                phase_remap[label] = []

            phase_remap[label].append(values)

        self.phase_color_remap = {label: color for label, color in zip(self.df_tx['label'], self.df_tx['color'])}

        return self.df_tx, self.final_phases, final_simplices, final_temps

    def plot_hsx(self):
        # Create a figure
        self.simplices = self.hull()
        fig = go.Figure()

        # Scatter plot
        scatter = go.Scatter3d(
            x=self.df['X [Fraction]'], y=self.df['S [J/mol/K]'], z=self.df['H [J/mol]'],
            mode='markers',
            marker=dict(size=6, opacity=1, color=self.df['Colors']),
            name='Scatter',
            showlegend=False
        )

        fig.add_trace(scatter)

        for simplex in self.simplices:
            x_coords = self.points[simplex, 0]
            y_coords = self.points[simplex, 1]
            z_coords = self.points[simplex, 2]

            i = np.array([0])
            j = np.array([1])
            k = np.array([2])

            trace = go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, alphahull=5, opacity=0.3, color='cyan', i=i, j=j, k=k,
                              name='Simplex')

            # Add both the triangle and vertex traces to the figure
            fig.add_trace(trace)

        # Create legend entries
        legend_elements = []

        for name, color in self.color_map.items():
            legend_elements.append(dict(x=0, y=0, xref='paper', yref='paper', text=name, marker=dict(color=color)))

        # Add legend entries
        for entry in legend_elements:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines+text', marker=dict(color=entry['marker']['color']),
                                     name=entry['text']))

        # Update layout and labels
        fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='S [J/mol/K]', zaxis_title='H [J/mol]'),
            legend=dict(itemsizing='constant'),
            font_size=15
        )

        # Show the 3D plot
        fig.show()

    def liquidus_invariants(self):
        tx_main = self.compute_tx()
        self.df_tx = tx_main[0]
        # convert all temps to Celsius
        self.df_tx['t'] = self.df_tx['t'] - 273.15
        self.final_phases = tx_main[1]
        final_simplices = tx_main[2]
        final_temps = tx_main[3] - 273.15
        compositions = []
        all_points = self.points[final_simplices]
        for point in all_points:
            comp = []
            for vertex in point:
                comp.append(vertex[0])
            compositions.append(comp)
        compositions = np.array(compositions)

        combined_list = []
        for i in range(len(compositions)):
            row_dict = {}
            for j in range(len(compositions[i])):
                row_dict[compositions[i][j]] = self.final_phases[i][j]
            if len(row_dict) == 2:
                for key in row_dict.keys():
                    if key == 0.0:
                        row_dict[key] = self.comps[0]
                    elif key == 1.0:
                        row_dict[key] = self.comps[1]
            combined_list.append([final_temps[i], row_dict])

        # remove elements from self.phases
        int_phases = self.phases.copy()
        int_phases.remove(self.comps[0])
        int_phases.remove(self.comps[1])
        if 'L' in int_phases:
            int_phases.remove('L')

        inv_points = {}
        eutectics = []
        peritectics = []
        peritectic_phases = []
        non_triples = []
        congruents = []
        for comb in combined_list:
            temp = comb[0]
            comb_dict = comb[1]
            sorted_dict = {k: v for k, v in sorted(comb_dict.items())}
            comp = list(sorted_dict.keys())
            phase = list(sorted_dict.values())
            if len(comp) == 3:
                if phase[0] != phase[1] and phase[1] != phase[2] and phase[0] != phase[2]:
                    if phase[1] == 'L':
                        eutectics.append([temp, comp[1], comp, phase])
                    else:
                        peritectics.append([temp, comp[1], comp, phase])
                        peritectic_phases.append(phase[1])
                else:
                    non_triples.append([temp, comp, phase])

        congruents_init = []
        misc_gaps = []
        for entry in non_triples:
            temp = entry[0]
            comp = entry[1]
            phase = entry[2]

            if phase[0] == 'L' and phase[2] != 'L':
                comp_diff = abs(comp[0] - comp[1])
                if comp_diff > 0.012:
                    misc_gaps.append([temp, comp[1], comp, phase])
            elif phase[0] != 'L':
                comp_diff = abs(comp[1] - comp[2])
                if comp_diff > 0.012:
                    misc_gaps.append([temp, comp[1], comp, phase])

            # drop L from phase
            phase = [x for x in phase if x != 'L']
            # if phase is empty, continue
            if not phase:
                continue
            # check if phase is in int_phases
            if phase[0] in int_phases and phase[0] not in peritectic_phases:
                congruents_init.append([temp, comp[0], comp, phase])

        # Group entries by phase label
        grouped_data = defaultdict(list)
        for entry in congruents_init:
            phase_label = entry[3][0]  # Extract phase label
            grouped_data[phase_label].append(entry)

        # Select the entry with the highest temperature for each phase label
        for phase_label, entries in grouped_data.items():
            max_entry = max(entries, key=lambda x: x[0])  # Find entry with max temperature
            congruents.append(max_entry)

        inv_points['Eutectics'] = eutectics
        inv_points['Peritectics'] = peritectics
        inv_points['Congruent Melting'] = congruents
        inv_points['Misc Gaps'] = misc_gaps

        # count the number of eutectics, peritectics, and congruent melting points and put the counts in a count dict
        count_dict = {}
        count_dict['Eutectics'] = len(eutectics)
        count_dict['Peritectics'] = len(peritectics)
        count_dict['Congruent Melting'] = len(congruents)
        count_dict['Misc Gaps'] = len(misc_gaps)

        return inv_points, combined_list, count_dict

    def plot_tx(self, param_type='', mpds_liquidus=[], fitted_liquidus=[]):
        liq_inv = self.liquidus_invariants()
        inv_points = liq_inv[0]
        combined_list = liq_inv[1]
        new_tx = []
        for comb in combined_list:
            temp = comb[0]
            comb_dict = comb[1]
            sorted_dict = {k: v for k, v in sorted(comb_dict.items())}
            comp = list(sorted_dict.keys())
            phase = list(sorted_dict.values())
            if len(comp) == 2:
                new_tx.append([temp, comp, phase])
            else:
                if phase[0] == 'L' and phase[1] == 'L':  # Liquid-Liquid-Solid or Liquid-Liquid-Liquid
                    # remove the item in the 1 index of the list comp
                    comp.pop(0)
                    phase.pop(0)
                    new_tx.append([temp, comp, phase])

                elif phase[1] == 'L' and phase[2] == 'L':  # Solid-Liquid-Liquid
                    # remove the item in the 0 index of the list comp
                    comp.pop(2)
                    phase.pop(2)
                    new_tx.append([temp, comp, phase])
                else:
                    new_tx.append([temp, comp, phase])

        temp_df_tx = []
        for entry in new_tx:
            t = entry[0]
            comp = entry[1]
            phase = entry[2]
            for j, x in enumerate(comp):
                label = phase[j]
                color = self.color_map.get(label)
                temp_df_tx.append([x, t, label, color])
        new_df_tx = pd.DataFrame(temp_df_tx, columns=['x', 't', 'label', 'color'])
        # change the x values to percentages
        new_df_tx['x'] = new_df_tx['x'] * 100

        # if in new_df_tx there are multiple rows with the same x, keep the row with the highest t
        # liq_df = new_df_tx[new_df_tx['label'] == 'L']
        liq_df = self.df_tx[self.df_tx['label'] == 'L']
        liq_df['x'] = liq_df['x'] * 100

        liq_df = liq_df.sort_values(by=['x', 't'])
        liq_df = liq_df.drop_duplicates(subset='x', keep='first')

        # liq_df = liq_df[liq_df['t'] <= self.conds[1]]
        solid_df = new_df_tx[new_df_tx['label'] != 'L']
        element_df1 = solid_df[solid_df['x'] == 0]
        element_df2 = solid_df[solid_df['x'] == 100]
        # only keep the row of element_df1 with the highest t
        element_df1 = element_df1.sort_values(by='t')
        element_df1 = element_df1.drop_duplicates(subset='label', keep='last')
        # only keep the row of element_df2 with the highest t
        element_df2 = element_df2.sort_values(by='t')
        element_df2 = element_df2.drop_duplicates(subset='label', keep='last')
        # combine the dataframes
        element_df = pd.concat([element_df1, element_df2])
        # change the values in the label colum to 'L'
        element_df['label'] = 'L'
        liq_df = pd.concat([liq_df, element_df])
        solid_phases = self.phases.copy()
        if 'L' in solid_phases:
            solid_phases.remove('L')
        # sort liq_df by increasing x 
        liq_df = liq_df.sort_values(by='x')

        # find the lowest liquidus temp in liq_df and store it in low_liq_temp variable
        low_liq_temp = liq_df['t'].min()

        # self.conds = [max(0, self.conds[0] - 50), max(high_t * 1.1 + 50, (self.conds[1] + 273.15))]

        data = []
        if mpds_liquidus:
            X = [point[0] * 100 for point in mpds_liquidus]
            T = [point[1] - 273.15 for point in mpds_liquidus]
            dashed_trace = go.Scatter(x=X, y=T, mode='lines',
                                      line=dict(color='#B82E2E', dash='dash'), showlegend=True)
            data.append(dashed_trace)
        if fitted_liquidus:
            X = [point[0] * 100 for point in fitted_liquidus]
            T = [point[1] - 273.15 for point in fitted_liquidus]
            dashed_trace = go.Scatter(x=X, y=T, mode='lines', name='Fitted Liquidus',
                                      line=dict(color='cornflowerblue', dash='dash'), showlegend=True)
            data.append(dashed_trace)

        fig = go.Figure(
            data=data,
            layout=go.Layout(
                title=f'{self.comps[0]}-{self.comps[1]} ' + param_type + 'Binary Phase Diagram',
                width=960,
                height=700,
                showlegend=True
            ))

        if param_type == 'Predicted ':
            self.phase_color_remap['L'] = '#117733'
            line_plot = px.line(liq_df, x='x', y='t', color='label',
                                color_discrete_map=self.phase_color_remap)
        else:
            line_plot = px.line(liq_df, x='x', y='t', color='label',
                                color_discrete_map=self.phase_color_remap)
        line_data = line_plot.data[0]
        fig.add_trace(line_data)

        solid_comp_list = []
        idx_tracker = 0
        for phase in solid_phases:
            phase_df = solid_df[solid_df['label'] == phase]
            # check if phase_df is empty
            if phase_df.empty:
                self.phases.remove(phase)
                continue
            elif phase in self.comps:
                continue

            solid_comp = phase_df['x'].values
            solid_comp = solid_comp[0] # causing an error for Ni-Si
            solid_comp_list.append(solid_comp)
            # add a row to phase_df
            new_row = {'x': solid_comp, 't': -273.15, 'label': phase, 'color': self.color_map.get(phase)}
            phase_df.loc[len(phase_df)] = new_row
            line = px.line(phase_df, x='x', y='t', color='label', color_discrete_map=self.phase_color_remap)
            fig.add_trace(line.data[0])
            vert_font_size = 10
            if idx_tracker == 0:
                if solid_comp < 5:
                    fig.add_annotation(
                        x=solid_comp + 1.5,
                        y=(self.conds[0] + low_liq_temp) / 2,
                        text=phase,
                        showarrow=False,
                        textangle=-90,
                        borderpad=5,
                        font=dict(
                            size=vert_font_size,
                            color='black'
                        ),
                    )
                else:
                    fig.add_annotation(
                        x=solid_comp - 2.5,
                        y=(self.conds[0] + low_liq_temp) / 2,
                        text=phase,
                        showarrow=False,
                        textangle=-90,
                        borderpad=5,
                        font=dict(
                            size=vert_font_size,
                            color='black'
                        ),
                    )
            else:
                if solid_comp < 5:
                    fig.add_annotation(
                        x=solid_comp + 1.5,
                        y=(self.conds[0] + low_liq_temp) / 2,
                        text=phase,
                        showarrow=False,
                        textangle=-90,
                        borderpad=5,
                        font=dict(
                            size=vert_font_size,
                            color='black'
                        ),
                    )
                elif solid_comp_list[idx_tracker] - solid_comp_list[idx_tracker - 1] < 5:
                    fig.add_annotation(
                        x=solid_comp + 1.5,
                        y=(self.conds[0] + low_liq_temp) / 2,
                        text=phase,
                        showarrow=False,
                        textangle=-90,
                        borderpad=5,
                        font=dict(
                            size=vert_font_size,
                            color='black'
                        ),
                    )
                else:
                    fig.add_annotation(
                        x=solid_comp - 2.5,
                        y=(self.conds[0] + low_liq_temp) / 2,
                        text=phase,
                        showarrow=False,
                        textangle=-90,
                        borderpad=5,
                        font=dict(
                            size=vert_font_size,
                            color='black'
                        ),
                )
            idx_tracker += 1

        for key in inv_points.keys():
            if key == 'Eutectics' or key == 'Peritectics' or key=='Misc Gaps':
                for point in inv_points[key]:
                    comps = point[2]
                    # multiple each value in comps by 100
                    comps = [x * 100 for x in comps]
                    temps = [point[0], point[0], point[0]]
                    line = px.line(x=comps, y=temps)
                    line.update_traces(line=dict(color='Silver'))
                    fig.add_trace(line.data[0])

        fig.update_traces(line=dict(width=4))

        # Define axis limits for the 't' axis
        fig.update_layout(
            yaxis=dict(range=[max(self.conds[0] - 100, -273), self.conds[1] + 100], ticksuffix="  "),
            xaxis=dict(range=[0, 100]),
            xaxis_title=f'{self.comps[1]} (at. %)',
            yaxis_title='T (' + chr(176) + 'C)',
            plot_bgcolor='white',
            showlegend=False,
            font_size=22
        )

        format_color = 'black'

        fig.update_xaxes(
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor=format_color,
            linewidth=2,
            tickcolor=format_color,
            tickformat=".0f"
        )

        fig.update_yaxes(
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor=format_color,
            linewidth=2,
            tickcolor=format_color
        )

        fig.add_annotation(
            x=50,
            y=self.conds[1] + 30,
            text='L',
            showarrow=False,
            font=dict(
                size=18,
                color='black'
            )
        )

        # Show the plot
        # fig.show()
        return fig

    def plot_tx_scatter(self):
        self.df_tx = self.compute_tx()[0]
        # convert all temps to Celsius
        self.df_tx['t'] = self.df_tx['t'] - 273.15
        # Create a scatter plot using Plotly Express
        fig = px.scatter(self.df_tx, x='x', y='t', color='label',
                         color_discrete_map=self.phase_color_remap,
                         title=f'{self.comps[0]}-{self.comps[1]} Binary Phase Diagram', 
                         width=100, height=100)

        fig.update_traces(marker=dict(size=12))
        # Update the layout to include a legend
        fig.update_layout(showlegend=True)

        # Define axis limits for the 't' axis
        fig.update_layout(
            yaxis=dict(range=[self.conds[0], self.conds[1] + 100], ticksuffix="  "),
            xaxis=dict(range=[0, 1]),
            xaxis_title=f'X_{self.comps[1]}',
            yaxis_title='T [C]',
            plot_bgcolor='white',
            showlegend=True,
            font_size=22
        )

        fig.update_xaxes(
            mirror=True,
            ticks="inside",
            showline=True,
            linecolor='gray',
            linewidth=2,
            tickcolor='gray'
        )

        fig.update_yaxes(
            mirror=True,
            ticks="inside",
            showline=True,
            linecolor='gray',
            linewidth=2,
            tickcolor='gray'
        )

        # Show the plot
        fig.show()

    def get_phase_points(self):
        df_tx = self.compute_tx()[0]
        # gives values in Celsius
        phase_points = {}

        for phase in self.phases:
            phase_df = df_tx[df_tx['label'] == phase]

            # if phase == 'L':
            #     phase_df = phase_df.sort_values(by=['x', 't'])
            #     phase_df = phase_df.drop_duplicates(subset='x', keep='first')  # Slower than remove_duplicates()

            # create a list of lists containing the x and t values of the line
            phase_points[phase] = phase_df[['x', 't']].values.tolist()
        phase_points['L'] = remove_duplicates(phase_points['L'])
        return phase_points

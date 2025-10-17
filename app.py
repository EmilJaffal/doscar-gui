import base64
import os
import zipfile
from dash import Dash, dcc, html, Input, Output, State
from dash.dependencies import ALL
from src.doscar import parse_doscar_and_plot
import plotly.io as pio
from io import BytesIO
import numpy as np
import dash
import math  
import plotly.graph_objects as go
import shutil
import mimetypes

pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 1200
pio.kaleido.scope.default_height = 800
pio.kaleido.scope.default_scale = 2

DEFAULTS = {
    "xmin": 0,
    "xmax": 28,
    "ymin": -8,
    "ymax": 2,
    "legend_y": 0.26,
}

app = Dash(__name__)

app.layout = html.Div([
    html.H1("DOSCAR Plotter", style={
        "fontSize": "32px", 
        "fontWeight": "bold", 
        "fontFamily": "DejaVu Sans, Arial, sans-serif", 
        "textAlign": "center", 
        "marginBottom": "20px", 
        "color": "#333"
    }),

    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload ZIP file with POSCAR & DOSCAR', style={
            "backgroundColor": "#007BFF", 
            "color": "white", 
            "padding": "12px 20px", 
            "border": "none", 
            "borderRadius": "5px", 
            "cursor": "pointer",
            "fontSize": "16px", 
            "fontWeight": "bold", 
            "fontFamily": "DejaVu Sans, Arial, sans-serif",
            "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
        }),
        multiple=False
    ),

    html.Div([
        html.Div([
            dcc.Graph(id='dos-plot', style={"border": "1px solid #ccc", "borderRadius": "5px", "padding": "10px"}),
        ], style={'width': '60%', 'display': 'inline-block', 'padding': '20px', 'backgroundColor': '#F9F9F9', 'borderRadius': '10px', 'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'}),

        html.Div([
            html.H3("Select atomic contributions and colors", style={
                "fontFamily": "DejaVu Sans, Arial, sans-serif", 
                "color": "#333", 
                "marginBottom": "10px",
                "textDecoration": "underline",
                "fontSize": "20px"
            }),
            html.P(
                "Use the checkboxes below to select specific orbitals for each atom type. "
                "You can also customize the colors for each atomic contribution.",
                style={"fontFamily": "DejaVu Sans, Arial, sans-serif", "fontSize": "18px", "marginBottom": "15px", "color": "#555"}
            ),
            html.Div(id='atomic-contributions-container', style={"marginBottom": "15px"}),
        ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#F9F9F9', 'borderRadius': '10px', 'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)', 'marginLeft': '20px'}),
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'padding': '20px'}),

    html.Div([
        html.H3("Graph adjustments", style={
            "fontFamily": "DejaVu Sans, Arial, sans-serif", 
            "color": "#333", 
            "marginBottom": "10px",
            "textDecoration": "underline",
            "fontSize": "20px"
        }),
        html.Div([
            html.Div([
                html.Label("X-axis limits:", style={"fontWeight": "bold", "fontFamily": "DejaVu Sans, Arial, sans-serif", "marginBottom": "5px"}),
                dcc.Input(id='xmin', type='number', value=DEFAULTS["xmin"], style={
                    "marginRight": "10px", "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif", "width": "80px"
                }),
                dcc.Input(id='xmax', type='number', value=DEFAULTS["xmax"], style={
                    "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif", "width": "80px"
                }),
            ], style={"marginBottom": "15px"}),

            html.Div([
                html.Label("Y-axis limits:", style={"fontWeight": "bold", "fontFamily": "DejaVu Sans, Arial, sans-serif", "marginBottom": "5px"}),
                dcc.Input(id='ymin', type='number', value=DEFAULTS["ymin"], style={
                    "marginRight": "10px", "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif", "width": "80px"
                }),
                dcc.Input(id='ymax', type='number', value=DEFAULTS["ymax"], style={
                    "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif", "width": "80px"
                }),
            ], style={"marginBottom": "15px"}),

            html.Div([
                html.Label("Legend Y-position:", style={"fontWeight": "bold", "fontFamily": "DejaVu Sans, Arial, sans-serif", "marginBottom": "5px"}),
                dcc.Input(id='legend-y', type='number', min=0, max=1, step=0.01, value=DEFAULTS["legend_y"], style={
                    "width": "80px", "padding": "8px", "borderRadius": "5px", "border": "1px solid #ccc",
                    "fontFamily": "DejaVu Sans, Arial, sans-serif"
                }),
            ], style={"marginBottom": "15px"}),

            # Add this new Div for the checklist
        ], style={'display': 'flex', 'alignItems': 'flex-end', 'gap': '40px'}),  # Add flex and gap

        html.Div([
            dcc.Checklist(
                id='show-titles',
                options=[
                    {'label': 'Plot title', 'value': 'plot_title'},
                    {'label': 'X axis title', 'value': 'x_title'},
                    {'label': 'Y axis title', 'value': 'y_title'},
                ],
                value=['plot_title', 'x_title', 'y_title'],  # All checked by default
                inline=True,
                style={"fontFamily": "DejaVu Sans, Arial, sans-serif", "marginLeft": "10px"}
            ),
        ], style={"marginBottom": "15px", "marginLeft": "auto"}),  # Push to right

        html.Div([
            dcc.Checklist(
                id='show-axis-scale',
                options=[
                    {'label': 'Show X axis scale', 'value': 'x_scale'},
                    {'label': 'Show Y axis scale', 'value': 'y_scale'},
                ],
                value=['x_scale', 'y_scale'],  # Both shown by default
                inline=True,
                style={"fontFamily": "DejaVu Sans, Arial, sans-serif", "marginLeft": "10px"}
            ),
        ], style={"marginBottom": "15px", "marginLeft": "auto"}),

        html.Div([
            html.Button("Reset Axes", id="reset-axes", n_clicks=0, style={
                "backgroundColor": "#FF5733", 
                "color": "white", 
                "padding": "10px 20px", 
                "border": "none", 
                "borderRadius": "5px", 
                "cursor": "pointer",
                "fontSize": "16px", 
                "fontWeight": "bold", 
                "fontFamily": "DejaVu Sans, Arial, sans-serif",
                "marginRight": "10px",
                "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
            }),
            html.Button("Demo File", id="demo-file", n_clicks=0, style={
                "backgroundColor": "#28a745",
                "color": "white",
                "padding": "10px 20px",
                "border": "none",
                "borderRadius": "5px",
                "cursor": "pointer",
                "fontSize": "16px",
                "fontWeight": "bold",
                "fontFamily": "DejaVu Sans, Arial, sans-serif",
                "marginRight": "10px",
                "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
            }),
            html.Button("Save Plot", id="save-plot", n_clicks=0, style={
                "backgroundColor": "#007BFF", 
                "color": "white", 
                "padding": "10px 20px", 
                "border": "none", 
                "borderRadius": "5px", 
                "cursor": "pointer",
                "fontSize": "16px", 
                "fontWeight": "bold", 
                "fontFamily": "DejaVu Sans, Arial, sans-serif",
                "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
            }),
        ], style={"marginTop": "15px"}),

        # Add spin-polarization and orbital messages
        html.Div(id="spin-message", style={
            "marginTop": "10px", 
            "color": "blue", 
            "fontSize": "16px", 
            "fontFamily": "DejaVu Sans, Arial, sans-serif"
        }),

        html.Div(id="save-confirmation", style={
            "marginTop": "10px", 
            "color": "#4CAF50", 
            "fontFamily": "DejaVu Sans, Arial, sans-serif"
        }),
    ], style={'width': '95%', 'margin': '20px auto', 'padding': '20px', 'backgroundColor': '#F9F9F9', 'borderRadius': '10px', 'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'}),

    # Add missing components
    dcc.Store(id='uploaded-contents'),
    dcc.Store(id='atom-defaults'),
    dcc.Store(id='spin-polarization'),
    dcc.Store(id='action-message-store'),
    dcc.Store(id='legend-order', data=[]),
    html.Div(id='debug-message', style={"display": "none"}),
    html.Div(id='folder-name', style={"display": "none"}),
    dcc.Download(id='download-plot'),
    html.Div(id='spin-polarization-status', style={
        "marginTop": "10px", 
        "color": "#333", 
        "fontFamily": "DejaVu Sans, Arial, sans-serif"
    }),
])

@app.callback(
    Output('uploaded-contents', 'data'),
    Output('folder-name', 'children'),
    Output('atom-defaults', 'data'),
    Input('upload-data', 'contents')
)
def store_uploaded_file(contents):
    if contents is None:
        return None, "", {}

    try:
        # Decode the uploaded content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Define paths for the ZIP file and extraction folder
        zip_path = os.path.join(os.path.dirname(__file__), 'uploaded_folder.zip')
        extracted_folder = os.path.join(os.path.dirname(__file__), 'uploaded_folder')

        # Clear the extracted folder if it exists
        if os.path.exists(extracted_folder):
            shutil.rmtree(extracted_folder)

        # Save the uploaded ZIP file
        with open(zip_path, 'wb') as f:
            f.write(decoded)

        # Validate and extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Check if the ZIP file is valid
            bad_file = zip_ref.testzip()
            if bad_file:
                return None, "", f"Error: Corrupted file '{bad_file}' found in the ZIP archive."
            zip_ref.extractall(extracted_folder)

        # Identify DOSCAR and POSCAR files in the extracted folder
        doscar_path, poscar_path = None, None
        for root, dirs, files in os.walk(extracted_folder):
            for file in files:
                if file == 'DOSCAR':
                    doscar_path = os.path.join(root, file)
                elif file == 'POSCAR':
                    poscar_path = os.path.join(root, file)
            if doscar_path and poscar_path:
                break

        # Check if both DOSCAR and POSCAR files are found
        if not doscar_path or not poscar_path:
            return None, "", "Error: POSCAR or DOSCAR file not found."

        # Read atom types from POSCAR
        with open(poscar_path, 'r') as f:
            poscar_lines = f.readlines()
        atom_types = poscar_lines[5].split()
        default_colors = ['blue', 'red', 'green', 'gray', 'black', 'orange', 'purple', 'pink', 'silver']
        color_map = {atom: default_colors[i % len(default_colors)] for i, atom in enumerate(atom_types)}

        # Extract folder name for display
        folder_name = os.path.basename(os.path.dirname(doscar_path))

        return {"DOSCAR": doscar_path, "POSCAR": poscar_path}, folder_name, color_map

    except zipfile.BadZipFile:
        return None, "", "Error: Uploaded file is not a valid ZIP file."
    except Exception as e:
        return None, "", f"Error: An unexpected error occurred. {str(e)}"

@app.callback(
    Output('dos-plot', 'figure'),
    Output('xmax', 'value'),
    Output('action-message-store', 'data'),
    Input('uploaded-contents', 'data'),
    Input('xmin', 'value'),
    Input('xmax', 'value'),
    Input('ymin', 'value'),
    Input('ymax', 'value'),
    Input('legend-y', 'value'),
    Input({'type': 'atom-checkbox', 'index': ALL}, 'value'),
    Input({'type': 'atom-checkbox', 'index': ALL}, 'id'),
    Input({'type': 'color-dropdown', 'index': ALL}, 'value'),
    Input({'type': 'color-dropdown', 'index': ALL}, 'id'),
    Input({'type': 'toggle-total', 'index': ALL}, 'value'),
    Input({'type': 'toggle-total', 'index': ALL}, 'id'),
    Input('spin-polarization', 'data'),
    Input('show-titles', 'value'),
    Input('show-axis-scale', 'value'),
    Input('legend-order', 'data'),
    State('dos-plot', 'figure')
)
def update_graph(
    contents, xmin, xmax, ymin, ymax, legend_y,
    selected_orbitals, atom_ids, selected_colors, color_ids,
    toggled_totals, toggle_ids, spin_polarized, show_titles, show_axis_scale, legend_order, current_figure
):

    if not contents or not isinstance(contents, dict) or 'POSCAR' not in contents or 'DOSCAR' not in contents:
        return {}, None, "Error: POSCAR or DOSCAR file not found."

    poscar_path = contents['POSCAR']
    doscar_path = contents['DOSCAR']

    # Extract required data from the DOSCAR file
    with open(doscar_path, 'r') as f:
        lines = f.readlines()

    fermi_energy = float(lines[5].split()[3])
    num_points = int(lines[5].split()[2])

    # Calculate energy and atom-summed total DOS
    energy = np.array([float(line.split()[0]) - fermi_energy for line in lines[6:6 + num_points]])
    atom_dos_blocks = []
    current_line = 6 + num_points
    num_atoms = int(lines[0].split()[0])

    for _ in range(num_atoms):
        current_line += 1
        block = np.array([
            [float(values[0]) - fermi_energy] + [float(v) for v in values[1:]]
            for values in (lines[current_line + i].split() for i in range(num_points))
        ])
        atom_dos_blocks.append(block)
        current_line += num_points

    total_dos = np.zeros(num_points)
    for block in atom_dos_blocks:
        total_dos += np.sum(block[:, 1:], axis=1)  # Sum all orbitals for each atom

    # Adjust the range calculation using the new atom-summed total DOS
    energy_range_mask = (energy >= (ymin if ymin is not None else -8)) & (energy <= (ymax if ymax is not None else 2))
    dos_in_range = total_dos[energy_range_mask]
    calculated_xmax = math.ceil(1.1 * np.max(dos_in_range)) if len(dos_in_range) > 0 else DEFAULTS["xmax"]

    # Use the calculated xmax if xmax is None (e.g., on app start or file upload), otherwise respect the user's input
    xmax_to_use = calculated_xmax if xmax is None else xmax

    # Ensure custom_colors is initialized
    custom_colors = {color_id['index']: color for color_id, color in zip(color_ids, selected_colors)} if selected_colors else {}

    # Parse the selected atoms and orbitals
    selected_atoms = {atom_id['index']: orbitals for atom_id, orbitals in zip(atom_ids, selected_orbitals) if orbitals}

    # Parse the toggled totals
    toggled_atoms = {toggle_id['index']: 'total' in toggled for toggle_id, toggled in zip(toggle_ids, toggled_totals)}

    # If no toggled totals are provided (e.g., on file load), default to showing all atomic totals
    if not toggled_totals:
        toggled_atoms = {atom_id['index']: True for atom_id in atom_ids}

    # Update the plot based on selected atoms, orbitals, toggled totals
    fig = parse_doscar_and_plot(
        doscar_path, poscar_path, xmin, xmax_to_use, ymin, ymax, legend_y, custom_colors, plot_type="total",
        spin_polarized=spin_polarized, selected_atoms=selected_atoms, toggled_atoms=toggled_atoms, show_titles=show_titles, show_axis_scale=show_axis_scale, legend_order=legend_order
    )

    # Return the updated plot and the calculated xmax only if it was used
    return fig, calculated_xmax if xmax is None else xmax, ""

@app.callback(
    Output('download-plot', 'data'),
    Output('save-confirmation', 'children'),
    Input('save-plot', 'n_clicks'),
    State('dos-plot', 'figure'),
    State('folder-name', 'children'),
    prevent_initial_call=True
)
def save_plot(n_clicks, figure, folder_name):
    if n_clicks:
        fig = pio.from_json(pio.to_json(figure))
        filename = f"{folder_name}_dos_plot.png"

        def write_image_to_bytesio(output_buffer):
            fig.write_image(output_buffer, format="png", scale=4)  # Set scale to 4 for higher DPI (~400 DPI)

        return dcc.send_bytes(write_image_to_bytesio, filename), html.Span(
            f"Plot downloaded as '{filename}'!",
            style={"fontWeight": "bold", "fontSize": "18px"}
        )
    return dash.no_update, ""

from dash.exceptions import PreventUpdate

@app.callback(
    Output({'type': 'atom-checkbox', 'index': ALL}, 'value', allow_duplicate=True),
    Output({'type': 'toggle-total', 'index': ALL}, 'value', allow_duplicate=True),
    Input('uploaded-contents', 'data'),
    State('atom-defaults', 'data'),
    prevent_initial_call=True
)
def select_atom_totals_on_file_load(contents, atom_defaults):
    if not contents or not isinstance(contents, dict) or 'POSCAR' not in contents or 'DOSCAR' not in contents:
        raise PreventUpdate

    atom_keys = list(atom_defaults.keys())
    orbital_values = [[] for _ in atom_keys]  # No orbitals selected by default
    total_values = [['total'] for _ in atom_keys]  # All totals toggled by default

    return orbital_values, total_values

@app.callback(
    Output('atomic-contributions-container', 'children', allow_duplicate=True),
    Output('action-message-store', 'data', allow_duplicate=True),
    Output('legend-order', 'data', allow_duplicate=True),
    Input('uploaded-contents', 'data'),
    State('atom-defaults', 'data'),
    State('spin-polarization', 'data'),
    prevent_initial_call=True
)
def handle_atomic_contributions_and_debug(contents, atom_defaults, spin_polarized):
    if not contents or not isinstance(contents, dict) or 'DOSCAR' not in contents:
        return [], "Error: POSCAR or DOSCAR file not found.", []

    doscar_path = contents['DOSCAR']

    try:
        with open(doscar_path, 'r') as f:
            lines = f.readlines()

        # Extract the number of atoms and number of points
        num_atoms = int(lines[0].split()[0])
        num_points = int(lines[5].split()[2])

        # Parse atom blocks to determine the number of columns
        atom_dos_blocks = []
        current_line = 6 + num_points
        for _ in range(num_atoms):
            current_line += 1
            block = np.array([
                [float(values[0])] + [float(v) for v in values[1:]]
                for values in (lines[current_line + i].split() for i in range(num_points))
            ])
            atom_dos_blocks.append(block)
            current_line += num_points

        # Determine the number of columns in the atom blocks
        num_columns = atom_dos_blocks[0].shape[1]
        debug_message = f"Detected {num_columns} columns in DOSCAR."

        # Dynamically set orbital labels based on the number of columns
        if num_columns == 10:
            orbital_labels = [
                "s", 
                "py", 
                "pz", 
                "px", 
                "dxy", 
                "dyz", 
                "dz²", 
                "dxz", 
                "dx²-y²"
            ]
        elif num_columns == 7:
            orbital_labels = ["s ↑", "s ↓", "p ↑", "p ↓", "d ↑", "d ↓"]
        elif num_columns == 19:
            orbital_labels = [
                "s ↑", "s ↓", "py ↑", "py ↓", "pz ↑", "pz ↓", "px ↑", "px ↓",
                "dxy ↑", "dxy ↓", "dyz ↑", "dyz ↓", "dz² ↑", "dz² ↓", "dxz ↑", "dxz ↓", "dx²-y² ↑", "dx²-y² ↓"
            ]
        elif num_columns == 17:
            orbital_labels = [
                "s",
                "py", 
                "pz", 
                "px", 
                "dxy", 
                "dyz", 
                "dz²", 
                "dxz", 
                "dx²-y²",
                "fx(3x²-y²)",
                "fxyz",
                "fyz²",
                "fz³",
                "fxz²",
                "fx(x²-y²)",
                "fx(x²-3y²)"]
        elif num_columns == 5:
            # Check for f-block elements in POSCAR
            poscar_path = contents['POSCAR'] if isinstance(contents, dict) and 'POSCAR' in contents else None
            atom_types = []
            if poscar_path and os.path.exists(poscar_path):
                with open(poscar_path, 'r') as f:
                    poscar_lines = f.readlines()
                atom_types = poscar_lines[5].split()
            f_elements = [
                "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
            ]
            contains_f_element = any(elem in atom_types for elem in f_elements)
            if contains_f_element:
                orbital_labels = ["s", "p", "d", "f"]
            else:
                orbital_labels = ["s", "py", "pz", "px"]  
        elif num_columns == 4:
            orbital_labels = ["s", "p", "d"]
        elif num_columns == 33:
            orbital_labels = [
            "s ↑", "s ↓", "py ↑", "py ↓", "pz ↑", "pz ↓", "px ↑", "px ↓",
            "dxy ↑", "dxy ↓", "dyz ↑", "dyz ↓", "dz² ↑", "dz² ↓",
            "dxz ↑", "dxz ↓",
            "dx²-y² ↑", "dx²-y² ↓",
            'fx(3x²-y²) ↑', 'fx(3x²-y²) ↓',
            'fxyz ↑', 'fxyz ↓',
            'fyz² ↑', 'fz³ ↑', 'fz³ ↓',
            'fxz² ↑', 'fxz² ↓',
            'fx(x²-y²) ↑', 'fx(x²-y²) ↓',
            'fx(x²-3y²) ↑', 'fx(x²-3y²) ↓'
        ]
        elif num_columns == 9:
            orbital_labels = ['s ↑', 's ↓', 'p ↑', 'p ↓', 'd ↑', 'd ↓', 'f ↑', 'f ↓']

        # Create a table-like layout for atom and orbital selection
        table_header = html.Tr([
            html.Th("Atom", style={"textAlign": "center", "padding": "10px", "fontWeight": "bold"}),
            html.Th("Orbitals", style={"textAlign": "center", "padding": "10px", "fontWeight": "bold"}),
            html.Th("Color", style={"textAlign": "center", "padding": "10px", "fontWeight": "bold"}),
            html.Th("Toggle Total", style={"textAlign": "center", "padding": "10px", "fontWeight": "bold"}),
            html.Th("Order", style={"textAlign": "center", "padding": "10px", "fontWeight": "bold"})
        ], style={"backgroundColor": "#f2f2f2"})

        table_rows = []

        table_rows.append(html.Tr([
            html.Td("Total", style={"textAlign": "center", "padding": "10px", "fontFamily": "DejaVu Sans, Arial, sans-serif", "fontWeight": "bold"}),
            html.Td("", style={"textAlign": "center", "padding": "10px"}),  # No orbitals for Total
            html.Td(
                dcc.Dropdown(
                    id={'type': 'color-dropdown', 'index': 'Total'},
                    options=[{
                        "label": html.Span([
                            html.Div(style={
                                "backgroundColor": c, "width": "15px", "height": "15px",
                                "display": "inline-block", "marginRight": "8px"
                            }), c
                        ], style={"display": "flex", "alignItems": "center", "fontFamily": "DejaVu Sans, Arial, sans-serif"}), "value": c} for c in
                        ['black', 'blue', 'red', 'green', 'gray', 'orange', 'purple', 'pink', 'silver']],
                    value='black',
                    clearable=False,
                    style={"width": "100px", "fontFamily": "DejaVu Sans, Arial, sans-serif"}
                ),
                style={"textAlign": "center", "padding": "10px"}
            ),
            html.Td(
                dcc.Checklist(
                    id={'type': 'toggle-total', 'index': 'Total'},
                    options=[{'label': '', 'value': 'total'}],
                    value=['total'],  # Default to showing total
                    inline=True,
                    style={"fontFamily": "DejaVu Sans, Arial, sans-serif"}
                ),
                style={"textAlign": "center", "padding": "10px"}
            ),
            html.Td(
                html.Div([
                    html.Button("↑", id={'type': 'order-up', 'index': 'Total'}, n_clicks=0, style={"marginRight": "5px"}),
                    html.Button("↓", id={'type': 'order-down', 'index': 'Total'}, n_clicks=0)
                ], style={"display": "flex", "justifyContent": "center"})
            )
        ], style={"borderBottom": "1px solid #ddd"}))

        atom_types = list(atom_defaults.keys())
        legend_order = ['Total'] + atom_types
        for atom in atom_types:
            table_rows.append(html.Tr([
                html.Td(atom, style={"textAlign": "center", "padding": "10px", "fontFamily": "DejaVu Sans, Arial, sans-serif"}),
                html.Td(
                    dcc.Checklist(
                        id={'type': 'atom-checkbox', 'index': atom},
                        options=[{'label': orbital, 'value': orbital} for orbital in orbital_labels],
                        value=[],  # Default to no selection
                        inline=True,
                        style={"fontFamily": "DejaVu Sans, Arial, sans-serif"}
                    ),
                    style={"textAlign": "center", "padding": "10px"}
                ),
                html.Td(
                    dcc.Dropdown(
                        id={'type': 'color-dropdown', 'index': atom},
                        options=[{
                            "label": html.Span([
                                html.Div(style={
                                    "backgroundColor": c, "width": "15px", "height": "15px",
                                    "display": "inline-block", "marginRight": "8px"
                                }), c
                            ], style={"display": "flex", "alignItems": "center", "fontFamily": "DejaVu Sans, Arial, sans-serif"}), "value": c} for c in
                            ['blue', 'red', 'green', 'gray', 'black', 'orange', 'purple', 'pink', 'silver']],
                        value=atom_defaults.get(atom, 'blue'),
                        clearable=False,
                        style={"width": "100px", "fontFamily": "DejaVu Sans, Arial, sans-serif"}
                    ),
                    style={"textAlign": "center", "padding": "10px"}
                ),
                html.Td(
                    dcc.Checklist(
                        id={'type': 'toggle-total', 'index': atom},
                        options=[{'label': '', 'value': 'total'}],
                        value=['total'],  # Default to showing total contribution
                        inline=True,
                        style={"fontFamily": "DejaVu Sans, Arial, sans-serif"}
                    ),
                    style={"textAlign": "center", "padding": "10px"}
                ),
                html.Td(
                    html.Div([
                        html.Button("↑", id={'type': 'order-up', 'index': atom}, n_clicks=0, style={"marginRight": "5px"}),
                        html.Button("↓", id={'type': 'order-down', 'index': atom}, n_clicks=0)
                    ], style={"display": "flex", "justifyContent": "center"})
                )
            ], style={"borderBottom": "1px solid #ddd"}))

        table = html.Table(
            [table_header] + table_rows,
            style={
                "width": "100%", 
                "borderCollapse": "collapse", 
                "marginTop": "20px", 
                "fontFamily": "DejaVu Sans, Arial, sans-serif"
            }
        )

        return table, debug_message, legend_order
    except Exception as e:
        return [], f"Error processing DOSCAR file: {str(e)}", []

@app.callback(
    Output('debug-message', 'children'),
    Input('action-message-store', 'data')
)
def display_debug_message(debug_message):
    return debug_message

@app.callback(
    [Output('xmin', 'value', allow_duplicate=True),
     Output('xmax', 'value', allow_duplicate=True),
     Output('ymin', 'value', allow_duplicate=True),
     Output('ymax', 'value', allow_duplicate=True)],
    [Input('reset-axes', 'n_clicks')],
    [State('xmin', 'value'),
     State('xmax', 'value'),
     State('ymin', 'value'),
     State('ymax', 'value'),
     State('uploaded-contents', 'data')],
    prevent_initial_call=True
)
def handle_axes_and_update(reset_clicks, xmin, xmax, ymin, ymax, contents):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'reset-axes':
        if not contents or not isinstance(contents, dict) or 'DOSCAR' not in contents:
            return DEFAULTS["xmin"], DEFAULTS["xmax"], DEFAULTS["ymin"], DEFAULTS["ymax"]

        doscar_path = contents['DOSCAR']

        # Calculate the buffer value for xmax based on the DOSCAR file
        try:
            with open(doscar_path, 'r') as f:
                lines = f.readlines()

            fermi_energy = float(lines[5].split()[3])
            num_points = int(lines[5].split()[2])
            energy = np.array([float(line.split()[0]) - fermi_energy for line in lines[6:6 + num_points]])
            total_dos = np.zeros(num_points)

            current_line = 6 + num_points
            num_atoms = int(lines[0].split()[0])
            for _ in range(num_atoms):
                current_line += 1
                block = np.array([
                    [float(values[0]) - fermi_energy] + [float(v) for v in values[1:]]
                    for values in (lines[current_line + i].split() for i in range(num_points))
                ])
                total_dos += np.sum(block[:, 1:], axis=1)
                current_line += num_points

            energy_range_mask = (energy >= DEFAULTS["ymin"]) & (energy <= DEFAULTS["ymax"])
            dos_in_range = total_dos[energy_range_mask]
            calculated_xmax = math.ceil(1.1 * np.max(dos_in_range)) if len(dos_in_range) > 0 else DEFAULTS["xmax"]

            return DEFAULTS["xmin"], calculated_xmax, DEFAULTS["ymin"], DEFAULTS["ymax"]

        except Exception:
            # Fallback to defaults if an error occurs
            return DEFAULTS["xmin"], DEFAULTS["xmax"], DEFAULTS["ymin"], DEFAULTS["ymax"]

    raise PreventUpdate

@app.callback(
    Output('spin-message', 'children'),
    Input('uploaded-contents', 'data')
)
def update_spin_message(contents):
    if not contents or not isinstance(contents, dict) or 'DOSCAR' not in contents:
        return html.Span("Error: POSCAR or DOSCAR file not found.", style={"fontWeight": "bold", "fontSize": "18px"})

    doscar_path = contents['DOSCAR']

    try:
        with open(doscar_path, 'r') as f:
            lines = f.readlines()

        # Check the first block for spin-polarization
        num_columns_first_block = len(lines[6].split())
        if num_columns_first_block == 5:
            spin_message = "Spin-polarized collinear calculation detected (ISPIN=2). This means the calculation includes spin-up and spin-down states."
        elif num_columns_first_block == 3:
            spin_message = "Non-spin-polarized calculation detected (ISPIN=1, or not entered as it is the default). This means the calculation does not include spin-up and spin-down states."
        else:
            spin_message = "Unknown DOSCAR format detected. Unable to determine spin polarization."

        # Check the second block for lm-resolved orbital calculations
        num_points = int(lines[5].split()[2])
        second_block_start = 6 + num_points + 1
        num_columns = len(lines[second_block_start].split())  # Use the second block to determine num_columns

        if num_columns == 10:
            orbital_message = "lm-resolved calculation detected (LORBIT=11, 12, 13, or 14 (VASP>6)). Orbitals are resolved into individual components: px, py, pz, etc."
        elif num_columns == 19:
            orbital_message = "lm-resolved calculation detected (LORBIT=11, 12, 13, or 14 (VASP>6)). Orbitals are resolved into individual components: px, py, pz, etc. with respective spin states."
        elif num_columns == 7:
            orbital_message = "Spin-polarized collinear calculation with grouped orbitals detected (LORBIT=0, 1, 2, 5 or 10, ISPIN=2). Only total p- and d-orbital contributions are available (i.e., p = px + py + pz) with respective spin states."        
        elif num_columns == 5:
            poscar_path = contents['POSCAR'] if isinstance(contents, dict) and 'POSCAR' in contents else None
            atom_types = []
            if poscar_path and os.path.exists(poscar_path):
                with open(poscar_path, 'r') as f:
                    poscar_lines = f.readlines()
                atom_types = poscar_lines[5].split()
            f_elements = [
                "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
            ]
            contains_f_element = any(elem in atom_types for elem in f_elements)
            if contains_f_element:
                orbital_message = "f-block element detected with 5 columns: s, p, d, and f contributions are available (grouped)."
            else:
                orbital_message = "Regular grouped atomic contribution detected (LORBIT=0, 1, 2, 5 or 10). s and total p-orbital contributions are available (i.e., p = px + py + pz)."
        elif num_columns == 4:
            orbital_message = "Regular grouped atomic contribution detected (LORBIT=0, 1, 2, 5 or 10). Only total p- and d-orbital contributions are available (i.e., p = px + py + pz)."
        elif num_columns == 17:
            orbital_message = "lm-resolved calculation detected (LORBIT=11, 12, 13, or 14 (VASP>6)). Orbitals are resolved into individual components: px, py, pz, etc. up to the f orbital"
        elif num_columns == 33:
            orbital_message = "lm-resolved calculation detected (LORBIT=11, 12, 13, or 14 (VASP>6)). Orbitals are resolved into individual components: px, py, pz, etc. up to the f orbital with respective spin states."
        elif num_columns == 9:
            orbital_message = "Regular grouped atomic contribution detected (LORBIT=0, 1, 2, 5 or 10). s and total p-, d- and f-orbital contributions are available (i.e., p = px + py + pz) with respective spin states."
        else:
            orbital_message = "Unknown orbital format detected."

        spin_message += f" {orbital_message}"
        return html.Span(spin_message, style={"fontWeight": "bold", "fontSize": "18px"})

    except Exception as e:
        return html.Span(f"Error processing DOSCAR file: {str(e)}", style={"fontWeight": "bold", "fontSize": "18px"})

@app.callback(
    Output('upload-data', 'contents', allow_duplicate=True),
    Input('demo-file', 'n_clicks'),
    prevent_initial_call=True
)
def load_demo_file(n_clicks):
    if n_clicks:
        demo_path = os.path.join(os.path.dirname(__file__), "ISPIN2_LORBIT14.zip")
        if not os.path.exists(demo_path):
            return dash.no_update
        with open(demo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        mime_type = mimetypes.guess_type(demo_path)[0] or "application/zip"
        contents = f"data:{mime_type};base64,{encoded}"
        return contents
    return dash.no_update

@app.callback(
    Output('legend-order', 'data', allow_duplicate=True),
    [Input({'type': 'order-up', 'index': ALL}, 'n_clicks'),
     Input({'type': 'order-down', 'index': ALL}, 'n_clicks')],
    [State('legend-order', 'data'),
     State('atomic-contributions-container', 'children')],
    prevent_initial_call=True
)
def update_legend_order(up_clicks, down_clicks, legend_order, table_children):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    trigger = ctx.triggered[0]['prop_id']
    # Extract the atom name or 'Total' from the trigger string
    import re
    match = re.search(r'index":\s*"([^"]+)"', trigger)
    if not match:
        return legend_order
    name = match.group(1)
    if name not in legend_order:
        return legend_order
    idx = legend_order.index(name)
    if 'order-up' in trigger and idx > 0:
        legend_order[idx-1], legend_order[idx] = legend_order[idx], legend_order[idx-1]
    elif 'order-down' in trigger and idx < len(legend_order)-1:
        legend_order[idx+1], legend_order[idx] = legend_order[idx], legend_order[idx+1]
    return legend_order

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))  # Use the PORT environment variable set by Heroku
    app.run_server(debug=False, host='0.0.0.0', port=port)  # Set debug=False for production
from dash import Dash, html, dcc, Output, Input, callback, Patch, clientside_callback, State
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import json
from utils import activation, discriminability, typicality, map_delta_freq


HEIGHT = 540
THEME = 'bootstrap'


data = pd.read_csv('data.csv')


def get_template(switch_on):
    template = pio.templates[THEME] if switch_on else pio.templates[THEME + '_dark']
    return template


# Themes and templates
load_figure_template([THEME, THEME + '_dark'])
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
bootstrap_css = eval('dbc.themes.' + THEME.upper())

# Create application
app = Dash(__name__, external_stylesheets=[
    bootstrap_css,
    dbc_css,
    dbc.icons.BOOTSTRAP
])


# Plots
activation_plot = dcc.Graph(
    id='activation_plot', clickData={'points': [{'x': 0}]})
activation_barplot = dcc.Graph(id='activation_barplot')
discr_plot = dcc.Graph(id='discr_plot')
typ_plot = dcc.Graph(id='typ_plot')

# Controls
color_mode_switch = html.Span(
    [
        dbc.Label(className="bi bi-moon-fill m-2",
                  html_for="color-mode-switch"),
        dbc.Switch(id="color-mode-switch", value=False,
                   className="d-inline-block ms-1", persistence=True),
        dbc.Label(className="bi bi-sun-fill m-2",
                  html_for="color-mode-switch"),
    ]
)

alpha_input = dbc.Input(id='alpha_input', type='number',
                        min=0.1, max=0.6, step=0.1, value=0.3)
delta_input = dbc.Input(id='delta_input', type='number',
                        min=0.25, max=1, step=.05, value=.5)
tau_input = dbc.Input(id='tau_input', type='number',
                      min=0.1, max=0.3, step=0.1, value=.1)
freq_input = dbc.Input(id='freq_input', type='number',
                       min=1, max=12, step=1, value=6)

param_controls = dbc.Col([
    color_mode_switch,
    html.H5('Model parameters'),

    html.Div([
        dbc.Label('activation window size (α)', html_for='alpha_input'),
        alpha_input,
    ], className='mb-2'),

    html.Div([
        dbc.Label('discriminability threshold (δ)', html_for='delta_input'),
        delta_input,
    ], className='mb-2'),

    html.Div([
        dbc.Label('typicality threshold (τ)', html_for='tau_input'),
        tau_input,
    ], className='mb-2'),

    html.Hr(),

    dbc.Switch(id='freq_switch', value=False,
               persistence=False, label='vary δ with type frequency'),

    html.Div([
        dbc.Label('type frequency', html_for='freq_input'),
        freq_input,
    ], className='mb-2', hidden=True, id='freq_group'),


], width=3)


# Application layout

app.layout = dbc.Container([
    dcc.Store(id='application_data'),


    dbc.Row([
        param_controls,

        dbc.Col([
            activation_plot
        ], width=6),
        dbc.Col([
            activation_barplot,
        ], width=3, align='end'),

    ]),
    dbc.Row([
        dbc.Col([
            discr_plot
        ], width=6),
        dbc.Col([
            typ_plot,
        ], width=6)
    ]),

    html.Footer(
        html.A("https://github.com/g-fabiani/exemplar_model_dashboard",
               href="https://github.com/g-fabiani/exemplar_model_dashboard")
    )

], fluid=True, className='dbc')


# Callbacks

@callback(
    Output('activation_plot', 'figure', allow_duplicate=True),
    Output('activation_barplot', 'figure', allow_duplicate=True),
    Output('discr_plot', 'figure', allow_duplicate=True),
    Output('typ_plot', 'figure', allow_duplicate=True),
    Input('color-mode-switch', 'value'),
    prevent_initial_call=True
)
def update_figure_template(switch_on):
    """
    Update template of all figures when the color mode switch
    is switched w/o having to re-draw the figures from scratch.
    Makes the switching between dark and light mode faster.
    """

    patched_figure = Patch()
    patched_figure['layout']['template'] = get_template(switch_on)

    return patched_figure, patched_figure, patched_figure, patched_figure


clientside_callback(
    """
    (switchOn) => {
       document.documentElement.setAttribute('data-bs-theme', switchOn ? 'light' : 'dark');
       return window.dash_clientside.no_update
    }
    """,
    Output("color-mode-switch", "id"),
    Input("color-mode-switch", "value"),
)


@callback(
    Output('activation_plot', 'figure'),
    Input('alpha_input', 'value'),
    Input('activation_plot', 'clickData'),
    Input('color-mode-switch', 'value')
)
def update_activation_window_plot(alpha, clickdata, switch_on):
    """
    Update activation window plot and exemplar space plot when
    a new alpha value or a new token is set.
    """

    x = np.linspace(-3.5, 1.7, 105)

    point = clickdata['points'][0]['x']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=[
        'Activation window', 'Exemplar space'])

    # Create traces for each category
    for category in data['category'].unique():
        category_data = data[data['category'] == category]
        fig.add_trace(go.Box(x=category_data['value'],
                             boxpoints='all',
                             jitter=1,
                             hoveron='points',
                             pointpos=0,
                             showlegend=False,
                             name=category,
                             line=dict(width=0),
                             fillcolor='rgba(255,255,255,0)',
                             marker=dict(opacity=0.5),
                             hovertext=category_data['value'],
                             hoverinfo='text'
                             ),
                      row=2, col=1)

    # Create trace for activation function
    fig.add_trace(
        go.Scatter(x=x, y=activation(x, point, alpha),
                   mode='lines', showlegend=False, name='activation'), row=1, col=1
    )
    fig.update_layout(
        height=HEIGHT,
        uirevision='stay there',
        template=get_template(switch_on)
    )

    # Create trace for plotting token
    fig.add_trace(
        go.Box(x=[point],
               name="Token",
               showlegend=False,
               boxpoints='all',
               pointpos=0,
               hovertext=round(point, 2),
               hoverinfo='text',
               line=dict(width=0),
               marker=dict(size=14, symbol='star')
               ),
        row=2, col=1)

    return fig


@callback(
    Output('activation_barplot', 'figure'),
    Output('application_data', 'data'),
    Input('alpha_input', 'value'),
    Input('activation_plot', 'clickData'),
    Input('color-mode-switch', 'value')
)
def update_activation_barplot(alpha, clickdata, switch_on):
    """
    Update total activation barplot when a new activation
    window function is plotted.
    Store aggregated data in Store.
    """

    point = clickdata['points'][0]['x']

    data['activation'] = data['value'].transform(
        lambda x: activation(x, point, alpha))

    # Aggregate data
    data_agg = data.groupby('category')[
        'activation'].aggregate(['sum', 'count'])

    ratio = data_agg.loc['Pushee']['sum'] / data_agg.loc['Pusher']['sum']
    avg_activation = (data_agg['sum'] / data_agg['count'])['Pushee']

    intermediate_data = {'ratio': ratio, 'avg_activation': avg_activation}

    # Create figure
    fig = px.bar(data_agg, y='sum',  color=data_agg.index, range_y=(0, 350))
    fig.update_yaxes(title='Total activation')
    fig.update_layout(
        height=HEIGHT,
        showlegend=False,
        template=get_template(switch_on)
    )

    return fig, json.dumps(intermediate_data)


@callback(
    Output('discr_plot', 'figure'),
    Input('delta_input', 'value'),
    Input('application_data', 'data'),
    Input('color-mode-switch', 'value')
)
def update_discr_plot(delta, intermediate_data, switch_on):
    """
    Update discriminability plot.
    This does three things:
    - plot the discriminabilty curve depending on the discriminability threshold
    - plot a point on the curve with x corresponding to the ratio between
        the intended category activation and the residual activation
    - update the resulting probability in the title of the plot
    """

    ratio = json.loads(intermediate_data)['ratio']
    discr = discriminability(ratio, delta)

    xmax = 12 if ratio < 12 else ratio + ratio * .1

    x = np.linspace(0, xmax, 1000)

    fig = px.line(x=x, y=discriminability(x, delta),
                  range_y=(-.05, 1.05))
    fig.add_trace(go.Scatter(
        x=[ratio], y=[discr], marker=dict(size=12), hoverinfo='text',
        hovertext=f'ratio: {round(ratio, 3)}, p: {round(discr, 3)}'))

    fig.update_xaxes(title='activation ratio')
    fig.update_yaxes(title='')

    fig.update_layout(
        height=0.6 * HEIGHT,
        title=f'Probability of passing discriminability evaluation: <b>{
            round(discr, 3)}</b>',
        showlegend=False,
        uirevision='stay there',
        template=get_template(switch_on)
    )

    return fig


@callback(
    Output('typ_plot', 'figure'),
    Input('tau_input', 'value'),
    Input('application_data', 'data'),
    Input('color-mode-switch', 'value')
)
def update_typ_plot(tau, intermediate_data, switch_on):
    """
    Update the typicality plot.
    This does three things:
    - plot the typicality curve depending on the typicality threshold
    - plot a point on the curve with x corresponding to the average activation
       of the intended category for the token
    - update the resulting probability in the title of the plot
    """

    avg_activation = json.loads(intermediate_data)['avg_activation']
    typ = typicality(avg_activation, tau)

    x = np.linspace(0, 1, 100)
    fig = px.line(x=x, y=typicality(x, tau), range_y=(-.05, 1.05))
    fig.add_trace(go.Scatter(x=[avg_activation],
                  y=[typ], marker=dict(size=12), hoverinfo='text',
                  hovertext=f'avg activation: {round(avg_activation, 3)}, p: {round(typ, 3)}'))
    fig.update_xaxes(title='average exemplar activation', range=(-.05, 1))
    fig.update_yaxes(title='')

    fig.update_layout(
        height=0.6*HEIGHT,
        title=f'Probability of passing typicality evaluation: <b>{
            round(typ, 3)}</b>',
        showlegend=False,
        uirevision='stay there',
        template=get_template(switch_on)
    )

    return fig


@callback(
    Output('freq_group', 'hidden'),
    Output('delta_input', 'value'),
    Output('delta_input', 'disabled'),
    Input('freq_switch', 'value'),
    Input('freq_input', 'value'),
    prevent_initial_call=True
)
def update_freq_group(switch_on, freq):

    delta = map_delta_freq[freq] if switch_on else .5

    return not switch_on, delta, switch_on


if __name__ == '__main__':
    app.run(debug=True)

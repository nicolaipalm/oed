from dash import Dash, dcc, html, Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def create_dashboard(does, does_benchmark):
    # Creating the figures
    fig_mean = does_benchmark.plot_estimated_mean(0)
    fig_CRLB = does_benchmark.plot_diagonal_entry_of_CRLB(0)
    fig_std = does_benchmark.plot_estimated_std(0)
    fig_normed_CRLB = does_benchmark.plot_normed_diagonal_entry_of_CRLB(0)

    fig_MLE_values = does_benchmark.plot_MLE_value(0)
    fig_dets_FI = does_benchmark.plot_dets_FI()

    app = Dash(external_stylesheets=external_stylesheets)

    # Creating the app
    app.layout = html.Div([
        html.Div([
            html.Div([
                html.Div([
                    dcc.RadioItems(
                        list(range(len(does.theta))),
                        0,
                        id='index',
                        labelStyle={'display': 'inline-block', 'marginTop': '5px'}
                    )
                ],
                    style={'width': '33%', 'display': 'inline-block'})
            ], style={
                'padding': '10px 5px'
            }),
            html.Div([
                dcc.Graph(
                    figure=fig_mean,
                    id='fig_mean',
                    hoverData={'points': [{'customdata': 'Japan'}]}
                )
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([
                dcc.Graph(
                    figure=fig_std,
                    id='fig_std',
                    hoverData={'points': [{'customdata': 'Japan'}]}
                )
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'})
        ]),
        html.Div([
            html.Div([
                dcc.Graph(
                    figure=fig_CRLB,
                    id='fig_CRLB',
                    hoverData={'points': [{'customdata': 'Japan'}]}
                )
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([
                dcc.Graph(
                    figure=fig_MLE_values,
                    id='fig_MLE_values',
                    hoverData={'points': [{'customdata': 'Japan'}]}
                )
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'})
        ]),
        html.Div([
            html.Div([
                dcc.Graph(
                    figure=fig_normed_CRLB,
                    id='fig_normed_CRLB',
                    hoverData={'points': [{'customdata': 'Japan'}]}
                )
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([
                dcc.Graph(
                    figure=fig_dets_FI,
                    id='fig_dets_FI',
                    hoverData={'points': [{'customdata': 'Japan'}]}
                )
            ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'})
        ])
    ])

    # interact
    @app.callback(
        Output('fig_mean', 'figure'),
        Input('index', 'value'))
    def update_graph(index):
        return does_benchmark.plot_estimated_mean(index)

    @app.callback(
        Output('fig_std', 'figure'),
        Input('index', 'value'))
    def update_graph(index):
        return does_benchmark.plot_estimated_std(index)

    @app.callback(
        Output('fig_normed_CRLB', 'figure'),
        Input('index', 'value'))
    def update_graph(index):
        return does_benchmark.plot_normed_diagonal_entry_of_CRLB(index)

    @app.callback(
        Output('fig_CRLB', 'figure'),
        Input('index', 'value'))
    def update_graph(index):
        return does_benchmark.plot_diagonal_entry_of_CRLB(index)

    @app.callback(
        Output('fig_MLE_values', 'figure'),
        Input('index', 'value'))
    def update_graph(index):
        return does_benchmark.plot_MLE_value(index)

    app.run_server(debug=True,use_reloader=False)

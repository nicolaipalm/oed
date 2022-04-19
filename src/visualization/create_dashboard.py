from typing import List

from dash import Dash, dcc, html

from src.benchmarking.benchmarking import Benchmarking
from src.metrics.interfaces.metric import Metric

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def create_dashboard(benchmarking: Benchmarking, metrics: List[Metric]):
    app = Dash(external_stylesheets=external_stylesheets)

    # Creating the app
    app.layout = html.Div([
        html.Div([
            dcc.Graph(
                figure=metric.plot(
                    evaluations_blackbox_function_for_each_design=benchmarking.evaluations_blackbox_function,
                    estimations_of_parameter_for_each_design=benchmarking.maximum_likelihood_estimations, ),
                id='fig_mean',
                hoverData={'points': [{'customdata': 'Japan'}]}
            )
        ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'})
        for metric in metrics
    ])

    app.run_server(debug=True, use_reloader=False)

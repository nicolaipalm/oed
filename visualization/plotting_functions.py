import numpy as np
import plotly.graph_objects as go


def update_layout_of_graph(fig: go.Figure, title: str = 'Plot') -> go.Figure:
    fig.update_layout(
        width=800,
        height=600,
        # autosize=False,
        plot_bgcolor='rgba(0,0,0,0)',
        title=title,

    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                      xaxis_title='input values',
                      yaxis_title='output values',
                      legend=dict(yanchor="top",
                                  y=0.9,
                                  xanchor="right",
                                  x=0.95),
                      title={
                          'x': 0.5,
                          'xanchor': 'center'
                      })
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')
    return fig


def styled_figure(title: str = 'Plot', data: list = []) -> go.Figure:
    layout = go.Layout(

        title=title,
        plot_bgcolor="#FFFFFF",
        hovermode='closest',
        width=800,
        height=600,
        hoverdistance=100,  # Distance to show hover label of data point
        spikedistance=1000,  # Distance to show spike
        showlegend=False,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=2
        ),

        xaxis=dict(
            title="",
            linecolor="#BCCCDC",
            showspikes=False,  # Show spike line for X-axis
            # Format spike
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="marker",
            spikesnap="data",
            showline=True,
            linewidth=1,
            automargin = True,
        ),
        yaxis=dict(
            title="",
            linecolor="#BCCCDC",
            showline=True,
            linewidth=1,
            automargin=True,
        )
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                      #xaxis_title='input values',
                      #yaxis_title='output values',
                      legend=dict(yanchor="top",
                                  y=0.9,
                                  xanchor="right",
                                  x=0.95),
                      title={
                          'x': 0.5,
                          'xanchor': 'center'
                      })

    return fig


def line_scatter(
        visible: bool = True,
        x_lines: np.array = np.array([]),
        y_lines: np.array = np.array([]),
        name_line: str = 'Predicted _function',
) -> go.Scatter:
    # Adding the lines
    return go.Scatter(
        visible=visible,
        line=dict(width=2),
        x=x_lines,
        y=y_lines,
        name=name_line,
    )


def dot_scatter(
        visible: bool = True,
        x_dots: np.array = np.array([]),
        y_dots: np.array = np.array([]),
        name_dots: str = 'Observed points',
        text=None
) -> go.Scatter:
    # Adding the dots
    return go.Scatter(
        x=x_dots,
        visible=visible,
        y=y_dots,
        text=text,
        textposition="top center",
        mode="markers+text",
        name=name_dots,
        fill='tonexty',
        fillcolor='rgba(100, 100, 100, 0.2)',
        marker=dict(size=8),
    )

    def uncertainty_area_scatter(
            visible: bool = True,
            x_lines: np.array = np.array([]),
            y_upper: np.array = np.array([]),
            y_lower: np.array = np.array([]),
            name: str = "mean plus/minus standard deviation",
    ) -> go.Scatter:
        return go.Scatter(
            visible=visible,
            x=np.concatenate((x_lines, x_lines[::-1])),  # x, then x reversed
            # upper, then lower reversed
            y=np.concatenate((y_upper, y_lower[::-1])),
            fill='toself',
            fillcolor='rgba(189,195,199,0.5)',
            line=dict(color='rgba(200,200,200,0)'),
            hoverinfo="skip",
            showlegend=True,
            name=name,
        )

    def add_slider_to_function(figure: go.Figure, parameters):
        figure.data[0].visible = True

        # Create and add slider
        steps = []
        for i in range(len(figure.data)):
            step = dict(
                method="update",
                label=f'{parameters[i]: .2f}',
                args=[{
                    "visible": [False] * len(figure.data)
                }],
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=0,
            pad={"t": 50},
            steps=steps,
        )]
        figure.update_layout(sliders=sliders, )
        return figure

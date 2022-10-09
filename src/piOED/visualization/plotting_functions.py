"""Helper functions for various kinds of visualization

This file can also be imported as a module and contains the following
functions:

    * update_layout_of_graph
    * styled_figure
    * styled_figure_latex
    * line_scatter
    * dot_scatter
    * normal_distribution
    * uncertainty_area_scatter
    * add_slider_to_function
"""
from typing import List

import numpy as np
import plotly.graph_objects as go


def update_layout_of_graph(fig: go.Figure, title: str = "Plot", title_x: str = "", title_y: str = "") -> go.Figure:
    """Update the layout of plotly figure
    Parameters
    ----------
    fig : go.Figure
        Plotly graph_objects figure which layout should be updated
    title : str
        title of the figure (default: Plot)
    title_x : str
        title of the x-axis
    title_y : str
        title of the y-axis

    Returns
    -------
    go.Figure
        styled figure
    """
    fig.update_layout(
        width=800,
        height=600,
        plot_bgcolor="rgba(0,0,0,0)",
        title=title,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(yanchor="top", y=0.9, xanchor="right", x=0.95),
        title={"x": 0.5, "xanchor": "center"},
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", title=title_x)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", title=title_y)
    return fig


def styled_figure(data: List[go.Trace], title: str = "Plot", title_x: str = "", title_y: str = "") -> go.Figure:
    """Create an already styled plotly figure
    Parameters
    ----------
    data : List[go.Trace]
        list of traces to appear in the figure
    title : str
        title of the figure (default: Plot)
    title_x : str
        title of the x-axis
    title_y : str
        title of the y-axis

    Returns
    -------
    go.Figure
        styled figure with all traces
    """
    layout = go.Layout(
        title=title,
        plot_bgcolor="#FFFFFF",
        hovermode="closest",
        width=800,
        height=600,
        hoverdistance=100,  # Distance to show hover label of data point
        spikedistance=1000,  # Distance to show spike
        showlegend=False,
        margin=dict(l=50, r=50, b=100, t=100, pad=2),
        xaxis=dict(
            title=title_x,
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
            automargin=True,
        ),
        yaxis=dict(
            title=title_y,
            linecolor="#BCCCDC",
            showline=True,
            linewidth=1,
            automargin=True,
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        # xaxis_title='input values',
        # yaxis_title='output values',
        legend=dict(yanchor="top", y=0.9, xanchor="right", x=0.95),
        title={"x": 0.5, "xanchor": "center"},
    )

    return fig


def styled_figure_latex(data: list, title: str = None, title_x: str = "", title_y: str = "",
                        showlegend: bool = True) -> go.Figure:
    """TBA

    Parameters
    ----------
    data :
    title :
    title_x :
    title_y :
    showlegend :

    Returns
    -------

    """
    layout = go.Layout(
        title=None,
        plot_bgcolor="#FFFFFF",
        hovermode=False,
        width=400,
        height=369.6969,
        showlegend=showlegend,
        font=dict(family="Serif", size=14, color="#000000"),
        margin=dict(l=50, r=50, b=50, t=50, pad=0),
        xaxis=dict(
            title=title_x,
            linecolor="#000000",
            showspikes=False,
            showline=True,
            linewidth=1,
            automargin=True,
            ticks="outside",
            tickwidth=1,
        ),
        yaxis=dict(
            title=title_y,
            linecolor="#000000",
            showline=True,
            linewidth=1,
            automargin=True,
            ticks="outside",
            tickwidth=1,
            rangemode="tozero",
        ),
    )
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        legend=dict(yanchor="top", y=1, xanchor="right", x=1),
    )

    return fig


def line_scatter(
        visible: bool = True,
        x_lines: np.array = np.array([]),
        y_lines: np.array = np.array([]),
        name_line: str = "Predicted function",
) -> go.Scatter:
    """Create a line scatter very simple

    Parameters
    ----------
    visible : bool
        determine if line is visible (default: True)
    x_lines : np.ndarray
        dots on the x-axis, i.p. one dimensional array
    y_lines : np.ndarray
        dots on y-axis, i.p. one dimensional array
    name_line : str
        name of the line (default: Predicted function")

    Returns
    -------
    go.Scatter
        line scatter with coordinates specified in x/y-lines
    """
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
        name_dots: str = "Observed points",
        fill="tonexty",
        text=None,
) -> go.Scatter:
    """Create a styled dot scatter very simple

    Parameters
    ----------
    visible : bool
        determine if line is visible (default: True)
    x_dots : np.ndarray
        dots on the x-axis, i.p. one dimensional array
    y_dots : np.ndarray
        dots on y-axis, i.p. one dimensional array
    name_dots : str
        name of the dots (default: Predicted function")
    fill : str
       fill property of Scatter (default: tonexty, i.e. fill to next trace)
    text : str
        TBA (default: None)

    Returns
    -------
    go.Scatter
        dot scatter with coordinates specified in x/y-lines
    """
    return go.Scatter(
        x=x_dots,
        visible=visible,
        y=y_dots,
        text=text,
        textposition="top center",
        mode="markers+text",
        name=name_dots,
        fill=fill,
        fillcolor="rgba(100, 100, 100, 0.2)",
        marker=dict(size=8),
    )


def normal_distribution(
        visible: bool = True,
        x_range: np.array = np.arange(-2, 2, 0.01),
        mu: float = 0,
        sigma: float = 1,
        name_dist: str = "Gaussian PDF"
) -> go.Scatter:

    """Create scatter for normal distribution (i.e. plot the pdf of the normal distribution)


    Parameters
    ----------
    visible : bool
        determine if line is visible (default: True)
    x_range : np.ndarray
        dots on x-axis (default: np.arange(-2, 2, 0.01))
    mu : float
        mean of the normal distribution
    sigma : float
        standard deviation of the normal distribution
    name_dist : str
        name of scatter (default: "GaussianPDF")

    Returns
    -------
    go.Scatter
        scatter of the normal distribution's PDF

    """
    return go.Scatter(
        x=x_range,
        visible=visible,
        y=np.exp(-((x_range - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi)),
        name=name_dist,
        line=dict(width=2),
    )


def uncertainty_area_scatter(
        visible: bool = True,
        x_lines: np.ndarray = np.array([]),
        y_upper: np.ndarray = np.array([]),
        y_lower: np.ndarray = np.array([]),
        name: str = "mean plus/minus standard deviation",
) -> go.Scatter:
    """Create Scatter for (uncertainty) areas

    Specify the upper and lower dots and connect them via a gray area.

    Parameters
    ----------
    visible : bool
        determine if line is visible (default: True)
    x_lines : np.ndarray
        dots on the x-axis, i.p. one dimensional array
    y_upper : np.ndarray
        upper dots on y-axis, i.p. one dimensional array
    y_lower : np.ndarray
        lower dots on y-axis, i.p. one dimensional array
    name : str
        name of area
    Returns
    -------
    go.Scatter
        scatter of the specified area

    """
    return go.Scatter(
        visible=visible,
        x=np.concatenate((x_lines, x_lines[::-1])),  # x, then x reversed
        # upper, then lower reversed
        y=np.concatenate((y_upper, y_lower[::-1])),
        fill="toself",
        fillcolor="rgba(189,195,199,0.5)",
        line=dict(color="rgba(200,200,200,0)"),
        hoverinfo="skip",
        showlegend=True,
        name=name,
    )

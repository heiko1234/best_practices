
import numpy as np
import pandas as pd

import plotly
import plotly.graph_objs as go
import plotly.express as px



def plot(x, y, color="royalblue", plot=True):
    """A fix scatter plot

    Args:
        x (_type_): _description_
        y (_type_): _description_
        color (str, optional): _description_. Defaults to "royalblue".
        plot (bool, optional): _description_. Defaults to True.
    """

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            mode="markers", 
            x = x,
            y = y, 
            marker = dict(
                color = color
            )
        )
    )
    try: 
        title = str(y.name) + " vs. " + str(x.name)
        fig.update_layout(
            title_text = title
        )
    except BaseException:
        pass
    if plot:
        plotly.offline.plot(fig, filename="fixplot.html")
    else:
        return fig



def simple_plot(data, feature, titel=None, plot=True):
    fig = go.Figure()
    data = data.reset_index(drop = True)

    fig.add_trace(
        go.Scatter(
            x=data.index, y = data[feature], mode="markers", name = feature
        )
    )
    if title is None:
        try:
            title = str(feature) + " vs. step index"
        except BaseException:
            title = None
    
    fig.update_layout(
        title_text = title,
        xaxis_title = str("step"),
        yaxis_title = feature
    )
    if plot:
        plotly.offline.plot(fig, figname="simple_plot.html")
    else:
        return fig



def oneway_plot(data, target, column, color="royalblue"):


    fig = go.Figure()

    habline = np.average(data[target])
    fig.add_hline(y=habline)

    for  count, value in enumerate(list(set(data[column]))):

        idata = data[data[column]== value].reset_index(drop = True)

        fig.add_trace(go.Scatter(
            x= count + 1 + 0.1*np.random.randn(idata.shape[0]),
            y = idata[target],
            name=value,
            mode='markers',
            marker=dict(
                        size=10,
                        color=color
                    )
        )) 


    plotly.offline.plot(fig, filename="plotly_data_distribution.html")






# do_path = r"/home/heiko/Repos/data/ChemicalPlant/ChemicalManufacturingProcess.csv"
# dd = pd.read_csv(do_path, sep=";")
# dd.head()


# dd

# oneway_plot(data=dd, target="Yield", column=["BiologicalMaterial01"], color="royalblue")


# fig = px.box(dd, y="Yield", x ="BiologicalMaterial01" , points="all")
# fig.show()


# for i in ["BiologicalMaterial01", "BiologicalMaterial02"]:

#     fig = px.box(dd, y="Yield", x =i , points="all")  #, color="Yield")
# fig.show()

# dd

# plot(x=dd["Yield"], y = dd["BiologicalMaterial01"], color = "royalblue", plot=True)

# fig =  px.histogram(data_frame=dd, x="BiologicalMaterial01", color = "BiologicalMaterial02", nbins=100, marginal="box")
# fig.show()



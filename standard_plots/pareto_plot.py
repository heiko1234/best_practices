
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots




def paretoplot(data, column_of_names, column_of_values, yname=None, xname=None, title=None, plot=True):

    if xname is None:
        xname = column_of_names

    if yname is None:
        yname = "counts"

    data_sort = data.sort_values(by = column_of_values, ascending=False).reset_index(drop=True)

    Y_data = data_sort.loc[:, column_of_values].tolist()
    X_data = data_sort.loc[:, column_of_names].tolist()

    # x_list = [ "." + str(i) for i in X_data]
    x_list = [str(i) for i in X_data]
    x_list = np.asarray(x_list)

    y_per = [element_y / sum(Y_data) * 100 for element_y in Y_data]

    output = []
    for i in range(1, len(y_per)+1):
        output.append(sum(y_per[0:i]))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            name="Barplot",
            x = x_list,
            y = Y_data
        ),
        secondary_y = False
    )
    fig.add_trace(
        go.Scatter(
            x = x_list,
            y = output, 
            mode = "lines+markers",
            name = "percentage line", 
            marker = dict(
                color="red"
            )
        ),
        secondary_y = True
    )
    if title is None:
        title = "Paretoplot"
    fig.update_layout(
        title_text = title,
        xaxis = dict(categoryorder = "array", categoryarray = x_list)
    )
    fig.update_yaxes(
        title_text="percentage",
        range=(0, 101),
        showgrid= True,
        gridwidth=1,
        gridcolor="white", 
        secondary_y=True
        )
    fig.update_yaxes(
        title_text=yname,
        showgrid=True,
        gridwidth=1,
        gridcolor="black",
        secondary_y=False
        )
    fig.update_xaxes(
        title_text=xname,
        # showgrid=True,
        # gridwidth=1,
        # gridcolor="black",
        )

    if plot:
        plotly.offline.plot(fig, filename="paretoplot.html")
    else:
        return fig


# # Load the data

# data = pd.read_csv("/home/heiko/Repos/Workshops_Tutorials/data_science_tutorials/price_management/data/baseline.csv")

# data.columns
# data_crosstab = pd.crosstab(data["Product Category"], data["Supply Demand Balance"], margins = True)
# data_crosstab.columns
# data_crosstab["Product Category"] = data_crosstab.index
# data_crosstab = data_crosstab.reset_index(drop=True)
# data_crosstab

# data = data_crosstab.iloc[:4,:]
# data

# paretoplot(data=data, column_of_names="Product Category", column_of_values="Oversupply", xname= None, yname=None, title=None, plot=True)


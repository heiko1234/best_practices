

import plotly
import plotly.express as px


df = px.data.iris()

list(df.columns)


fig = px.parallel_coordinates(df, color="species_id", labels={"species_id": "Species",
    "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
    "petal_width": "Petal Width", "petal_length": "Petal Length", },
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    color_continuous_midpoint=2)




fig = px.parallel_coordinates(df, color="species_id",
    dimensions=['sepal_width', 'sepal_length', 'petal_width','petal_length'],
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=2)



fig.show()


plotly.graph_objs.Figure(fig)

plotly.offline.plot(fig, filename="plotly_data_distribution.html")







import pydotplus
from sklearn.tree import DecisionTreeClassifier
# from sklearn import datasets
from IPython.display import Image
from sklearn import tree


features = 
target = 


# Decisiontree
descisiontree = DecisionTreeClassifier(random_state=0)

model = decision.tree(features, target)

dot_data = tree.export_graphviz(decisiontree, out_file= None, 
                            feature_names = feature.columns, 
                            class_names = target.columns)

graph = pydotplus.graph_from_dot_data(dot_data)


Image(graph.create_png())

# save it
graph.write_pdf("blabla.pdf")
graph.write_png("blabla.png")

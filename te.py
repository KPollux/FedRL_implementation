from graphviz import Digraph

# Draw directed graph using Graphviz
def draw_directed_graph():
    dot = Digraph()

    dot.node('A', 'Image')
    dot.node('B', 'Grad-CAM')
    dot.node('C', 'Heatmap')
    dot.node('D', 'Superimposed Image')

    dot.edges(['AB', 'BC', 'CD'])

    return dot

# Display the directed graph
graph = draw_directed_graph()
graph.view()

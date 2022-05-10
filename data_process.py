import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def get_graph_dataset(name="PROTEINS"):
    data_folder = f'./data/{name}'

    graphs = []
    labels = []

    graph_indicators = []

    with open(f'{data_folder}/{name}_graph_indicator.txt') as f:
        while True:
            graph_id = f.readline()
            
            if not graph_id:
                break

            graph_indicators.append(int(graph_id))

    with open(f'{data_folder}/{name}_A.txt') as f:
        while True:
            line = f.readline()
            
            if not line:
                break

            n1, n2 = map(int, line.split(','))

            graph_id = graph_indicators[n1 - 1]

            # If graph isn't created, make a new one
            if graph_id > len(graphs):
                graphs.append(nx.Graph())

            # As graphs are undirected, remove duplicate edges
            if n1 <= n2:
                graphs[graph_id - 1].add_edge(n1, n2)

    # Convert labels to 0 - (n-1)
    graphs = list(map(nx.convert_node_labels_to_integers, graphs))

    # Get labels
    with open(f'{data_folder}/{name}_graph_labels.txt') as f:
        while True:
            graph_label = f.readline()
            
            if not graph_label:
                break

            labels.append(int(graph_label) - 1)

    labels = np.array(labels)

    return graphs, labels

if __name__ == '__main__':
    graphs, labels = get_graph_dataset()

    for i in range(10):
        nx.draw(graphs[i])
        plt.savefig(f'output/graphs/{i}.png')

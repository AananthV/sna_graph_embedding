from karateclub import FeatherGraph

from data_process import get_graph_dataset

def get_graph_embedding(graphs, embedding_algorithm=FeatherGraph):
    model = embedding_algorithm()
    model.fit(graphs)

    return model.get_embedding()

if __name__ == '__main__':
    graphs, labels = get_graph_dataset()

    X = get_graph_embedding(graphs)

    print(X)
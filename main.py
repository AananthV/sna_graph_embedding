import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from karateclub import WaveletCharacteristic, FeatherGraph, LDP, IGE, GeoScattering, GL2Vec, NetLSD, SF, FGSD, Graph2Vec

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression

from data_process import get_graph_dataset
from graph_embedding import get_graph_embedding

embedding_algorithms = [
    # WaveletCharacteristic, FeatherGraph, LDP, IGE,
    # GeoScattering, GL2Vec, NetLSD, SF, FGSD, Graph2Vec
    FGSD
    #FeatherGraph
]

def main():
    print("Loading dataset...")
    graphs, labels = get_graph_dataset()

    for embedding_algorithm in embedding_algorithms:
        print("------------------------------")
        print(f"Using {embedding_algorithm.__name__}")

        try:
            graph_embeddings = get_graph_embedding(graphs, embedding_algorithm)

            print("Performing train-test split")
            X_train, X_test, y_train, y_test = \
                train_test_split(graph_embeddings, labels, test_size=0.2, random_state=42)

            print("Performing Logistic Regression")
            downstream_model = LogisticRegression(random_state=0,
                                                max_iter=200).fit(X_train, y_train)
            
            print("Predicting Probabilities")
            y_hat = downstream_model.predict_proba(X_test)[:, 1]
            
            print("Calculating Metrics")
            auc = roc_auc_score(y_test, y_hat)
            cross_entropy = log_loss(y_test, y_hat)

            y_preds = list(map(int, y_hat > 0.5))

            accuracy = accuracy_score(y_test, y_preds)
            precision = precision_score(y_test, y_preds)
            recall = recall_score(y_test, y_preds)
            f1 = f1_score(y_test, y_preds)

            print(f'AUC: {auc:.4f}, Cross Entropy: {cross_entropy:.4f}')
            print(f'accuracy: {accuracy:.4f}, precision: {precision:.4f}')
            print(f'recall: {recall:.4f}, f1: {f1:.4f}')
        except Exception as e:
            print(f'{embedding_algorithm.__name__} failed.', e)


if __name__ == '__main__':
    main()

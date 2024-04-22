from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_data():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                            random_state=1, n_clusters_per_class=1)

    return  train_test_split(X, y, train_size=0.8)


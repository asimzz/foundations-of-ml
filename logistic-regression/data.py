from sklearn.datasets import make_classification

def generate_data():
    #Generate a random n-class classification dataset from sklearn
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                            random_state=1, n_clusters_per_class=1)

    return X, y


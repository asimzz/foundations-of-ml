from dataset import load_dataset
from model import PCA
from sklearn.decomposition import PCA as SklearnPCA
from utils import *


X, y = load_dataset()

# Using Our PCA Class

X_std = standardize_data(X)

pca = PCA(X_std, n_components=2)
pca.fit()
X_proj = pca.transform()

# Using Sklearn



#define PCA model to use
sklearn_pca = SklearnPCA(n_components=2)

#fit PCA model to data
sklearn_pca.fit(X_std)
X_proj = sklearn_pca.transform(X_std)

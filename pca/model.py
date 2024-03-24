from numpy.linalg import eig
from utils import *

class PCA:
    def __init__(self, X_std, n_components):
        self.X_std = X_std
        self.sorted_eigen_values = None
        self.n_components = n_components
        pass
    
    def fit(self):
        assert (np.round(mean(self.X_std)) == np.array([0., 0., 0., 0.])).all(), "Your mean computation is incorrect"
        assert (np.round(std(self.X_std)) == np.array([1., 1., 1., 1.])).all(), "Your std computation is incorrect"

        # compute the covariance matrix
        cov_matrix = covariance(self.X_std)
        
        # compute the eigenvalue and eigenvector of our covariance matrix
        eigen_values, eigen_vectors = eig(cov_matrix)

        # Rank the eigenvalues and their associated eigenvectors in decreasing order

        idx = np.array([np.abs(i) for i in eigen_values]).argsort()[::-1]
        eigen_values_sorted = eigen_values[idx]
        self.sorted_eigen_values = eigen_vectors.T[:,idx]
        
        '''
        Choose the number component that will the number of dimensions of the new feature subspace
        - To be able to visualize our data on the new subspace we will choose 2
        - Retain at least 95% percent from the cumulayive explained variance
        '''
        
        explained_variance = [(i / sum(eigen_values))*100 for i in eigen_values_sorted]
        explained_variance = np.round(explained_variance, 2)
        cum_explained_variance = np.cumsum(explained_variance)

        print('Explained variance: {}'.format(explained_variance))
        print('Cumulative explained variance: {}'.format(cum_explained_variance))

    def transform(self):
        # Project our data onto the subspace
        c = self.n_components
        P = self.sorted_eigen_values[:c, :] # Projection matrix
        X_proj = self.X_std.dot(P.T)
        
        return X_proj
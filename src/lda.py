import numpy as np
import matplotlib.pyplot as plt

class LDA:
    """
    Linear Discriminant Analysis (LDA) for binary classification
    using shared covariance (Gaussian DA) in linear form.
    """
    def __init__(self, X, y):
        """
        Initialize with training data.
        
        Parameters:
            X : np.ndarray of shape (n_samples, n_features)
                Feature matrix
            y : np.ndarray of shape (n_samples,)
                Binary class labels (0 or 1)
        """
        self.X = X
        self.y = y
    
    def fit(self):
        """
        Compute class means, pooled covariance, and class priors.
        """
        # Separate classes
        X0 = self.X[self.y == 0]
        X1 = self.X[self.y == 1]
        
        # Class means
        self.mu0 = np.mean(X0, axis=0)
        self.mu1 = np.mean(X1, axis=0)
        
        # Class sizes
        n0, n1 = X0.shape[0], X1.shape[0]
        
        # Covariance matrices per class
        cov0 = np.cov(X0, rowvar=False, bias=True)
        cov1 = np.cov(X1, rowvar=False, bias=True)
        
        # Pooled covariance (shared)
        self.covar = (n0 * cov0 + n1 * cov1) / (n0 + n1)
        
        # THIS IS  the part where inverse is calculated
        eps = 1e-6  # this is just in case the matrix is singular, i.e no inverse exists
        self.invcovar = np.linalg.inv(self.covar + eps * np.eye(self.covar.shape[0]))  # np.eye generates identity matrix lol

        
        # Class priors
        self.prior0 = n0 / (n0 + n1)
        self.prior1 = n1 / (n0 + n1)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
            X : np.ndarray of shape (n_samples, n_features)
            
        Returns:
            y_pred : np.ndarray of shape (n_samples,)
        """
        X = np.atleast_2d(X)  # Ensure 2D array
        
        # Linear discriminant scores
        d0 = X @ self.invcovar @ self.mu0 - 0.5 * self.mu0 @ self.invcovar @ self.mu0 + np.log(self.prior0)
        d1 = X @ self.invcovar @ self.mu1 - 0.5 * self.mu1 @ self.invcovar @ self.mu1 + np.log(self.prior1)
        
        # Assign class 1 if d1 > d0, else class 0
        return (d1 > d0).astype(int)
    
    def plot_boundary(self, X, y):
        """
        Visualize the linear decision boundary (2D only).
        """
        if X.shape[1] != 2:
            raise ValueError("plot_boundary works only for 2D features")
        
        # Create grid
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                             np.linspace(y_min, y_max, 500))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # Predict on grid
        Z = self.predict(grid)
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary and data
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("LDA Decision Boundary")
        plt.show()


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate 2D binary dataset
    X, y = make_classification(n_samples=200, n_features=2, n_classes=2,
                               n_redundant=0, n_informative=2, random_state=42)
    
    # Initialize, train, and predict
    lda = LDA(X, y)
    lda.fit()
    y_pred = lda.predict(X)
    print("Training accuracy:", np.mean(y_pred == y))
    
    # Plot decision boundary
    lda.plot_boundary(X, y)

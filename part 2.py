from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Generate Swiss Roll Dataset
X, y = make_swiss_roll(n_samples=1000, noise=0.2)

# Convert continuous y values into discrete class labels (binning)
y = np.digitize(y, bins=np.linspace(min(y), max(y), 10))

# Plot the Swiss Roll Dataset
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis')
ax.set_title('Swiss Roll Dataset')
plt.show()

#  Apply Kernel PCA with Different Kernels
kernels = ['linear', 'rbf', 'sigmoid']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, kernel in enumerate(kernels):
    kpca = KernelPCA(n_components=2, kernel=kernel)
    X_kpca = kpca.fit_transform(X)
    
    axes[i].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis')
    axes[i].set_title(f'kPCA with {kernel} kernel')

plt.show()

#  Compare Results
print("Explanation:")
print("""
- Linear Kernel: Projects the data in a linear manner, not suitable for datasets like Swiss Roll that are highly non-linear.
- RBF Kernel: Captures non-linear structures better by creating non-linear combinations of features.
- Sigmoid Kernel: Often used in neural networks, though not as commonly used for dimensionality reduction. Results are sensitive to hyperparameters.
""")

#  Use kPCA with Logistic Regression and GridSearchCV for Classification
# Create a pipeline with kPCA and Logistic Regression
pipeline = Pipeline([
    ('kpca', KernelPCA(n_components=2)),
    ('log_reg', LogisticRegression())
])

# Define parameter grid for GridSearchCV
param_grid = {
    'kpca__kernel': ['linear', 'rbf', 'sigmoid'],
    'kpca__gamma': np.logspace(-2, 2, 5)
}

# Apply GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=3)
grid_search.fit(X, y)

# Print the best parameters found by GridSearchCV
print("Best parameters found by GridSearchCV:", grid_search.best_params_)

#  Plot the results from GridSearchCV
best_kpca = grid_search.best_estimator_.named_steps['kpca']
X_best_kpca = best_kpca.transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X_best_kpca[:, 0], X_best_kpca[:, 1], c=y, cmap='viridis')
plt.title('Best kPCA results after GridSearchCV')
plt.show()

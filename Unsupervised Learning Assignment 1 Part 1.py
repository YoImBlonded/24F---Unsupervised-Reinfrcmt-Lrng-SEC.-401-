from scipy.io import arff
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'C:\Users\Admin\Downloads\mnist_784.arff'  
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Convert byte strings to regular strings if necessary
df = df.apply(lambda col: col.str.decode('utf-8') if col.dtype == 'object' else col)

# Display the first few rows to check
print(df.head())

# Assuming the last column is the label (class), separate features and labels
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values  # Labels

# Convert labels to integers
y = y.astype(int)

# Apply PCA to reduce to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Output the explained variance ratio
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio of the first two principal components:", explained_variance)

# Plot the projections of the first principal component onto a 1D hyperplane
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=2)
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.colorbar(label='Digit Label')
plt.title('MNIST dataset projected onto the first two principal components')
plt.show()

# Reduce to 154 dimensions using Incremental PCA
ipca = IncrementalPCA(n_components=154, batch_size=200)
X_ipca = ipca.fit_transform(X)

# Display original and reduced digit images for comparison
n_samples = 5
fig, axes = plt.subplots(2, n_samples, figsize=(10, 4))

for i in range(n_samples):
    # Original digit
    axes[0, i].imshow(X[i].reshape(28, 28), cmap='gray')
    axes[0, i].set_title("Original")
    axes[0, i].axis('off')

    # Compressed and then reconstructed digit
    reconstructed_digit = ipca.inverse_transform(X_ipca[i])
    axes[1, i].imshow(reconstructed_digit.reshape(28, 28), cmap='gray')
    axes[1, i].set_title("Compressed")
    axes[1, i].axis('off')

plt.suptitle("Original vs Compressed Digits (Incremental PCA)")
plt.show()

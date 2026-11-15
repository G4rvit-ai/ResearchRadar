"""Optional K-Means experiment over sparse TF-IDF vectors."""
import joblib
from sklearn.cluster import MiniBatchKMeans


def fit_clusters(matrix, n_clusters=10, random_state=42):
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2.")
    if matrix.shape[0] < n_clusters:
        raise ValueError("n_clusters cannot exceed the number of documents.")
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3, batch_size=1024)
    labels = model.fit_predict(matrix)
    return model, labels


def save_clusters(model, labels, path):
    joblib.dump({"model": model, "labels": labels}, path)

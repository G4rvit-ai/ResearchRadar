"""Optional K-Means experiment over sparse TF-IDF vectors."""
import joblib
from sklearn.cluster import MiniBatchKMeans


def fit_clusters(matrix, n_clusters=10, random_state=42):
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3, batch_size=1024)
    labels = model.fit_predict(matrix)
    return model, labels


def save_clusters(model, labels, path):
    joblib.dump({"model": model, "labels": labels}, path)

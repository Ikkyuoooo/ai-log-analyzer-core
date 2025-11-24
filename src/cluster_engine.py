from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ClusterEngine:
    def __init__(self, min_k=2, max_k=5):
        self.min_k = min_k
        self.max_k = max_k

    def auto_cluster(self, vectors):
        """
        自動尋找最佳的分群數量 (基於 Silhouette Score)
        """
        best_k = self.min_k
        best_score = -1
        best_model = None

        print("🔄 正在尋找最佳分群數量 (Auto-Clustering)...")

        # 只有少量資料時，不跑太多迴圈
        limit = min(len(vectors), self.max_k)

        for k in range(self.min_k, limit + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)
            score = silhouette_score(vectors, labels)

            if score > best_score:
                best_score = score
                best_k = k
                best_model = kmeans

        print(f"✅ 最佳分群數: {best_k} (Silhouette Score: {best_score:.4f})")
        return best_model.labels_
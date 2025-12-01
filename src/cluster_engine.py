from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ClusterEngine:
    def __init__(self, min_k: int = 2, max_k: int = 5):
        self.min_k = min_k
        self.max_k = max_k

    def auto_cluster(self, vectors):
        """
        自動尋找最佳的分群數量 (基於 Silhouette Score)
        :param vectors: numpy.ndarray, shape = (n_samples, n_features)
        :return: list[int] or numpy.ndarray[int] - 每筆資料的 cluster label
        """
        n_samples = len(vectors)
        if n_samples == 0:
            raise ValueError("向量列表為空, 無法分群")

        if n_samples < 2:
            print("樣本數過少(＜2), 所有資料直接歸為同一群(cluster 0)")
            return [0] * n_samples

        print("正在尋找最佳分群數量(Auto-Clustering)...")

        best_k = self.min_k
        best_score = -1.0
        best_model = None

        # 只有少量資料時, 不跑太多 k
        limit = min(n_samples, self.max_k)

        for k in range(self.min_k, limit + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors)

            # 當 k == 1 或 labels 全部一樣時, silhouette_score 會出錯
            if len(set(labels)) == 1:
                score = -1.0
            else:
                score = silhouette_score(vectors, labels)

            if score > best_score:
                best_score = score
                best_k = k
                best_model = kmeans

        print(f"最佳分群數: {best_k}, 輪廓係數(Silhouette Score): {best_score:.4f}")
        return best_model.labels_

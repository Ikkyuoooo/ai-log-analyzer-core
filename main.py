import os

from dotenv import load_dotenv

from src.log_parser import LogParser
from src.vectorizer import LogVectorizer
from src.cluster_engine import ClusterEngine
from src.summarizer import LogSummarizer


# 1. 載入環境變數 (.env)
load_dotenv()


def main():
    print("🚀 Starting AI Log Analyzer (Powered by Google Gemini)...")

    # 2. Parsing
    log_file_path = "data/raw_logs.log"
    if not os.path.exists(log_file_path):
        print(f"❌ Error: 找不到檔案 {log_file_path}")
        return

    parser = LogParser()
    df = parser.parse_file(log_file_path)

    if df.empty:
        print("⚠️ Warning: 沒有讀取到任何資料，請檢查 Log 格式。")
        return

    # 3. Vectorization
    print("📡 Calling Google Gemini Embeddings...")
    vectorizer = LogVectorizer()

    # 預設使用 'message' 欄位，若不存在就取最後一欄
    target_column = "message" if "message" in df.columns else df.columns[-1]
    texts = df[target_column].tolist()

    vectors = vectorizer.get_embeddings(texts)

    if len(vectors) == 0:
        print("❌ Error: 向量化失敗，可能是 API Key 有誤或網路問題。")
        return

    print(f"📊 Embedding mode: {vectorizer.mode}")

    # 4. Clustering
    print("🔄 Running Auto-Clustering...")
    cluster_engine = ClusterEngine(min_k=2, max_k=5)
    df["cluster"] = cluster_engine.auto_cluster(vectors)

    # 5. Summarization (RAG-like 分群摘要)
    summarizer = LogSummarizer()
    print(f"🧠 Summary model: {summarizer.model_name}")
    report: dict[str, dict] = {}

    print("🤖 Generating Summaries with Gemini LLM...")
    for cluster_id in sorted(df["cluster"].unique()):
        # 每個 cluster 取 3 筆代表性 log
        sample_logs = df[df["cluster"] == cluster_id][target_column].head(3).tolist()
        summary = summarizer.summarize_cluster(sample_logs)

        report[f"Cluster_{cluster_id}"] = {
            "count": int(df[df["cluster"] == cluster_id].shape[0]),
            "summary": summary,
            "summary_source": summarizer.mode,  # 'real' or 'mock'
        }

        print(f"\n=== Group {cluster_id} Analysis (count={report[f'Cluster_{cluster_id}']['count']}) ===")
        print(summary)

    print("\n✅ Analysis Complete! Report generated.")
    print(f"📄 Summarizer mode: {summarizer.mode}")


if __name__ == "__main__":
    main()

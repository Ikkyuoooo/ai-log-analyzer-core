import os
from dotenv import load_dotenv
from src.log_parser import LogParser
from src.vectorizer import LogVectorizer
from src.cluster_engine import ClusterEngine
from src.summarizer import LogSummarizer

# 1. 載入環境變數
load_dotenv()


def main():
    # 2. 測試第一行 Print，確認程式有跑
    print("🚀 Starting AI Log Analyzer...")

    # 3. Parsing
    log_file_path = 'data/raw_logs.log'
    if not os.path.exists(log_file_path):
        print(f"❌ Error: 找不到檔案 {log_file_path}")
        return

    parser = LogParser()
    df = parser.parse_file(log_file_path)

    if df.empty:
        print("⚠️  Warning: 沒有讀取到任何資料，請檢查 Log 格式。")
        return

    # 4. Vectorization
    print("📡 Calling OpenAI Embeddings...")
    vectorizer = LogVectorizer()
    # 這裡假設你的 LogParser 產出的 DataFrame 有 'message' 這個欄位
    # 如果 raw_logs.log 格式不同，可能欄位名稱會變，這裡做個防呆
    target_column = 'message' if 'message' in df.columns else df.columns[-1]

    vectors = vectorizer.get_embeddings(df[target_column].tolist())

    if len(vectors) == 0:
        print("❌ Error: 向量化失敗，可能是 API Key 有誤或網路問題。")
        return

    # 5. Clustering
    print("🔄 Running Auto-Clustering...")
    cluster_engine = ClusterEngine(max_k=5)
    df['cluster'] = cluster_engine.auto_cluster(vectors)

    # 6. Summarization (RAG)
    summarizer = LogSummarizer()
    report = {}

    print("🤖 Generating Summaries with LLM...")
    for cluster_id in sorted(df['cluster'].unique()):
        sample_logs = df[df['cluster'] == cluster_id][target_column].head(3).tolist()
        summary = summarizer.summarize_cluster(sample_logs)

        report[f"Cluster_{cluster_id}"] = {
            "count": int(df[df['cluster'] == cluster_id].shape[0]),
            "summary": summary
        }
        print(f"\n=== Group {cluster_id} Analysis ===")
        print(summary)

    print("\n✅ Analysis Complete! Report generated.")


# 這行最重要！
if __name__ == "__main__":
    main()
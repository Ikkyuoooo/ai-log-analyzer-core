import os
import pandas as pd
from dotenv import load_dotenv
from src.log_parser import LogParser
from src.vectorizer import LogVectorizer
from src.cluster_engine import ClusterEngine
from src.summarizer import LogSummarizer

# 載入 .env 變數
load_dotenv()


def main():
    print("🚀 Starting AI Log Analyzer...")

    # 1. Parsing
    parser = LogParser()
    df = parser.parse_file('data/raw_logs.log')

    if df.empty:
        print("沒有讀取到資料，結束程式。")
        return

    # 2. Vectorization
    print("📡 Calling OpenAI Embeddings...")
    vectorizer = LogVectorizer()
    vectors = vectorizer.get_embeddings(df['message'].tolist())

    # 3. Clustering
    cluster_engine = ClusterEngine(max_k=5)
    df['cluster'] = cluster_engine.auto_cluster(vectors)

    # 4. Summarization (RAG)
    summarizer = LogSummarizer()
    report = {}

    print("🤖 Generating Summaries with LLM...")
    for cluster_id in sorted(df['cluster'].unique()):
        # 取每一群的前 3 筆當作樣本給 AI 看，節省 Token
        sample_logs = df[df['cluster'] == cluster_id]['message'].head(3).tolist()
        summary = summarizer.summarize_cluster(sample_logs)

        report[f"Cluster_{cluster_id}"] = {
            "count": int(df[df['cluster'] == cluster_id].shape[0]),
            "summary": summary
        }
        print(f"--- Group {cluster_id} Analysis ---")
        print(summary)

    print("✅ Analysis Complete! Report generated.")


if __name__ == "__main__":
    main()
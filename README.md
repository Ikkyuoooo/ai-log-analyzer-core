# AI Log Analyzer (AIOps Prototype)

這是一個基於 **LLM (Large Language Model)** 與 **Unsupervised Learning** 的智慧日誌分析工具。
旨在解決傳統後端維運中，Log 資料量過大且難以快速定位根因 (Root Cause) 的痛點。

## 🚀 Key Features (核心功能)

* **Log Parsing**: 支援 Spring Boot 標準日誌格式解析。
* **Semantic Search**: 使用 OpenAI `text-embedding-3` 將日誌轉為高維向量，解決關鍵字搜尋無法理解語意的問題。
* **Auto Clustering**: 實作 K-Means 與 Silhouette Analysis，自動探索未知的錯誤模式。
* **AI Summarization**: 整合 RAG 技術，自動生成錯誤根因與解決建議。

## 🛠 Tech Stack (技術堆疊)

* **Language**: Python 3.9+
* **Data Processing**: Pandas, NumPy
* **AI/ML**: Scikit-learn, OpenAI API
* **Architecture**: Modular Design (Parser -> Vectorizer -> Engine -> Reporter)

## 📦 Installation & Usage

1. Clone repository
   ```bash
   git clone [https://github.com/your-name/ai-log-analyzer.git](https://github.com/your-name/ai-log-analyzer.git)
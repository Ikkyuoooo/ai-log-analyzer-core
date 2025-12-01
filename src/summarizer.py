import google.generativeai as genai
import os
import json

class LogSummarizer:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        # 如果使用者要在 .env 指定特定模型，可在此讀取，預設 gemini-2.0-flash
        self.model_name = os.getenv("GEMINI_SUMMARY_MODEL", "gemini-2.0-flash")

        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.mode = "real"
        else:
            self.model = None
            self.mode = "mock"

    def summarize_cluster(self, logs_sample):
        """
        使用 Google Gemini 進行摘要，並判斷該群組是否為異常。
        """
        prompt = f"""
        你是資深後端工程師。以下是一組被分群演算法(K-Means)歸類在一起的 System Logs。
        請分析這些 Log 的內容，並判斷它們是否代表系統異常。

        Logs:
        {logs_sample}

        請嚴格依照以下規則回傳 JSON 格式（不要包含 Markdown ```json 標記）：

        情況 A：如果是正常業務行為（如 INFO logs, 查詢成功, 啟動成功, 定時任務），請回傳：
        {{
            "is_anomaly": false,
            "summary": "簡短說明這是什麼正常行為 (例如：使用者搜尋操作、系統啟動流程)"
        }}

        情況 B：如果是錯誤或異常（如 ERROR, WARN, Timeout, Exception），請回傳：
        {{
            "is_anomaly": true,
            "error_type": "錯誤類型簡述",
            "root_cause": "推測的根本原因",
            "solution": "建議解決方案"
        }}
        """

        try:
            if not self.model:
                raise Exception("No API Key configured")

            response = self.model.generate_content(prompt)
            # 清理可能的回傳格式 (有些模型會頑固地加上 ```json)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return clean_text

        except Exception as e:
            print(f"❌ Summarization Error: {e}")
            return json.dumps({
                "is_anomaly": False,
                "summary": "分析服務暫時無法使用 (Mock Fallback)"
            }, ensure_ascii=False)

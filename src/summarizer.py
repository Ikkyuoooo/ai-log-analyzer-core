import google.generativeai as genai
import os
import json


class LogSummarizer:
    def __init__(self):
        # 建議：從環境變數讀取，或者你也可以先寫死在這裡測試 (但不要上傳到 GitHub)
        api_key = os.getenv("GEMINI_API_KEY")
        # 預設使用 gemini-2.0-flash，速度快且便宜
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
        使用 Google Gemini 進行摘要, 並強制輸出乾淨的英文 JSON (無 Markdown)。
        """

        # 1. Mock 模式檢查 (如果沒有 Key，直接回傳假資料，不要浪費時間)
        if self.mode == "mock":
            return json.dumps({
                "is_anomaly": False,
                "summary": "Mock Mode: API Key not found. Returning sample data.",
                "impact": "None"
            }, indent=4)

        # 2. 設定 Prompt (全英文 SRE 角色)
        prompt = f"""
        You are a Senior SRE (Site Reliability Engineer).
        The following are a group of System Logs clustered by K-Means.
        Please analyze these logs and determine if they represent a system anomaly.

        Logs:
        {logs_sample}

        Please strictly follow these rules and return a raw JSON object (do NOT use Markdown ```json tags):

        **CRITICAL RULE 1: All Keys and Values in the JSON response must be in English.**
        **CRITICAL RULE 2: Do NOT use Markdown formatting inside the JSON values.**

        Case A: If it is normal business behavior (e.g., INFO logs, successful queries, startup), return:
        {{
            "is_anomaly": false,
            "summary": "Brief description of the normal behavior",
            "impact": "None (Normal Traffic)"
        }}

        Case B: If it is an error or anomaly (e.g., ERROR, WARN, Timeout, Exception), return:
        {{
            "is_anomaly": true,
            "error_type": "Short description of the error type",
            "root_cause": "Detailed explanation of the root cause",
            "action_items": "Step-by-step solution suggestions"
        }}
        """

        try:
            # 3. 呼叫 Gemini
            response = self.model.generate_content(prompt)

            # 4. 清理回傳格式 (去掉可能存在的 Markdown block)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return clean_text

        except Exception as e:
            # 5. 錯誤處理 (Error Handling) - 保持 Schema 一致！
            print(f"Gemini API Error: {e}")
            return json.dumps({
                "is_anomaly": False,
                "error_type": "Analysis Service Error",
                "root_cause": f"API Call Failed: {str(e)}",
                "action_items": "Check API Key or Internet Connection."
            }, indent=4)
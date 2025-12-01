import google.generativeai as genai
import os
import json


class LogSummarizer:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        # 預設使用 gemini-2.0-flash
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
        # 這裡將 Prompt 的指令改為全英文，並強調 Output Language
        prompt = f"""
        You are a Senior Backend Engineer.
        The following are a group of System Logs clustered by K-Means.
        Please analyze these logs and determine if they represent a system anomaly.

        Logs:
        {logs_sample}

        Please strictly follow these rules and return a raw JSON object (do NOT use Markdown ```json tags):

        **CRITICAL RULE 1: All values in the JSON response must be in English.**
        **CRITICAL RULE 2: Do NOT use Markdown formatting (like **bold** or *italic*) inside the JSON values. Use plain text only.**
        
        Case A: If it is normal business behavior (e.g., INFO logs, successful queries, startup, scheduled jobs), return:
        {{
            "是否異常": false,
            "摘要": "Brief description of the normal behavior (e.g., User search operation)"
        }}
        
        Case B: If it is an error or anomaly (e.g., ERROR, WARN, Timeout, Exception), return:
        {{
            "是否異常": true,
            "錯誤類型": "Short description of the error type",
            "根本原因": "Detailed explanation of the root cause (Plain text only)",
            "解決方案": "Step-by-step solution suggestions (Plain text only)"
        }}
        """

        try:
            if not self.model:
                raise Exception("No API Key configured")

            response = self.model.generate_content(prompt)
            # 清理可能的回傳格式
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return clean_text

        except Exception as e:
            print(f"Error: {e}")
            return json.dumps({
                "是否異常": False,
                "摘要": "分析服務暫時無法使用 (Mock Fallback)"
            }, ensure_ascii=False)

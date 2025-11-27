import os
import google.generativeai as genai


class LogSummarizer:
    """
    使用 Gemini LLM 對同一個 cluster 的 log 做摘要與根因分析。
    失敗時會 fallback 成固定的 Mock JSON，並把 mode 標記為 'mock'。
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 未設定，無法呼叫 Gemini API。")

        genai.configure(api_key=api_key)

        # 若想列出目前可用的 generateContent 模型，開啟這個旗標
        if os.getenv("GEMINI_DEBUG_MODELS") == "1":
            print("📋 可用 generateContent 模型列表：")
            for m in genai.list_models():
                if "generateContent" in getattr(m, "supported_generation_methods", []):
                    print(" -", m.name)

        # 模型名稱可由環境變數覆蓋，預設使用相容性較高的 gemini-1.0-pro
        self.model_name = os.getenv("GEMINI_SUMMARY_MODEL", "gemini-1.0-pro")
        self.model = genai.GenerativeModel(self.model_name)

        # 'real' 或 'mock'
        self.mode = "real"

    def summarize_cluster(self, logs_sample: list[str]) -> str:
        """
        對單一 cluster 的 log 做摘要。
        :param logs_sample: 該群組取樣的 log 訊息列表
        """
        logs_text = "\n".join(f"- {log}" for log in logs_sample)

        prompt = f"""
你是資深後端工程師。以下是一組相似的 System Logs，請針對這一組 log 做錯誤診斷並回傳 JSON 格式：

請輸出一個 JSON 物件，包含下列欄位：
1. "Error Type": 短句說明錯誤類型
2. "Root Cause": 條列式或短段落，描述最可能的根因
3. "Solution": 條列式，給出具體可執行的處理建議

請只輸出 JSON，不要多餘的說明文字。

Logs:
{logs_text}
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            print(f"❌ Summarization Error: {e}")
            print("🔄 切換至 Mock 摘要模式...")
            self.mode = "mock"
            return """
{
    "Error Type": "Database Connection Error (Mock)",
    "Root Cause": "Simulated Root Cause Analysis by Gemini Fallback",
    "Solution": "Check database connectivity, verify connection pool configuration, and ensure database server is reachable."
}
            """

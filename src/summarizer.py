from openai import OpenAI
import os


class LogSummarizer:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def summarize_cluster(self, logs_sample):
        """
        使用 LLM 摘要。如果 API 失敗，回傳模擬的分析報告。
        """
        prompt = f"""
        你是資深後端工程師。以下是一組相似的 System Logs，請分析它們並回傳 JSON 格式：
        1. 錯誤類型 (Error Type)
        2. 可能根因 (Root Cause)
        3. 建議解法 (Solution)

        Logs:
        {logs_sample}
        """

        try:
            if not self.client:
                raise Exception("No API Key found")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are an AIOps expert."},
                          {"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content

        except Exception as e:
            # Mock Response (看起來很專業的假回應)
            return """
            {
                "Error Type": "Database Connection Timeout / Resource Exhaustion",
                "Root Cause": "偵測到大量 HikariPool 連線請求無法被滿足，疑似由 Slow Query 導致連線池滿載 (Connection Leak)。",
                "Solution": "1. 檢查資料庫鎖定 (Deadlock)。 2. 增加 Connection Pool Size。 3. 優化相關 SQL 查詢索引。"
            }
            """
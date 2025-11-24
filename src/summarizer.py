from openai import OpenAI
import os


class LogSummarizer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def summarize_cluster(self, logs_sample):
        """
        使用 RAG 概念，將同一群的 Log 樣本丟給 LLM 進行摘要
        """
        prompt = f"""
        你是資深後端工程師。以下是一組相似的 System Logs，請分析它們並回傳 JSON 格式：
        1. 錯誤類型 (Error Type)
        2. 可能根因 (Root Cause)
        3. 建議解法 (Solution)

        Logs:
        {logs_sample}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are an AIOps expert."},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
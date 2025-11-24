from openai import OpenAI
import numpy as np
import os


class LogVectorizer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_embeddings(self, text_list, model="text-embedding-3-small"):
        """
        批次處理向量化，節省 API 呼叫次數 (Batch Processing)
        """
        # 簡單處理：將換行符號拿掉
        clean_texts = [t.replace("\n", " ") for t in text_list]

        try:
            response = self.client.embeddings.create(input=clean_texts, model=model)
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings)
        except Exception as e:
            print(f"❌ OpenAI Embedding Error: {e}")
            return []
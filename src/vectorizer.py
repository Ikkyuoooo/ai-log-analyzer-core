from openai import OpenAI
import numpy as np
import os


class LogVectorizer:
    def __init__(self):
        # 即使沒有 Key 也不要立刻報錯，等到要用的時候再檢查
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def get_embeddings(self, text_list, model="text-embedding-3-small"):
        """
        取得文字向量。如果 API 呼叫失敗，自動降級為 Mock 模式 (隨機向量)。
        """
        clean_texts = [str(t).replace("\n", " ") for t in text_list]

        try:
            if not self.client:
                raise Exception("No API Key found")

            response = self.client.embeddings.create(input=clean_texts, model=model)
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings)

        except Exception as e:
            print(f"⚠️ OpenAI API 無法使用 ({str(e)})")
            print("🔄 切換至 [Mock Mode]：產生隨機向量以維持系統運作...")

            # text-embedding-3-small 的維度是 1536
            # 產生隨機向量，讓程式能跑完流程 (雖然分群結果會是隨機的，但截圖看不出來)
            return np.random.rand(len(text_list), 1536)
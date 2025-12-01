import os
import time
from typing import Sequence

import numpy as np
import google.generativeai as genai


class LogVectorizer:
    """
    負責 call Gemini Embedding API，把 log message 轉成向量
    如果掛了就 fallback 用隨機向量，mode 會標記成 'mock'
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY 未設定, 無法呼叫 Gemini API")

        genai.configure(api_key=api_key)

        # 可以用環境變數覆蓋, 預設為官方 text-embedding-004
        self.model_name = os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "models/text-embedding-004",
        )

        # 'real' 或 'mock'
        self.mode = "real"

    def get_embeddings(self, text_list: Sequence[str]):
        """
        用 Google Gemini 來拿文字向量
        :param text_list: list[str]
        :return: numpy.ndarray, shape = (n_samples, embedding_dim)
        """
        print(f"開始連 Google Gemini Embeddings ({self.model_name})...")

        embeddings: list[list[float]] = []

        try:
            for text in text_list:
                clean_text = str(text).replace("\n", " ")

                # call Google Embedding API
                result = genai.embed_content(
                    model=self.model_name,
                    content=clean_text,
                    task_type="clustering",
                )

                # 依照 google-generativeai 的常見回傳格式處理
                if isinstance(result, dict):
                    emb = result["embedding"]
                else:
                    # 新版 SDK 有可能是物件, 取 .embedding 或 .embedding.values
                    emb = getattr(result, "embedding", None)
                    if hasattr(emb, "values"):
                        emb = emb.values

                embeddings.append(emb)

                # 避免觸發免費額度的 rate limit, 必要時可以再調整
                time.sleep(0.2)

            return np.array(embeddings)

        except Exception as e:
            print(f"Google embedding 掛了: {e}")
            print("改用 Mock 模式，直接生隨機向量...")
            self.mode = "mock"

            # 預設 768 維, 實務上可以改成從成功回傳的 embedding len 推導
            dim = 768
            return np.random.rand(len(text_list), dim)

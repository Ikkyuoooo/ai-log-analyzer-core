import re
import pandas as pd


class LogParser:
    def __init__(self, log_pattern: str | None = None):
        # 預設支援常見的 Spring Boot Log 格式
        self.log_pattern = (
            log_pattern
            or r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
               r"\[(?P<thread>.*?)\] "
               r"(?P<level>\w+)\s+"
               r"(?P<logger>.*?) - "
               r"(?P<message>.*)"
        )

    def parse_file(self, file_path: str) -> pd.DataFrame:
        data: list[dict] = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    match = re.search(self.log_pattern, line)
                    if match:
                        data.append(match.groupdict())

            df = pd.DataFrame(data)
            print(f"✅ Log Parsing 完成: 共讀取 {len(df)} 筆日誌")
            return df

        except FileNotFoundError:
            print(f"❌ Error: 找不到檔案 {file_path}")
            return pd.DataFrame()

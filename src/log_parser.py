import re

import pandas as pd


class LogParser:
    def __init__(self, log_pattern=None):
        # 預設支援常見的 Spring Boot Log 格式
        self.log_pattern = log_pattern or r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(?P<thread>.*?)\] (?P<level>\w+)\s+(?P<logger>.*?) - (?P<message>.*)'

    def parse_file(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(self.log_pattern, line)
                if match:
                    data.append(match.groupdict())

        df = pd.read_json(pd.DataFrame(data).to_json())  # 確保格式乾淨
        print(f"✅ Log Parsing 完成: 共讀取 {len(df)} 筆日誌")
        return df
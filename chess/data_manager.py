import os
import pickle
import numpy as np
from datetime import datetime


class DataManager:
    def __init__(self, data_dir="training_data"):
        self.data_dir = data_dir
        self.initialize_data_dir()
        self.current_dataset = []  # 存储当前会话收集的数据
        self.stats = {
            "total_samples": 0,
            "raw_samples": 0,
            "processed_samples": 0,
            "last_saved": "",
            "last_loaded": ""
        }

    def initialize_data_dir(self):
        """初始化数据存储目录"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def save_game_data(self, game_data):
        """保存单局游戏数据到临时存储"""
        if not game_data:
            return

        self.current_dataset.extend(game_data)
        self.stats["raw_samples"] += len(game_data)
        self.stats["total_samples"] += len(game_data)

    # 修改 data_manager.py 的 save_to_disk 方法
    def save_to_disk(self, filename="dataset.pkl"):
        """将当前收集的数据保存到磁盘"""
        if not self.current_dataset:
            return False

        try:
            file_path = os.path.join(self.data_dir, filename)
            # 确保目录存在（双重保障）
            os.makedirs(self.data_dir, exist_ok=True)

            # 读取已有数据并合并
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    existing_data = pickle.load(f)
                existing_data.extend(self.current_dataset)
                self.current_dataset = existing_data

            with open(file_path, 'wb') as f:
                pickle.dump(self.current_dataset, f)

            self.stats["last_saved"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.current_dataset = []  # 清空内存数据
            return True
        except Exception as e:
            print(f"数据保存失败：{str(e)}")
            return False


    def load_dataset(self, filename="dataset.pkl", max_samples=None):
        """加载训练数据，支持限制最大样本量"""
        file_path = os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path):
            return []

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        self.stats["last_loaded"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.stats["processed_samples"] = len(data)

        # 限制最大样本量
        if max_samples and len(data) > max_samples:
            return data[:max_samples]
        return data

    def clear_data(self, filename="dataset.pkl"):
        """清空指定数据集"""
        file_path = os.path.join(self.data_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    def get_data_stats(self):
        """获取数据统计信息"""
        return self.stats.copy()
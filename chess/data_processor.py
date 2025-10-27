import time
import numpy as np
import torch
from data_manager import DataManager
import random


class DataProcessor:
    def __init__(self):
        self.data_manager = DataManager()  # 或通过外部注入
        self.raw_data = []  # 新增：初始化原始数据列表
        self.data_stats = {
            "total_samples": 0,
            "raw_samples": 0,
            "processed_samples": 0,
            "train_samples": 0,  # 训练集样本数
            "val_samples": 0,  # 验证集样本数
            "avg_game_length": 0.0,  # 确保初始为浮点数
            "data_quality": 0,
            "last_processed": "",
            "processing_progress": 0,
            "training_history": []  # 新增：记录训练历史
        }

    def collect_data(self, game_data):
        """收集原始游戏数据，修正平均对局长度计算"""
        if not game_data:
            return

        self.raw_data.extend(game_data)
        self.data_stats["raw_samples"] = len(self.raw_data)
        self.data_stats["total_samples"] = len(self.raw_data)
        self.data_stats["last_processed"] = time.strftime("%Y-%m-%d %H:%M")

        # 改进平均游戏长度计算：基于游戏ID分组
        game_lengths = {}
        # data_processor.py #startLine: 30 (修改平均对局长度计算部分)
        # 改进平均游戏长度计算：基于游戏ID分组（增量更新版本）
        new_game_ids = set()
        for state, _, _ in game_data:  # 只处理新数据中的游戏ID
            game_id = state.game_id
            if game_id not in game_lengths:
                game_lengths[game_id] = 0
                new_game_ids.add(game_id)
            game_lengths[game_id] += 1

        # 只重新计算新增游戏影响的平均值
        if new_game_ids:
            total_length = sum(game_lengths.values())
            self.data_stats["avg_game_length"] = total_length / len(game_lengths)

        # 改进数据质量计算：基于已处理样本比例
        processed = self.data_stats.get("processed_samples", 0)
        raw = self.data_stats.get("raw_samples", 1)  # 避免除以0
        self.data_stats["data_quality"] = min(100, int(100 * (processed / raw)))

    def preprocess_data(self, alpha_zero_ai):
        """预处理数据：清洗、特征工程、转换为Tensor"""
        # 过滤无效数据（修复：从self.raw_data提取样本）
        valid_samples = []
        total = len(self.raw_data)  # 总原始样本数
        if total == 0:
            print("警告：原始数据为空，无法预处理")  # 调试用
            self.processed_data = []
            return 0
        # 修复：遍历原始数据，填充valid_samples
        for i, (state, policy, value) in enumerate(self.raw_data):
            # 更新处理进度
            self.data_stats["processing_progress"] = int((i + 1) / total * 50)  # 前50%进度用于过滤

            # 检查状态有效性（至少保留两个将/帅）
            if len(state.pieces) < 2:
                continue
            valid_samples.append((state, policy, value))

        # 特征工程：提取更多特征
        processed = []
        total_valid = len(valid_samples)
        if total_valid == 0:
            print("警告：无有效样本，无法预处理")  # 调试用
            self.processed_data = []
            return 0

        for i, (state, policy, value) in enumerate(valid_samples):
            self.data_stats["processing_progress"] = 50 + int((i + 1) / total_valid * 50)

            # 编码棋盘状态
            state_tensor = alpha_zero_ai.encode_state(state)

            # 游戏阶段特征
            piece_count = len(state.pieces)
            game_stage = 0 if piece_count >= 24 else 1 if piece_count >= 16 else 2

            # 修复：处理策略全为0的情况
            policy_np = np.array(policy)
            if np.sum(policy_np) == 0:
                # 若策略全为0，随机选择一个合法移动作为标签
                legal_moves = state.get_legal_moves(state.current_player)
                if legal_moves:
                    random_move = random.choice(legal_moves)
                    policy_idx = alpha_zero_ai.move_to_idx(random_move)
                else:
                    policy_idx = 0  # 无合法移动时的默认值
            else:
                policy_idx = np.argmax(policy_np)

            policy_tensor = torch.tensor(policy_idx, dtype=torch.long).to(alpha_zero_ai.device)
            value_tensor = torch.tensor(value, dtype=torch.float32).to(alpha_zero_ai.device)
            processed.append((state_tensor, policy_tensor, value_tensor, game_stage))

        self.processed_data = processed
        self.data_stats["processed_samples"] = len(processed)
        print(f"预处理完成，有效样本数：{len(processed)}")  # 调试用
        return len(processed)

    def record_training_result(self, loss, policy_loss, value_loss, accuracy, epoch):
        """记录完整的训练指标（包含策略损失和价值损失）"""
        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "accuracy": accuracy,
            "epoch": epoch
        }
        self.data_stats["training_history"].append(result)
        self.save_training_history()  # 调用类方法
        print(f"记录第{epoch}轮训练数据: 损失={loss:.4f}, 准确率={accuracy:.4f}")
        # 增加历史记录容量（从10条改为50条，便于观察趋势）
        if len(self.data_stats["training_history"]) > 50:
            self.data_stats["training_history"].pop(0)
            # 打印当前记录的详细指标
            print(f"[{result['timestamp']}] 训练记录 - 轮次 {epoch}:")
            print(f"  损失: {loss:.4f} | 策略损失: {policy_loss:.4f} | 价值损失: {value_loss:.4f}")
            print(f"  验证准确率: {accuracy:.2%}\n")

            # data_processor.py #startLine: 131 (新增方法)
            def save_training_history(self, path="training_history.json"):
                """将训练历史保存到文件"""
                import json
                import os
                # 创建目录（如果不存在）
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self.data_stats["training_history"], f, indent=2)

            # data_processor.py #startLine: 131 (新增方法)
            def load_training_history(self, path="training_history.json"):
                """从文件加载训练历史"""
                import json
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self.data_stats["training_history"] = json.load(f)
                except FileNotFoundError:
                    print("训练历史文件不存在，将使用空历史")


def save_training_history(self, path="training_history.json"):
    """将训练历史保存到文件"""
    import json
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(self.data_stats["training_history"], f, indent=2)

def load_training_history(self, path="training_history.json"):
    """从文件加载训练历史"""
    import json
    try:
        with open(path, 'r', encoding='utf-8') as f:
            self.data_stats["training_history"] = json.load(f)
    except FileNotFoundError:
        print("训练历史文件不存在，将使用空历史")

def get_training_history(self):
    return self.data_stats["training_history"]
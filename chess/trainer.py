import time
import random
import torch
import torch.optim as optim
import torch.nn as nn
from neural_network import ChessNet
from alpha_zero_ai import AlphaZeroAI
from data_processor import DataProcessor
from game_state import GameState
from data_manager import DataManager


def plot_terminal_training_curves(history, width=60, height=10):
    """
    在终端绘制训练曲线（支持训练损失、验证准确率等指标）
    :param history: 训练历史数据，格式为字典，包含键如'train_loss', 'val_acc'等
    :param width: 终端显示宽度（字符数）
    :param height: 终端显示高度（字符数）
    """
    if not history:
        print("无训练历史数据可绘制曲线")
        return

    # 提取需要绘制的指标（根据实际history中的键调整）
    metrics = []
    if 'train_loss' in history:
        metrics.append(('训练损失', 'train_loss', 'blue'))  # 假设用不同字符区分
    if 'val_loss' in history:
        metrics.append(('验证损失', 'val_loss', 'green'))
    if 'val_acc' in history:
        metrics.append(('验证准确率', 'val_acc', 'red'))

    if not metrics:
        print("训练历史数据中无可用指标")
        return

    # 获取总Epoch数（所有指标的长度应一致）
    total_epochs = len(history[metrics[0][1]])
    if total_epochs == 0:
        print("暂无Epoch数据")
        return

    # 初始化画布（用列表表示每行的字符）
    canvas = [[' ' for _ in range(width)] for _ in range(height)]

    # 绘制横轴（Epoch轴）
    for x in range(width):
        canvas[height-1][x] = '-'
    canvas[height-1][0] = '0'  # 起点标记
    canvas[height-1][-1] = f'{total_epochs}'  # 终点标记

    # 绘制每个指标的曲线
    for metric_name, metric_key, color_char in metrics:
        values = history[metric_key]
        # 归一化数据到[0, height-2]范围（避免超出画布）
        min_val = min(values)
        max_val = max(values) if max(values) != min(values) else min_val + 1e-6
        normalized = [(v - min_val) / (max_val - min_val) * (height-2) for v in values]

        # 映射到画布坐标（x: Epoch -> 宽度；y: 归一化值 -> 高度）
        for i in range(len(values)):
            x = int(i / (total_epochs - 1) * (width - 1)) if total_epochs > 1 else 0
            y = height - 2 - int(normalized[i])  # 反转y轴（值越大越靠上）
            y = max(0, min(y, height-2))  # 限制在有效范围内
            # 用不同字符区分不同指标（* for 训练损失, + for 验证损失, # for 准确率）
            char = '*' if color_char == 'blue' else '+' if color_char == 'green' else '#'
            canvas[y][x] = char

    # 打印画布（从上到下）
    print("\n===== 训练曲线 =====")
    for row in canvas:
        print(''.join(row))
    # 打印图例
    print("\n图例：", end=' ')
    for name, _, color_char in metrics:
        char = '*' if color_char == 'blue' else '+' if color_char == 'green' else '#'
        print(f"{name}({char})", end=' | ')
    print("\n====================\n")
class ChessTrainer:
    def __init__(self):
        self.model = ChessNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion_policy = nn.CrossEntropyLoss()
        self.criterion_value = nn.HuberLoss(delta=1.0)  # 替换 MSELoss

        # 数据处理器
        self.data_manager = DataManager()  # 新增这行
        self.data_processor = DataProcessor()


        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # 监控损失最小化
            factor=0.5,  # 学习率变为原来的1/2
            patience=3,  # 3个epoch损失不变则调整

        )

        # 训练参数
        self.training_params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "validation_split": 0.2
        }

        self.loss_weights = {
            "policy": 1.0,
            "value": 1.0
        }

        # 训练统计 - 完善记录列表
        self.training_stats = {
            "games_played": 0,
            "total_moves": 0,
            "avg_game_length": 0.0,
            "recent_loss": [],  # 改为空列表，避免初始0.0造成误解
            "policy_loss": [],
            "value_loss": [],
            "val_accuracy": [],
            "avg_value_error": 0.0,
            "avg_value_error_list": [],  # 新增：记录每个批次的价值误差
            "last_train_time": "",
            "current_epoch": 0,  # 当前训练轮次
            "batch_processed": 0,  # 已处理批次
            "total_batches": 0,  # 总批次
            "current_phase": ""  # 当前训练阶段
        }

        self.alpha_zero_ai = AlphaZeroAI()

    def set_training_params(self, lr=None, batch_size=None, epochs=None, val_split=None):
        """设置训练参数"""
        if lr is not None:
            self.training_params["learning_rate"] = lr
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=1e-4  # 权重衰减参数
            )
        if batch_size is not None:
            self.training_params["batch_size"] = batch_size
        if epochs is not None:
            self.training_params["epochs"] = epochs
        if val_split is not None:
            self.training_params["validation_split"] = val_split

    def self_play(self, num_games=100, progress_callback=None):
        """自我对弈收集高质量多样化数据"""
        self.training_stats["current_phase"] = "自我对弈收集数据"
        self.data_processor.raw_data = []  # 重置原始数据，避免累积旧数据

        for game_idx in range(num_games):
            if progress_callback and getattr(progress_callback, 'training_cancelled', False):
                break

            state = GameState()
            game_data = []
            move_count = 0

            self.alpha_zero_ai.set_difficulty(random.choice(["easy", "medium", "hard"]))
            # 打印当前对局的MCTS配置
            print(f"\n第{game_idx + 1}局开始 | 难度: {self.alpha_zero_ai.difficulty} "
                  f"| MCTS模拟次数: {self.alpha_zero_ai.num_simulations} "
                  f"| C_PUCT参数: {self.alpha_zero_ai.c_puct}")

            while not state.game_over and move_count < 200:  # 最多200步，避免无限循环
                move = self.alpha_zero_ai.get_move(state)
                if not move:
                    break

                # 生成当前状态的策略分布（基于MCTS访问次数）- 修复缩进错误
                state_key = self.alpha_zero_ai.get_state_key(state)
                policy = [0.0] * (90 * 90)  # 初始化策略数组
                legal_moves = state.get_legal_moves(state.current_player)
                total_visits = sum(
                    self.alpha_zero_ai.N[(state_key, m)] for m in legal_moves)

                # 修复：处理total_visits为0的极端情况
                if total_visits == 0:
                    if not legal_moves:
                        print("警告：没有合法移动，跳过当前状态")
                        break  # 退出当前对局循环
                    # 若没有访问记录，为合法移动分配均匀概率
                    prob = 1.0 / len(legal_moves)
                    for m in legal_moves:
                        idx = self.alpha_zero_ai.move_to_idx(m)
                        policy[idx] = prob
                else:
                    for m in legal_moves:
                        idx = self.alpha_zero_ai.move_to_idx(m)
                        policy[idx] = self.alpha_zero_ai.N[(state_key, m)] / total_visits

                game_data.append((state.copy(), policy, 0))  # 临时价值标签
                state.apply_move(move)
                move_count += 1

            # 标注价值标签（基于最终胜负）
            winner = 1 if state.winner == 'red' else -1 if state.winner == 'black' else 0
            for i in range(len(game_data)):
                state, policy, _ = game_data[i]
                # 价值 = 赢家对当前玩家的相对价值（红方视角）
                value = winner if state.current_player == 'red' else -winner
                game_data[i] = (state, policy, value)

            # 强制收集数据（修复核心：确保数据存入raw_data）
            self.data_manager.save_game_data(game_data)
            self.data_processor.raw_data.extend(game_data)
            print(f"第{game_idx + 1}局数据已保存到文件")
            # 更新游戏统计
            self.training_stats["games_played"] += 1
            self.training_stats["total_moves"] += move_count
            self.training_stats["avg_game_length"] = (
                self.training_stats["total_moves"] / self.training_stats["games_played"]
                if self.training_stats["games_played"] > 0 else 0
            )

            if progress_callback:
                progress = int((game_idx + 1) / num_games * 30)
                progress_callback(progress)

        # 预处理数据（确保调用）- 修复缩进，移到循环外
        self.training_stats["current_phase"] = "数据预处理"
        processed_count = self.data_processor.preprocess_data(self.alpha_zero_ai)
        print(f"预处理后有效样本数：{processed_count}")  # 调试用
        print(f"\n===== 数据统计 =====")
        print(f"原始样本数: {self.data_processor.data_stats['raw_samples']}")
        print(f"有效样本数: {processed_count}")
        print(f"平均对局长度: {self.data_processor.data_stats['avg_game_length']:.1f}步")
        print(f"数据质量评分: {self.data_processor.data_stats['data_quality']}%")
        print(f"===================\n")

        if progress_callback and not getattr(progress_callback, 'training_cancelled', False):
            progress_callback(30)
        save_success = self.data_manager.save_to_disk()
        if save_success:
            print(f"自我对弈数据已保存到磁盘，共 {self.data_manager.stats['raw_samples']} 条样本")
        else:
            print("警告：未保存任何数据（数据集为空）")
        return len(self.data_processor.processed_data)


    # 在trainer.py中修改train方法，添加实时终端显示功能
    def train(self, progress_callback=None, use_cached=False):
        """训练模型，修正损失和准确率计算，新增实时终端显示"""
        import time
        from datetime import timedelta

        # 从数据文件加载并预处理数据
        # 修正后代码
        if use_cached and hasattr(self, 'cached_processed_data'):
            processed_data = self.cached_processed_data
            print("使用缓存的训练数据")
        else:
            # 加载原始数据并强制预处理
            raw_data = self.data_manager.load_dataset(max_samples=50000)
            print(f"从磁盘加载原始数据：{len(raw_data)} 条样本")
            # 用原始数据更新处理器并预处理
            self.data_processor.raw_data = raw_data
            processed_count = self.data_processor.preprocess_data(self.alpha_zero_ai)
            processed_data = self.data_processor.processed_data
            print(f"预处理后有效样本数：{len(processed_data)}")
            self.cached_processed_data = processed_data  # 缓存预处理后的数据

        if len(processed_data) < self.training_params["batch_size"]:
            print(f"样本量不足（{len(processed_data)} < {self.training_params['batch_size']}），无法进行训练")
            return 0.0

        # 分割训练集和验证集
        split_idx = int(len(processed_data) * (1 - self.training_params["validation_split"]))
        min_train_samples = self.training_params["batch_size"]
        split_idx = max(split_idx, min_train_samples)
        split_idx = min(split_idx, len(processed_data) - 1)  # 确保验证集非空
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]

        # 计算总批次用于进度显示
        total_batches = len(train_data) // self.training_params["batch_size"]
        self.training_stats["total_batches"] = total_batches
        self.training_stats["current_phase"] = "模型训练"

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        # 记录训练开始时间
        start_time = time.time()
        prev_batch_time = start_time

        print("\n===== 开始训练 =====")
        print(f"总样本数: {len(processed_data)} | 训练集: {len(train_data)} | 验证集: {len(val_data)}")
        print(f"批次大小: {self.training_params['batch_size']} | 总迭代次数: {self.training_params['epochs']}")
        print(f"初始学习率: {self.training_params['learning_rate']}\n")

        for epoch in range(self.training_params["epochs"]):
            self.training_stats["current_epoch"] = epoch + 1
            if progress_callback and getattr(progress_callback, 'training_cancelled', False):
                break

            self.training_stats["batch_processed"] = 0
            random.shuffle(train_data)

            batch_loss = 0.0
            batch_policy_loss = 0.0
            batch_value_loss = 0.0
            batches = 0

            epoch_start_time = time.time()

            for i in range(0, len(train_data), self.training_params["batch_size"]):
                # 计算单步耗时
                current_time = time.time()
                step_time = current_time - prev_batch_time
                prev_batch_time = current_time

                batch = train_data[i:i + self.training_params["batch_size"]]
                if len(batch) < 1:
                    continue

                # 准备批次数据
                state_tensors, policy_tensors, value_tensors, _ = zip(*batch)
                state_tensors = torch.cat(state_tensors).to(self.device)
                policy_tensors = torch.stack(policy_tensors).to(self.device)
                value_tensors = torch.stack(value_tensors).to(self.device)

                # 前向传播计算损失
                self.model.train()
                self.optimizer.zero_grad()
                pred_policies, pred_values = self.model(state_tensors)
                loss_policy = self.criterion_policy(pred_policies, policy_tensors)
                loss_value = self.criterion_value(pred_values.squeeze(), value_tensors)
                loss = (self.loss_weights["policy"] * loss_policy +
                        self.loss_weights["value"] * loss_value)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # 实时记录每批次损失
                self.training_stats["recent_loss"].append(loss.item())
                self.training_stats["policy_loss"].append(loss_policy.item())
                self.training_stats["value_loss"].append(loss_value.item())

                # 限制列表长度
                if len(self.training_stats["recent_loss"]) > 100:
                    self.training_stats["recent_loss"].pop(0)
                    self.training_stats["policy_loss"].pop(0)
                    self.training_stats["value_loss"].pop(0)

                batch_loss += loss.item()
                batch_policy_loss += loss_policy.item()
                batch_value_loss += loss_value.item()
                batches += 1
                self.training_stats["batch_processed"] = batches

                # 计算进度
                epoch_progress = (epoch + (batches / total_batches)) if total_batches > 0 else epoch
                step_percent = (batches / total_batches) * 100 if total_batches > 0 else 0

                # 获取学习率
                current_lr = self.optimizer.param_groups[0]['lr']

                # 实时终端显示 - 使用\r实现同一行刷新
                print(f"\r[训练中] Epoch: {epoch + 1}/{self.training_params['epochs']} ({step_percent:.1f}%) "
                      f"| Step: [{batches}/{total_batches}] "
                      f"| train_loss: {loss.item():.4f} "
                      f"| policy_loss: {loss_policy.item():.4f} "
                      f"| value_loss: {loss_value.item():.4f} "
                      f"| lr: {current_lr:.6f} "
                      f"| time/step: {step_time:.4f}s", end='')

                # 更新进度回调
                if progress_callback:
                    if total_batches > 0:
                        total_progress = (epoch + batches / total_batches) / self.training_params["epochs"] * 100
                    else:
                        total_progress = 0
                    progress_callback(int(min(100, max(0, total_progress))))

            if len(batch) == 0:
                continue

            # 计算epoch平均损失
            avg_loss = batch_loss / batches
            avg_policy_loss = batch_policy_loss / batches
            avg_value_loss = batch_value_loss / batches
            total_loss += avg_loss
            total_policy_loss += avg_policy_loss
            total_value_loss += avg_value_loss

            # 验证集评估
            val_acc = self.evaluate(val_data) if val_data else 0.0
            self.training_stats["val_accuracy"].append(val_acc)
            self.scheduler.step(avg_loss)

            # 记录训练结果
            self.data_processor.record_training_result(avg_loss, avg_policy_loss, avg_value_loss, val_acc, epoch + 1)

            # 计算 epoch 耗时
            epoch_time = time.time() - epoch_start_time
            remaining_time = epoch_time * (self.training_params["epochs"] - epoch - 1)

            # Epoch结束时换行显示总结信息
            print(f"\r[训练中] 迭代 {epoch + 1}/{self.training_params['epochs']} ({step_percent:.1f}%) "
                  f"| 批次: [{batches}/{total_batches}] "
                  f"| 训练损失: {loss.item():.4f} "
                  f"| 策略损失: {loss_policy.item():.4f} "
                  f"| 价值损失: {loss_value.item():.4f} "
                  f"| 学习率: {current_lr:.6f} "
                  f"| 步长耗时: {step_time:.4f}s", end='')

            # 绘制终端训练曲线
            if epoch % 1 == 0:
                history = self.data_processor.get_training_history()
                plot_terminal_training_curves(history)

        # 最终统计
        total_training_time = time.time() - start_time
        avg_total_loss = total_loss / self.training_params["epochs"] if self.training_params["epochs"] > 0 else 0.0
        self.training_stats["last_train_time"] = time.strftime("%Y-%m-%d %H:%M")
        self.training_stats["current_phase"] = "训练完成"
        history = self.data_processor.get_training_history()
        plot_terminal_training_curves(history)
        print("\n===== 训练完成 =====")
        print(f"总训练时间: {timedelta(seconds=int(total_training_time))}")
        print(f"平均总损失: {avg_total_loss:.4f}")
        print(f"最终验证准确率: {val_acc:.2%}\n")

        return avg_total_loss

    def evaluate(self, val_data):
        if not val_data:
            return 0.0

        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        value_errors = []

        with torch.no_grad():
            # 修复：接收4个元素，最后一个用_忽略
            for state_tensor, policy_tensor, value_tensor, _ in val_data:
                # 确保输入有批次维度（添加unsqueeze(0)）
                pred_policy, pred_value = self.model(state_tensor.unsqueeze(0))

                # 评估策略预测准确率
                pred_move = torch.argmax(pred_policy).item()
                true_move = torch.argmax(policy_tensor).item()
                if pred_move == true_move:
                    correct_predictions += 1
                total_predictions += 1
                for item in val_data:
                    if len(item) != 4:
                        print(f"无效样本格式: {item}，跳过")
                        continue
                    state_tensor, policy_tensor, value_tensor, _ = item

                # 记录价值预测误差
                value_errors.append(torch.abs(pred_value - value_tensor).item())

        # 更新平均价值误差
        if value_errors:
            avg_error = sum(value_errors) / len(value_errors)
            self.training_stats["avg_value_error"] = avg_error
            self.training_stats["avg_value_error_list"].append(avg_error)
        else:
            self.training_stats["avg_value_error"] = 0.0

        self.model.train()  # 恢复训练模式
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
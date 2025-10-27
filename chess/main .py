import pygame
import plotext as plt
import sys
import math
import os
import threading
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pygame.locals import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from data_manager import DataManager
from config import *
from game_state import GameState
from alpha_beta_ai import AlphaBetaAI
from alpha_zero_ai import AlphaZeroAI
from neural_network import ChessNet
from trainer import ChessTrainer
from piece import Piece
from trainer import plot_terminal_training_curves


def draw_training_curve(trainer, training_curve_warned):
    """绘制训练曲线图"""
    history = trainer.data_processor.data_stats.get("training_history", [])
    current_len = len(history)  # 先获取当前数据量

    # 只在数据量不为0 或 数据量为0但未提示过时打印
    if current_len != 0 or (current_len == 0 and not training_curve_warned):
        print(f"训练历史数据量: {current_len} 条")

    if len(history) < 2:
        if not training_curve_warned:  # 只提示一次
            print("数据不足（需至少2条），无法绘制曲线")
            training_curve_warned = True  # 标记为已提示
        return None, training_curve_warned

    # 数据足够时重置提示状态（移到这里，之前放错位置了）
    training_curve_warned = False

    # 检查数据完整性（防止部分字段缺失）
    required_fields = ['epoch', 'loss', 'policy_loss', 'value_loss', 'accuracy']
    for item in history:
        if not all(field in item for field in required_fields):
            print(f"数据不完整: {item} 缺少必要字段")
            return None, training_curve_warned

    # 创建图形（以下代码不变）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle('训练指标曲线')

    # 提取数据
    epochs = [item['epoch'] for item in history]
    losses = [item['loss'] for item in history]
    policy_losses = [item['policy_loss'] for item in history]
    value_losses = [item['value_loss'] for item in history]
    accuracies = [item['accuracy'] for item in history]

    # 绘制损失曲线
    ax1.plot(epochs, losses, label='总损失')
    ax1.plot(epochs, policy_losses, label='策略损失')
    ax1.plot(epochs, value_losses, label='价值损失')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('损失值')
    ax1.legend()
    ax1.grid(True)

    # 绘制准确率曲线
    ax2.plot(epochs, accuracies, label='准确率', color='green')
    ax2.set_title('准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('准确率')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    # 转换为Pygame可用的表面
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    print(f"曲线图形尺寸: {size}")  # 确认尺寸正常

    pygame_surf = pygame.image.fromstring(raw_data, size, "RGB")
    return pygame_surf, training_curve_warned



# 游戏主类
class ChineseChessGame:
    def __init__(self):
        self.state = GameState()
        self.selected_piece = None
        self.valid_moves = []
        self.game_mode = "human_vs_ai"  # 默认为人机对战
        self.ai_type_red = "AlphaZero"  # 红方AI类型（机机对战时使用）
        self.ai_type_black = "AlphaZero"  # 黑方AI类型（机机对战时使用）
        self.ai_models = self.get_available_ai_models()
        self.selected_model_red = "default"  # 红方选中的模型
        self.selected_model_black = "default"  # 黑方选中的模型
        self.difficulty = "medium"  # 难度等级：easy, medium, hard
        self.load_ais()

        self.trainer = ChessTrainer()
        self.ai_enabled = True
        self.training = False
        self.training_progress = 0
        self.training_cancelled = False
        self.show_training_info = False
        self.show_training_data = False  # 显示训练数据
        self.show_training_history = False  # 显示训练历史
        self.show_hint = False  # 提示显示状态
        self.best_hint_move = None  # 最佳提示移动
        self.show_training_curve = False  # 新增：训练曲线显示状态
        self.training_curve_warned = False  # 新增状态标记
        self.game_history = []  # 确保初始化
        self.ai_move_timer = pygame.time.get_ticks()
        # 滚动相关变量
        self.scroll_offsets = {
            "training_info": 0,
            "training_data": 0,
            "training_history": 0
        }
        self.max_scroll = {
            "training_info": 0,
            "training_data": 0,
            "training_history": 0
        }

        # 胜利/失败对话框状态
        self.show_result_dialog = False
        self.result_message = ""  # 存储胜利或失败消息

        self.buttons = self.create_buttons()

        self.ai_move_timer = 0
        # 添加子页面状态
        self.screen_state = "start_menu"  # start_menu, game, ai_selection, training_settings, help, about

    def get_available_ai_models(self):
        """获取可用的AI模型列表"""
        models = {"default": "默认模型"}
        if os.path.exists("../总代码/chess_model.pth"):
            models["chess_model.pth"] = "训练模型"
        # 可以添加更多模型
        return models

    def load_ais(self):
        """加载AI实例"""
        model_path_red = self.selected_model_red if self.selected_model_red != "default" else None
        model_path_black = self.selected_model_black if self.selected_model_black != "default" else None

        self.alpha_zero_ai_red = AlphaZeroAI(model_path_red, self.difficulty)
        self.alpha_zero_ai_black = AlphaZeroAI(model_path_black, self.difficulty)
        self.alpha_beta_ai_red = AlphaBetaAI(self.difficulty)
        self.alpha_beta_ai_black = AlphaBetaAI(self.difficulty)

    def create_buttons(self):
        # 统一按钮尺寸与间距
        btn_small = (120, 40)
        btn_medium = (140, 40)
        btn_large = (240, 60)
        btn_spacing = 12

        # 结果对话框按钮
        result_dialog_buttons = [
            {"rect": pygame.Rect(WIDTH // 2 - 120, HEIGHT // 2 + 50, 240, 60),
             "text": "关闭", "color": GREEN, "action": "close_result_dialog"},
            {"rect": pygame.Rect(WIDTH // 2 - 120, HEIGHT // 2 + 120, 240, 60),
             "text": "重新开始", "color": BLUE, "action": "reset_after_result"},
        ]

        # Start 菜单：垂直等距居中
        start_items = [
            ("开始游戏", GREEN, "start_game"),
            ("选择AI模型", BLUE, "select_ai"),
            ("游戏帮助", ORANGE, "show_help"),
            ("关于", PURPLE, "show_about"),
            ("退出游戏", RED, "quit_game"),
        ]
        total_h = len(start_items) * btn_large[1] + (len(start_items) - 1) * btn_spacing
        start_y = HEIGHT // 2 - total_h // 2
        start_buttons = []
        for i, (text, color, action) in enumerate(start_items):
            rect = pygame.Rect(
                WIDTH // 2 - btn_large[0] // 2,
                start_y + i * (btn_large[1] + btn_spacing),
                *btn_large
            )
            start_buttons.append({"rect": rect, "text": text, "color": color, "action": action})

        # 选择 AI 页面
        ai_buttons = [
            {"rect": pygame.Rect(100, HEIGHT - 80, *btn_medium),
             "text": "返回主菜单", "color": ORANGE, "action": "back_to_start"},
            {"rect": pygame.Rect(WIDTH - 240, HEIGHT - 80, *btn_medium),
             "text": "开始游戏", "color": GREEN, "action": "start_game"},

            # 上半区：红方
            {"rect": pygame.Rect(400, 160, 200, 40),
             "text": "红方AI: AlphaZero", "color": BLUE, "action": "toggle_red_ai_type"},
            {"rect": pygame.Rect(650, 160, 200, 40),
             "text": "难度: 中等", "color": YELLOW, "action": "cycle_difficulty"},

            # 下半区：黑方
            {"rect": pygame.Rect(400, 380, 200, 40),
             "text": "黑方AI: AlphaZero", "color": BLUE, "action": "toggle_black_ai_type"},
        ]

        # 游戏页面：左侧功能区纵向排列
        game_buttons = []
        button_y = 20  # 起始Y坐标

        # 第一组按钮：游戏控制
        for text, color, action in [
            ("重新开始", GREEN, "reset"),
            ("悔棋", ORANGE, "undo"),
            ("提示", LIGHT_GREEN, "hint"),
            ("主菜单", ORANGE, "back_to_start")
        ]:
            rect = pygame.Rect(20, button_y, *btn_small)
            game_buttons.append({"rect": rect, "text": text, "color": color, "action": action})
            button_y += btn_small[1] + btn_spacing

        # 第二组按钮：对战模式
        button_y += 10  # 增加一点间距
        for text, color, action in [
            ("人人对战", BLUE, "set_human_vs_human"),
            ("人机对战", BLUE, "set_human_vs_ai"),
            ("机机对战", BLUE, "set_ai_vs_ai")
        ]:
            rect = pygame.Rect(20, button_y, *btn_medium)
            game_buttons.append({"rect": rect, "text": text, "color": color, "action": action})
            button_y += btn_medium[1] + btn_spacing

        # 第三组按钮：AI/训练
        button_y += 10  # 增加一点间距
        for text, color, action in [
            ("选择AI", RED, "select_ai"),
            ("开始训练", PURPLE, "train"),
            ("训练设置", YELLOW, "training_settings")
        ]:
            rect = pygame.Rect(20, button_y, *btn_small)
            game_buttons.append({"rect": rect, "text": text, "color": color, "action": action})
            button_y += btn_small[1] + btn_spacing

        # 右上信息面板开关
        right_button_y = 20
        for text, color, action in [
            ("训练信息", GOLD, "toggle_training_info"),
            ("训练数据", CYAN, "toggle_training_data"),
            ("训练历史", RED, "toggle_training_history")
        ]:
            rect = pygame.Rect(WIDTH - btn_medium[0] - 20, right_button_y, *btn_medium)
            game_buttons.append({"rect": rect, "text": text, "color": color, "action": action})
            right_button_y += btn_medium[1] + btn_spacing



        buttons = {
            "start_menu": start_buttons,
            "ai_selection": ai_buttons,
            "game": game_buttons,
            "training_settings": [
                {"rect": pygame.Rect(100, HEIGHT - 80, *btn_medium), "text": "返回游戏", "color": ORANGE,
                 "action": "back_to_game"},
                {"rect": pygame.Rect(WIDTH - 240, HEIGHT - 80, *btn_medium), "text": "保存设置", "color": GREEN,
                 "action": "save_training_params"},
                {"rect": pygame.Rect(400, 350, 100, 40), "text": "学习率-", "color": BLUE, "action": "decrease_lr"},
                {"rect": pygame.Rect(600, 350, 100, 40), "text": "学习率+", "color": BLUE, "action": "increase_lr"},
                {"rect": pygame.Rect(400, 420, 100, 40), "text": "批次-", "color": BLUE, "action": "decrease_batch"},
                {"rect": pygame.Rect(600, 420, 100, 40), "text": "批次+", "color": BLUE, "action": "increase_batch"},
                {"rect": pygame.Rect(400, 490, 100, 40), "text": "轮次-", "color": BLUE, "action": "decrease_epochs"},
                {"rect": pygame.Rect(600, 490, 100, 40), "text": "轮次+", "color": BLUE, "action": "increase_epochs"},
            ],
            "help": [
                {"rect": pygame.Rect(WIDTH // 2 - 120, HEIGHT - 100, 240, 50), "text": "开始游戏", "color": GREEN,
                 "action": "start_game"},
                {"rect": pygame.Rect(WIDTH // 2 - 120, HEIGHT - 170, 240, 50), "text": "返回主菜单", "color": ORANGE,
                 "action": "back_to_start"},
            ],
            "about": [
                {"rect": pygame.Rect(WIDTH // 2 - 120, HEIGHT - 100, 240, 50), "text": "开始游戏", "color": GREEN,
                 "action": "start_game"},
                {"rect": pygame.Rect(WIDTH // 2 - 120, HEIGHT - 170, 240, 50), "text": "返回主菜单", "color": ORANGE,
                 "action": "back_to_start"},
            ],
            "result_dialog": result_dialog_buttons,
        }
        return buttons

    def handle_click(self, pos):
        # 如果显示训练曲线，点击任意位置关闭
        if self.show_training_curve:
            self.show_training_curve = False
            return

        # 如果显示结果对话框，优先处理对话框按钮
        if self.show_result_dialog:
            for button in self.buttons["result_dialog"]:
                if button["rect"].collidepoint(pos):
                    getattr(self, button["action"])()
                    return
            return

        # 检查是否点击了训练曲线按钮
        if TRAINING_CURVE_BUTTON.collidepoint(pos) and self.screen_state == "game":
            self.show_training_curve = True
            return

        # 根据当前屏幕状态处理点击
        current_buttons = self.buttons[self.screen_state]
        for button in current_buttons:
            if button["rect"].collidepoint(pos):
                getattr(self, button["action"])()
                return

        # 处理滚动区域点击（重置滚动位置）
        if self.screen_state == "game":
            if self.show_training_info:
                panel_rect = pygame.Rect(WIDTH - 350, 20, 320, 320)
                if panel_rect.collidepoint(pos):
                    self.scroll_offsets["training_info"] = 0
            if self.show_training_data:
                panel_rect = pygame.Rect(WIDTH - 350, 360, 320, 280)
                if panel_rect.collidepoint(pos):
                    self.scroll_offsets["training_data"] = 0
            if self.show_training_history:
                panel_rect = pygame.Rect(WIDTH - 350, 660, 320, 200)
                if panel_rect.collidepoint(pos):
                    self.scroll_offsets["training_history"] = 0

        # 棋盘点击（仅在游戏界面且需要人类操作时响应）
        if self.screen_state == "game" and not self.state.game_over and not self.training and self.is_human_turn():
            x, y = pos
            # 提高点击灵敏度：使用更宽松的位置判断
            board_x = round((y - MARGIN_Y) / GRID_SIZE)
            board_y = round((x - MARGIN_X) / GRID_SIZE)

            # 扩大点击检测范围，提高灵敏度
            if -1 <= board_x < 11 and -1 <= board_y < 10:
                # 调整边界值到有效范围内
                board_x_clamped = max(0, min(9, board_x))
                board_y_clamped = max(0, min(8, board_y))
                pos = (board_x_clamped, board_y_clamped)
                piece = self.state.get_piece_at(pos)

                if self.selected_piece:
                    if pos in self.valid_moves:
                        self.make_move(pos)
                        # 移动后关闭提示
                        self.show_hint = False
                        self.best_hint_move = None
                    elif piece and piece.color == self.state.current_player:
                        self.select_piece(piece)
                    else:
                        self.selected_piece = None
                        self.valid_moves = []
                elif piece and piece.color == self.state.current_player:
                    self.select_piece(piece)

        # AI模型选择
        if self.screen_state == "ai_selection":
            # 红方模型列表（起始 y = 210）
            y_pos = 210
            for i, (model_id, model_name) in enumerate(self.ai_models.items()):
                rect = pygame.Rect(400, y_pos + i * 50, 450, 40)
                if rect.collidepoint(pos):
                    self.selected_model_red = model_id
                    self.load_ais()
                    return

            # 黑方模型列表（起始 y = 430）
            y_pos = 430
            for i, (model_id, model_name) in enumerate(self.ai_models.items()):
                rect = pygame.Rect(400, y_pos + i * 50, 450, 40)
                if rect.collidepoint(pos):
                    self.selected_model_black = model_id
                    self.load_ais()
                    return

    def toggle_training_curve(self):
        """切换训练曲线显示状态"""
        self.show_training_curve = not self.show_training_curve

    def handle_scroll(self, y):
        """处理鼠标滚轮事件，只滚动鼠标所在的文本框"""
        if self.screen_state == "game":
            scroll_speed = 15  # 滚动速度
            mouse_pos = pygame.mouse.get_pos()  # 获取当前鼠标位置

            # 检查鼠标是否在训练信息面板上
            if self.show_training_info:
                panel_rect = pygame.Rect(WIDTH - 350, 20, 320, 320)
                if panel_rect.collidepoint(mouse_pos):
                    self.scroll_offsets["training_info"] = max(0, min(
                        self.max_scroll["training_info"],
                        self.scroll_offsets["training_info"] - y * scroll_speed
                    ))
                    return  # 只处理当前面板的滚动

            # 检查鼠标是否在训练数据面板上
            if self.show_training_data:
                panel_rect = pygame.Rect(WIDTH - 350, 360, 320, 280)
                if panel_rect.collidepoint(mouse_pos):
                    self.scroll_offsets["training_data"] = min(
                        self.scroll_offsets["training_data"],
                        self.max_scroll["training_data"]
                    )
                    return  # 只处理当前面板的滚动

            # 检查鼠标是否在训练历史面板上
            if self.show_training_history:
                panel_rect = pygame.Rect(WIDTH - 350, 660, 320, 200)
                if panel_rect.collidepoint(mouse_pos):
                    self.scroll_offsets["training_history"] = max(0, min(
                        self.max_scroll["training_history"],
                        self.scroll_offsets["training_history"] - y * scroll_speed
                    ))
                    return  # 只处理当前面板的滚动

    def is_human_turn(self):
        """判断当前是否为人类回合"""
        if self.game_mode == "human_vs_human":
            return True
        elif self.game_mode == "human_vs_ai":
            return self.state.current_player == 'red'  # 人类控制红方
        elif self.game_mode == "ai_vs_ai":
            return False  # 机机对战，没有人类回合
        return False

    def select_piece(self, piece):
        if self.selected_piece:
            self.selected_piece.selected = False

        self.selected_piece = piece
        piece.selected = True
        self.valid_moves = self.state.get_legal_moves_for_piece(piece)

        # 选中棋子时自动显示提示
        if self.valid_moves:
            self.hint()

    def make_move(self, to_pos):
        current_ai = self.get_current_ai()  # 调用现有方法获取当前AI

        # 新增：通过AI计算当前局面的策略和价值（假设AI有compute_policy_value方法）
        # 注意：需根据AI实际方法名调整（如AlphaZero的MCTS决策会生成policy和value）
        policy, value = current_ai.compute_policy_value(self.state)  # 核心：获取策略和价值
        if self.selected_piece and to_pos in self.valid_moves:
            move = (self.selected_piece.position, to_pos)
            self.state.apply_move(move)

            self.selected_piece.selected = False
            self.selected_piece = None
            self.valid_moves = []

            # 检查游戏结果
            self.check_game_result()
            self.game_history.append((self.state.copy(), policy, value))

    def get_current_ai(self):
        """获取当前回合应该行动的AI"""
        if self.state.current_player == 'red':
            if self.ai_type_red == "AlphaZero":
                return self.alpha_zero_ai_red
            else:
                return self.alpha_beta_ai_red
        else:
            if self.ai_type_black == "AlphaZero":
                return self.alpha_zero_ai_black
            else:
                return self.alpha_beta_ai_black

    def ai_move(self):
        """执行AI移动，优化响应速度"""
        if self.state.game_over or self.training or self.screen_state != "game":
            return

        current_ai = self.get_current_ai()
        # 异步获取AI移动，提高响应速度
        result = [None]

        def get_ai_move():
            result[0] = current_ai.get_move(self.state)

        thread = threading.Thread(target=get_ai_move)
        thread.start()
        thread.join(timeout=1.0)  # 设置超时，防止长时间无响应

        ai_move = result[0]
        if ai_move:
            self.state.apply_move(ai_move)
            # 检查游戏结果
            self.check_game_result()
            return current_ai.move_delay  # 返回AI思考延迟
        return 300  # 更短的默认延迟

    def toggle_red_ai_type(self):
        """切换红方AI类型"""
        self.ai_type_red = "AlphaBeta" if self.ai_type_red == "AlphaZero" else "AlphaZero"
        self.load_ais()
        # 更新按钮文本
        for button in self.buttons["ai_selection"]:
            if button["action"] == "toggle_red_ai_type":
                button["text"] = f"红方AI: {self.ai_type_red}"

    def toggle_black_ai_type(self):
        """切换黑方AI类型"""
        self.ai_type_black = "AlphaBeta" if self.ai_type_black == "AlphaZero" else "AlphaZero"
        self.load_ais()
        # 更新按钮文本
        for button in self.buttons["ai_selection"]:
            if button["action"] == "toggle_black_ai_type":
                button["text"] = f"黑方AI: {self.ai_type_black}"

    def cycle_difficulty(self):
        """循环切换难度等级"""
        difficulties = ["easy", "medium", "hard"]
        try:
            current_idx = difficulties.index(self.difficulty)
        except ValueError:
            current_idx = 0  # 默认使用第一个难度
        new_idx = (current_idx + 1) % len(difficulties)
        self.difficulty = difficulties[new_idx]

        difficulty_names = {"easy": "简单", "medium": "中等", "hard": "困难"}
        # 更新按钮文本
        for button in self.buttons["ai_selection"]:
            if button["action"] == "cycle_difficulty":
                button["text"] = f"难度: {difficulty_names[self.difficulty]}"

        self.load_ais()  # 重新加载AI以应用新难度

    def reset(self):
        self.state = GameState()
        self.selected_piece = None
        self.valid_moves = []
        self.ai_move_timer = 0
        self.show_result_dialog = False  # 重置结果对话框
        self.show_hint = False  # 重置提示
        self.best_hint_move = None

    def undo(self):
        if len(self.state.move_history) > 0:  # 修复判断条件
            # 只撤销最后一步
            moves = self.state.move_history[:-1] if len(self.state.move_history) > 1 else []
            self.reset()
            for move in moves:
                self.state.apply_move(move)
        elif len(self.state.move_history) == 1:
            self.reset()

    def train(self):
        if not self.training:
            self.training = True
            self.training_progress = 0
            self.training_cancelled = False
            threading.Thread(target=self._train_thread, daemon=True).start()
        else:
            self.training_cancelled = True
            self.training = False

    def _train_thread(self):
        self.trainer.alpha_zero_ai = AlphaZeroAI()
        self.training_progress = 0

        # 自我对弈收集数据
        self.trainer.self_play(num_games=10, progress_callback=self.update_training_progress)

        if self.training_cancelled:
            self.training = False
            return

        # 训练模型
        avg_loss = self.trainer.train(progress_callback=self.update_training_progress)

        # 记录最终训练结果
        final_acc = self.trainer.training_stats["val_accuracy"][-1] if self.trainer.training_stats[
            "val_accuracy"] else 0.0
        final_policy_loss = self.trainer.training_stats["policy_loss"][-1] if self.trainer.training_stats[
            "policy_loss"] else 0.0
        final_value_loss = self.trainer.training_stats["value_loss"][-1] if self.trainer.training_stats[
            "value_loss"] else 0.0
        # 原代码
        self.trainer.data_processor.record_training_result(avg_loss, final_policy_loss, final_value_loss, final_acc,
                                                           "最终")
        # 修改为
        self.trainer.data_processor.record_training_result(avg_loss, final_policy_loss, final_value_loss, final_acc,
                                                           -1)  # 用-1标记最终结果

        # 保存模型
        torch.save(self.trainer.model.state_dict(), "chess_model.pth")

        # 更新AI模型列表和AI实例
        self.ai_models = self.get_available_ai_models()
        self.load_ais()

        self.training = False
        self.training_progress = 0

    def update_training_progress(self, progress):
        self.training_progress = progress

    def toggle_training_info(self):
        """切换训练信息显示状态"""
        self.show_training_info = not self.show_training_info
        if self.show_training_info:
            self.scroll_offsets["training_info"] = 0  # 重置滚动位置

    def toggle_training_data(self):
        """切换训练数据显示状态"""
        self.show_training_data = not self.show_training_data
        if self.show_training_data:
            self.scroll_offsets["training_data"] = 0  # 重置滚动位置

    def toggle_training_history(self):
        """切换训练历史显示状态"""
        self.show_training_history = not self.show_training_history
        if self.show_training_history:
            self.scroll_offsets["training_history"] = 0  # 重置滚动位置

    def hint(self):
        """改进提示功能，确保显示最佳走法"""
        if not hasattr(self, 'hint_ai'):  # 提前初始化
            self.hint_ai = AlphaBetaAI(difficulty="medium")
        if not self.state.game_over and self.is_human_turn():
            self.show_hint = not self.show_hint  # 切换提示显示状态

            if not self.show_hint:
                self.best_hint_move = None
                return

            # 如果有选中的棋子，只为该棋子寻找最佳走法
            if self.selected_piece:
                current_pos = self.selected_piece.position
                legal_moves = self.state.get_legal_moves_for_piece(self.selected_piece)

                if legal_moves:
                    # 优先提示吃子
                    for move in legal_moves:
                        if self.state.get_piece_at(move):
                            self.best_hint_move = (current_pos, move)
                            return

                    # 使用AI寻找最佳走法
                    temp_ai = AlphaBetaAI(difficulty="medium")  # 使用中等难度快速计算
                    all_moves = self.state.get_legal_moves(self.state.current_player)

                    best_score = -float('inf')
                    best_move = None

                    for move in all_moves:
                        if move[0] == current_pos:
                            # 评估这个走法
                            new_state = self.state.copy()
                            if not hasattr(self, 'hint_ai'):
                                self.hint_ai = AlphaBetaAI(difficulty="medium")
                            temp_ai = self.hint_ai  # 复用实例
                            new_state.apply_move(move)
                            score = -temp_ai.alpha_beta(new_state, 2, -float('inf'), float('inf'), False)

                            if score > best_score:
                                best_score = score
                                best_move = move

                    self.best_hint_move = best_move
            else:
                # 如果没有选中棋子，提示应该移动哪个棋子
                temp_ai = self.hint_ai
                best_move = temp_ai.get_move(self.state)
                self.best_hint_move = best_move

                # 如果找到了最佳走法，选中相应的棋子
                if best_move:
                    from_pos, _ = best_move
                    piece = self.state.get_piece_at(from_pos)
                    if piece:
                        self.select_piece(piece)

    def set_human_vs_human(self):
        self.game_mode = "human_vs_human"
        self.reset()

    def set_human_vs_ai(self):
        self.game_mode = "human_vs_ai"
        self.reset()

    def set_ai_vs_ai(self):
        self.game_mode = "ai_vs_ai"
        self.reset()

    # 训练参数调整方法
    # 训练参数调整方法
    def decrease_lr(self, reason="未指定"):
        """
        降低学习率（乘以0.5，不低于最小值）
        :param reason: 学习率调整的原因（如"验证集损失上升"、"固定步数调整"等）
        """
        if not hasattr(self.trainer, 'optimizer'):
            print("警告：训练器未初始化优化器，无法调整学习率")
            return

        # 获取当前学习率（兼容多参数组，取第一个组）
        param_groups = self.trainer.optimizer.param_groups
        current_lr = param_groups[0]['lr']
        new_lr = max(0.00001, current_lr * 0.5)  # 最低学习率限制

        # 仅在学习率有变化时执行更新
        if new_lr != current_lr:
            # 同步更新所有参数组的学习率（原逻辑可能只更新了训练器参数，未同步优化器）
            for group in param_groups:
                group['lr'] = new_lr
            # 若训练器有参数记录，同步更新
            if hasattr(self.trainer, 'lr'):
                self.trainer.lr = new_lr

            # 详细日志：包含调整原因、新旧值
            print(f"学习率调整（原因：{reason}）: {current_lr:.6f} -> {new_lr:.6f}")
        else:
            print(f"学习率已达最小值 {current_lr:.6f}，无需调整")

    def increase_lr(self):
        current_lr = self.trainer.training_params["learning_rate"]
        new_lr = min(0.1, current_lr * 2)  # 最大学习率调整为10%
        self.trainer.set_training_params(lr=new_lr)

    def decrease_batch(self):
        current_batch = self.trainer.training_params["batch_size"]
        new_batch = max(4, current_batch // 2)
        self.trainer.set_training_params(batch_size=new_batch)

    def increase_batch(self):
        current_batch = self.trainer.training_params["batch_size"]
        new_batch = min(128, current_batch * 2)
        self.trainer.set_training_params(batch_size=new_batch)

    def decrease_epochs(self):
        current_epochs = self.trainer.training_params["epochs"]
        new_epochs = max(1, current_epochs - 5)
        self.trainer.set_training_params(epochs=new_epochs)

    def increase_epochs(self):
        current_epochs = self.trainer.training_params["epochs"]
        new_epochs = min(100, current_epochs + 5)
        self.trainer.set_training_params(epochs=new_epochs)
        # 若训练未开始，重置当前 epoch
        if self.trainer.training_stats["current_phase"] != "模型训练":
            self.trainer.training_stats["current_epoch"] = 0

    def save_training_params(self):
        """保存训练参数设置"""
        import json
        import os
        import time  # 添加此行
        params = self.trainer.training_params
        # 增加时间戳确保文件唯一性
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        params_path = f"training_params_{timestamp}.json"
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2)
        print(f"训练参数已保存至 {params_path}")
        self.back_to_game()

    # 子页面相关方法
    def show_help(self):
        """显示帮助页面"""
        self.screen_state = "help"

    def show_about(self):
        """显示关于页面"""
        self.screen_state = "about"

    # 屏幕状态切换方法
    def start_game(self):
        self.screen_state = "game"
        self.reset()
        self.game_history = []  # 新增：用于记录当前对局的完整数据

    def select_ai(self):
        self.screen_state = "ai_selection"
        # 更新AI选择界面的按钮文本
        for button in self.buttons["ai_selection"]:
            if button["action"] == "toggle_red_ai_type":
                button["text"] = f"红方AI: {self.ai_type_red}"
            elif button["action"] == "toggle_black_ai_type":
                button["text"] = f"黑方AI: {self.ai_type_black}"
            elif button["action"] == "cycle_difficulty":
                difficulty_names = {"easy": "简单", "medium": "中等", "hard": "困难"}
                button["text"] = f"难度: {difficulty_names[self.difficulty]}"

    def training_settings(self):
        """打开训练设置界面"""
        self.screen_state = "training_settings"

    def back_to_game(self):
        self.screen_state = "game"

    def back_to_start(self):
        self.screen_state = "start_menu"

    def quit_game(self):
        pygame.quit()
        sys.exit()

    # 处理游戏结果显示
    def check_game_result(self):
        if self.state.game_over:
            # 明确显示获胜方
            winner_text = "红方" if self.state.winner == 'red' else "黑方"
            self.result_message = f"{winner_text}获胜！"

            # 人机对战补充玩家胜负信息
            if self.game_mode == "human_vs_ai":
                # 简化胜负判断逻辑，使用更清晰的条件表达式
                is_human_win = (self.state.winner == 'red' and self.ai_type_red == "human") or \
                               (self.state.winner == 'black' and self.ai_type_black == "human")
                self.result_message += "\n恭喜你赢得了比赛！" if is_human_win else "\n再接再厉，下次一定能赢！"

            # 机机对战补充对战信息
            elif self.game_mode == "ai_vs_ai":
                self.result_message += f"\n对战组合: {self.ai_type_red} vs {self.ai_type_black}"

            self.show_result_dialog = True
            # 规范游戏结束日志格式，增加时间戳便于追溯
            import time
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n===== 游戏结束 [{current_time}] =====")
            print(f"获胜方: {winner_text}")
            print(f"对战组合: {self.ai_type_red} vs {self.ai_type_black}")
            print(f"游戏ID: {self.state.game_id}")
            print(f"总步数: {self.state.move_count}")
            print(f"===================\n")

            # 仅在游戏结束时收集数据（原逻辑可能在非结束状态触发，移到内部更合理）
            if hasattr(self, 'data_processor'):  # 先检查数据处理器是否初始化
                if hasattr(self, 'game_history') and self.game_history:
                    self.data_processor.collect_data(self.game_history)
                    print(f"本局数据已收集，共{len(self.game_history)}条记录")
                else:
                    print("警告：本局无有效游戏数据可收集（game_history为空或未初始化）")
            else:
                print("警告：数据处理器未初始化，无法收集游戏数据")

    # 关闭结果对话框
    def close_result_dialog(self):

        if self.show_result_dialog:

            self.show_result_dialog = False  # 修复：应该关闭对话框而不是绘制
            # 新增：如果有对话框实例，调用关闭方法
        if hasattr(self, 'result_dialog') and self.result_dialog is not None:
            self.result_dialog.destroy()  # 或对应的关闭方法

    # 重新开始游戏（从结果对话框）
    def reset_after_result(self):
        self.show_result_dialog = False
        self.reset()

    def update(self):
        """游戏状态更新：处理AI自动移动及状态刷新"""
        # 过滤无需更新的场景（提前返回，减少嵌套）
        if self.training or self.screen_state != "game" or self.show_result_dialog or self.show_training_curve:
            return

        # 仅在游戏未结束且非人类回合时处理AI移动
        if not self.state.game_over and not self.is_human_turn():
            current_time = pygame.time.get_ticks()
            # 难度-延迟映射表，增加默认值说明
            difficulty_delays = {
                "easy": 800,  # 简单难度：800ms延迟
                "medium": 500,  # 中等难度：500ms延迟
                "hard": 300  # 困难难度：300ms延迟
            }
            # 获取延迟值，对未知难度增加警告
            base_delay = difficulty_delays.get(self.difficulty)
            if base_delay is None:
                base_delay = 500  # 默认延迟
                print(f"警告：未知难度等级 '{self.difficulty}'，使用默认延迟{base_delay}ms")

            # 检查延迟是否达标，执行AI移动
            if current_time - self.ai_move_timer > base_delay:
                # 二次确认游戏状态（避免极端情况下的重复移动）
                if not self.state.game_over:
                    self.ai_move()
                    self.ai_move_timer = current_time  # 重置计时器
                else:
                    print("跳过AI移动：游戏已结束")

    def draw(self, surface):
        surface.fill(BACKGROUND)

        if self.show_training_history:
            self.draw_training_history(surface)

        if self.show_training_curve:
            # 绘制训练曲线图
            curve_surf, self.training_curve_warned = draw_training_curve(self.trainer, self.training_curve_warned)
            if curve_surf:
                # 居中显示曲线图
                curve_rect = curve_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))
                surface.blit(curve_surf, curve_rect)

            # 绘制返回按钮
            back_button = pygame.Rect(WIDTH - 200, 30, 100, 40)
            pygame.draw.rect(surface, BLUE, back_button)
            pygame.draw.rect(surface, WHITE, back_button, 2)
            text = font_small.render("返回", True, WHITE)
            text_rect = text.get_rect(center=back_button.center)
            surface.blit(text, text_rect)
            pygame.display.flip()
            return

        if self.screen_state == "start_menu":
            self.draw_start_menu(surface)
        elif self.screen_state == "ai_selection":
            self.draw_ai_selection(surface)
        elif self.screen_state == "training_settings":
            self.draw_training_settings(surface)
        elif self.screen_state == "game":
            self.draw_game(surface)
        elif self.screen_state == "help":
            self.draw_help(surface)
        elif self.screen_state == "about":
            self.draw_about(surface)

        # 绘制结果对话框（如果显示）
        if self.show_result_dialog:
            self.draw_result_dialog(surface)

        pygame.display.flip()

    def draw_result_dialog(self, surface):
        # 创建半透明遮罩
        s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        s.fill(TRANSPARENT_BLACK)
        surface.blit(s, (0, 0))

        # 绘制对话框
        dialog_width = 500
        dialog_height = 350
        dialog_x = WIDTH // 2 - dialog_width // 2
        dialog_y = HEIGHT // 2 - dialog_height // 2

        pygame.draw.rect(surface, WHITE, (dialog_x, dialog_y, dialog_width, dialog_height), border_radius=10)
        pygame.draw.rect(surface, BLUE, (dialog_x, dialog_y, dialog_width, dialog_height), 3, border_radius=10)

        # 绘制标题
        title = font_large.render("游戏结束", True, PURPLE)
        title_rect = title.get_rect(center=(WIDTH // 2, dialog_y + 50))
        surface.blit(title, title_rect)

        # 绘制结果文本 - 支持多行显示
        lines = self.result_message.split('\n')
        for i, line in enumerate(lines):
            text = font_medium.render(line, True, BLUE if "获胜" in line else BLACK)
            text_rect = text.get_rect(center=(WIDTH // 2, dialog_y + 120 + i * 50))
            surface.blit(text, text_rect)

        # 绘制按钮
        for button in self.buttons["result_dialog"]:
            pygame.draw.rect(surface, button["color"], button["rect"], border_radius=8)
            pygame.draw.rect(surface, WHITE, button["rect"], 2, border_radius=8)
            text = font_medium.render(button["text"], True, WHITE)
            text_rect = text.get_rect(center=button["rect"].center)
            surface.blit(text, text_rect)

    def draw_start_menu(self, surface):
        # 绘制标题
        title = font_large.render("中国象棋", True, RED)
        title_rect = title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 200))
        surface.blit(title, title_rect)

        subtitle = font_medium.render("选择游戏模式", True, BLACK)
        subtitle_rect = subtitle.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 150))
        surface.blit(subtitle, subtitle_rect)

        # 绘制按钮
        for button in self.buttons["start_menu"]:
            pygame.draw.rect(surface, button["color"], button["rect"], border_radius=8)
            pygame.draw.rect(surface, WHITE, button["rect"], 2, border_radius=8)
            text = font_medium.render(button["text"], True, WHITE)
            text_rect = text.get_rect(center=button["rect"].center)
            surface.blit(text, text_rect)

    def draw_ai_selection(self, surface):
        # 标题
        title = font_large.render("选择AI模型和难度", True, BLUE)
        title_rect = title.get_rect(center=(WIDTH // 2, 100))
        surface.blit(title, title_rect)

        # 红方区域
        red_label = font_medium.render("红方AI模型:", True, BLACK)
        surface.blit(red_label, (400, 130))
        # 红方模型列表起始 y
        red_y_start = 210

        # 黑方区域
        black_label = font_medium.render("黑方AI模型:", True, BLACK)
        surface.blit(black_label, (400, 350))
        # 黑方模型列表起始 y
        black_y_start = 430

        # 红方列表
        y_pos = red_y_start
        for i, (model_id, model_name) in enumerate(self.ai_models.items()):
            rect = pygame.Rect(400, y_pos + i * 50, 450, 40)
            color = GREEN if model_id == self.selected_model_red else LIGHT_BLUE
            pygame.draw.rect(surface, color, rect, border_radius=5)
            pygame.draw.rect(surface, WHITE, rect, 2, border_radius=5)
            text = font_small.render(model_name, True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            surface.blit(text, text_rect)

        # 黑方列表
        y_pos = black_y_start
        for i, (model_id, model_name) in enumerate(self.ai_models.items()):
            rect = pygame.Rect(400, y_pos + i * 50, 450, 40)
            color = GREEN if model_id == self.selected_model_black else LIGHT_BLUE
            pygame.draw.rect(surface, color, rect, border_radius=5)
            pygame.draw.rect(surface, WHITE, rect, 2, border_radius=5)
            text = font_small.render(model_name, True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            surface.blit(text, text_rect)

        # 绘制按钮
        for button in self.buttons["ai_selection"]:
            pygame.draw.rect(surface, button["color"], button["rect"], border_radius=5)
            pygame.draw.rect(surface, WHITE, button["rect"], 2, border_radius=5)
            text = font_small.render(button["text"], True, WHITE)
            text_rect = text.get_rect(center=button["rect"].center)
            surface.blit(text, text_rect)

    def draw_training_settings(self, surface):
        """训练设置界面"""
        # 绘制标题
        title = font_large.render("训练参数设置", True, PURPLE)
        title_rect = title.get_rect(center=(WIDTH // 2, 60))
        surface.blit(title, title_rect)

        # 增加说明文本
        desc = font_small.render("调整AI训练参数，影响训练效果和速度", True, BLACK)
        desc_rect = desc.get_rect(center=(WIDTH // 2, 120))
        surface.blit(desc, desc_rect)

        # 显示当前参数
        params = self.trainer.training_params

        # 参数区域布局设置
        param_start_y = 160
        param_spacing = 180
        text_to_button_spacing = 20
        button_width = 120
        button_height = 50

        # 学习率设置区块
        lr_label = font_medium.render("学习率 :", True, BLACK)
        surface.blit(lr_label, (WIDTH // 2 - 260, param_start_y))

        lr_text = font_medium.render(f"{params['learning_rate']:.6f}", True, BLACK)
        lr_text_x = WIDTH // 2 - 10
        surface.blit(lr_text, (lr_text_x, param_start_y))

        # 学习率按钮
        lr_dec_btn = next((b for b in self.buttons["training_settings"] if b["action"] == "decrease_lr"), None)
        lr_inc_btn = next((b for b in self.buttons["training_settings"] if b["action"] == "increase_lr"), None)
        lr_dec_btn["rect"] = pygame.Rect(0, 0, button_width, button_height)
        lr_inc_btn["rect"] = pygame.Rect(0, 0, button_width, button_height)
        lr_dec_btn["rect"].midtop = (lr_text_x - 70, param_start_y + text_to_button_spacing + 40)
        lr_inc_btn["rect"].midtop = (lr_text_x + 70, param_start_y + text_to_button_spacing + 40)
        if not lr_dec_btn or not lr_inc_btn:
            return  # 或添加错误处理

        # 批次大小设置区块
        batch_label = font_medium.render("批次大小:", True, BLACK)
        batch_y_pos = param_start_y + param_spacing
        surface.blit(batch_label, (WIDTH // 2 - 260, batch_y_pos))

        batch_text = font_medium.render(f"{params['batch_size']}", True, BLACK)
        batch_text_x = WIDTH // 2 - 10
        surface.blit(batch_text, (batch_text_x, batch_y_pos))

        # 批次大小按钮
        batch_dec_btn = [b for b in self.buttons["training_settings"] if b["action"] == "decrease_batch"][0]
        batch_inc_btn = [b for b in self.buttons["training_settings"] if b["action"] == "increase_batch"][0]
        batch_dec_btn["rect"] = pygame.Rect(0, 0, button_width, button_height)
        batch_inc_btn["rect"] = pygame.Rect(0, 0, button_width, button_height)
        batch_dec_btn["rect"].midtop = (batch_text_x - 70, batch_y_pos + text_to_button_spacing + 40)
        batch_inc_btn["rect"].midtop = (batch_text_x + 70, batch_y_pos + text_to_button_spacing + 40)

        # 训练轮次设置区块
        epochs_label = font_medium.render("训练轮次:", True, BLACK)
        epochs_y_pos = param_start_y + 2 * param_spacing
        surface.blit(epochs_label, (WIDTH // 2 - 260, epochs_y_pos))

        epochs_text = font_medium.render(f"{params['epochs']}", True, BLACK)
        epochs_text_x = WIDTH // 2 - 10
        surface.blit(epochs_text, (epochs_text_x, epochs_y_pos))

        # 训练轮次按钮
        epochs_dec_btn = [b for b in self.buttons["training_settings"] if b["action"] == "decrease_epochs"][0]
        epochs_inc_btn = [b for b in self.buttons["training_settings"] if b["action"] == "increase_epochs"][0]
        epochs_dec_btn["rect"] = pygame.Rect(0, 0, button_width, button_height)
        epochs_inc_btn["rect"] = pygame.Rect(0, 0, button_width, button_height)
        epochs_dec_btn["rect"].midtop = (epochs_text_x - 70, epochs_y_pos + text_to_button_spacing + 40)
        epochs_inc_btn["rect"].midtop = (epochs_text_x + 70, epochs_y_pos + text_to_button_spacing + 40)

        # 验证集比例设置区块
        val_label = font_medium.render("验证集比例:", True, BLACK)
        val_y_pos = param_start_y + 3 * param_spacing
        surface.blit(val_label, (WIDTH // 2 - 260, val_y_pos))

        val_text = font_medium.render(f"{params['validation_split']:.1f}", True, BLACK)
        val_text_x = WIDTH // 2 - 10
        surface.blit(val_text, (val_text_x, val_y_pos))

        # 底部按钮区域
        bottom_button_y = HEIGHT - 110
        back_btn = [b for b in self.buttons["training_settings"] if b["action"] == "back_to_game"][0]
        save_btn = [b for b in self.buttons["training_settings"] if b["action"] == "save_training_params"][0]
        back_btn["rect"] = pygame.Rect(0, 0, 180, 55)
        save_btn["rect"] = pygame.Rect(0, 0, 180, 55)
        back_btn["rect"].center = (WIDTH // 2 - 240, bottom_button_y)
        save_btn["rect"].center = (WIDTH // 2 + 240, bottom_button_y)

        # 绘制所有按钮
        for button in self.buttons["training_settings"]:
            if "rect" not in button:
                button["rect"] = pygame.Rect(0, 0, button_width, button_height)
            pygame.draw.rect(surface, button["color"], button["rect"], border_radius=14)
            pygame.draw.rect(surface, WHITE, button["rect"], 2, border_radius=14)
            text = font_medium.render(button["text"], True, WHITE)
            text_rect = text.get_rect(center=button["rect"].center)
            surface.blit(text, text_rect)

    def draw_game(self, surface):
        # 先绘制棋盘
        self.draw_board(surface)

        # 绘制有效移动标记
        for pos in self.valid_moves:
            x, y = pos
            center_x = MARGIN_X + y * GRID_SIZE
            center_y = MARGIN_Y + x * GRID_SIZE

            if self.state.get_piece_at(pos):
                pygame.draw.circle(surface, RED, (center_x, center_y), 10, 3)
            else:
                pygame.draw.circle(surface, GREEN, (center_x, center_y), 8)

        # 绘制提示
        if self.show_hint and self.best_hint_move:
            from_pos, to_pos = self.best_hint_move
            # 绘制起点高亮
            fx, fy = from_pos
            center_x = MARGIN_X + fy * GRID_SIZE
            center_y = MARGIN_Y + fx * GRID_SIZE
            pygame.draw.circle(surface, CYAN, (center_x, center_y), GRID_SIZE // 2 - 6, 4)

            # 绘制终点高亮
            tx, ty = to_pos
            center_x = MARGIN_X + ty * GRID_SIZE
            center_y = MARGIN_Y + tx * GRID_SIZE
            pygame.draw.circle(surface, GOLD, (center_x, center_y), 15, 3)

            # 绘制箭头指示
            start_x = MARGIN_X + fy * GRID_SIZE
            start_y = MARGIN_Y + fx * GRID_SIZE
            end_x = MARGIN_X + ty * GRID_SIZE
            end_y = MARGIN_Y + tx * GRID_SIZE

            # 绘制虚线箭头
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                dx_normalized = dx / length
                dy_normalized = dy / length

                # 缩短箭头，避免穿过棋子
                arrow_length = max(0, length - GRID_SIZE // 2)
                adjusted_end_x = start_x + dx_normalized * arrow_length
                adjusted_end_y = start_y + dy_normalized * arrow_length

                # 绘制虚线
                for i in range(0, int(arrow_length), 10):
                    segment_start_x = start_x + dx_normalized * i
                    segment_start_y = start_y + dy_normalized * i
                    segment_end_x = start_x + dx_normalized * min(i + 6, arrow_length)
                    segment_end_y = start_y + dy_normalized * min(i + 6, arrow_length)
                    pygame.draw.line(surface, GOLD,
                                     (segment_start_x, segment_start_y),
                                     (segment_end_x, segment_end_y), 3)

                # 绘制箭头头部
                angle = math.atan2(dy, dx)
                arrow_size = 10
                arrow1_x = adjusted_end_x - arrow_size * math.cos(angle - math.pi / 6)
                arrow1_y = adjusted_end_y - arrow_size * math.sin(angle - math.pi / 6)
                arrow2_x = adjusted_end_x - arrow_size * math.cos(angle + math.pi / 6)
                arrow2_y = adjusted_end_y - arrow_size * math.sin(angle + math.pi / 6)

                pygame.draw.line(surface, GOLD, (adjusted_end_x, adjusted_end_y), (arrow1_x, arrow1_y), 3)
                pygame.draw.line(surface, GOLD, (adjusted_end_x, adjusted_end_y), (arrow2_x, arrow2_y), 3)

        # 绘制棋子
        for piece in self.state.pieces:
            piece.draw(surface)

        # 绘制按钮
        self.draw_buttons(surface)

        # 绘制信息
        self.draw_info(surface)

        # 绘制训练信息面板（如果启用）
        if self.show_training_info:
            self.draw_training_info(surface)

        # 绘制训练数据面板（如果启用）
        if self.show_training_data:
            self.draw_training_data(surface)

        # 绘制训练历史面板（如果启用）
        if self.show_training_history:
            self.draw_training_history(surface)

        # 绘制训练曲线按钮
        pygame.draw.rect(surface, GREEN, TRAINING_CURVE_BUTTON)
        pygame.draw.rect(surface, WHITE, TRAINING_CURVE_BUTTON, 2)
        text = font_small.render("训练曲线", True, WHITE)
        text_rect = text.get_rect(center=TRAINING_CURVE_BUTTON.center)
        surface.blit(text, text_rect)

    def draw_board(self, surface):
        # 绘制棋盘背景
        pygame.draw.rect(surface, BOARD_COLOR,
                         (MARGIN_X - GRID_SIZE // 2, MARGIN_Y - GRID_SIZE // 2,
                          BOARD_WIDTH + GRID_SIZE, BOARD_HEIGHT + GRID_SIZE))

        # 绘制网格线
        for i in range(10):
            y = MARGIN_Y + i * GRID_SIZE
            pygame.draw.line(surface, BLACK,
                             (MARGIN_X, y),
                             (MARGIN_X + BOARD_WIDTH, y), 2)

        for j in range(9):
            x = MARGIN_X + j * GRID_SIZE
            pygame.draw.line(surface, BLACK,
                             (x, MARGIN_Y),
                             (x, MARGIN_Y + BOARD_HEIGHT), 2)

        # 楚河汉界
        river_rect = pygame.Rect(MARGIN_X, MARGIN_Y + 4 * GRID_SIZE,
                                 BOARD_WIDTH, GRID_SIZE)
        pygame.draw.rect(surface, (245, 228, 197), river_rect)
        river_text = font_medium.render("楚 河        汉 界", True, BLUE)
        text_rect = river_text.get_rect(center=river_rect.center)
        surface.blit(river_text, text_rect)

        # 九宫格斜线
        pygame.draw.line(surface, BLACK,
                         (MARGIN_X + 3 * GRID_SIZE, MARGIN_Y),
                         (MARGIN_X + 5 * GRID_SIZE, MARGIN_Y + 2 * GRID_SIZE), 2)
        pygame.draw.line(surface, BLACK,
                         (MARGIN_X + 5 * GRID_SIZE, MARGIN_Y),
                         (MARGIN_X + 3 * GRID_SIZE, MARGIN_Y + 2 * GRID_SIZE), 2)

        pygame.draw.line(surface, BLACK,
                         (MARGIN_X + 3 * GRID_SIZE, MARGIN_Y + 7 * GRID_SIZE),
                         (MARGIN_X + 5 * GRID_SIZE, MARGIN_Y + 9 * GRID_SIZE), 2)
        surface.blit(river_text, text_rect)  # 绘制"楚河汉界"文本

        # 九宫格斜线
        pygame.draw.line(surface, BLACK,
                         (MARGIN_X + 3 * GRID_SIZE, MARGIN_Y),
                         (MARGIN_X + 5 * GRID_SIZE, MARGIN_Y + 2 * GRID_SIZE), 2)
        pygame.draw.line(surface, BLACK,
                         (MARGIN_X + 5 * GRID_SIZE, MARGIN_Y),
                         (MARGIN_X + 3 * GRID_SIZE, MARGIN_Y + 2 * GRID_SIZE), 2)

        pygame.draw.line(surface, BLACK,
                         (MARGIN_X + 3 * GRID_SIZE, MARGIN_Y + 7 * GRID_SIZE),
                         (MARGIN_X + 5 * GRID_SIZE, MARGIN_Y + 9 * GRID_SIZE), 2)
        pygame.draw.line(surface, BLACK,
                         (MARGIN_X + 5 * GRID_SIZE, MARGIN_Y + 7 * GRID_SIZE),
                         (MARGIN_X + 3 * GRID_SIZE, MARGIN_Y + 9 * GRID_SIZE), 2)

        # 兵/卒位置标记
        for x, y in [(3, 0), (3, 2), (3, 4), (3, 6), (3, 8),
                     (6, 0), (6, 2), (6, 4), (6, 6), (6, 8)]:
            center_x = MARGIN_X + y * GRID_SIZE
            center_y = MARGIN_Y + x * GRID_SIZE
            pygame.draw.circle(surface, BLACK, (center_x, center_y), 5, 1)

        # 炮位置标记
        for x, y in [(2, 1), (2, 7), (7, 1), (7, 7)]:
            center_x = MARGIN_X + y * GRID_SIZE
            center_y = MARGIN_Y + x * GRID_SIZE
            pygame.draw.circle(surface, BLACK, (center_x, center_y), 5, 1)

    def draw_buttons(self, surface):
        for button in self.buttons["game"]:
            color = button["color"]
            if self.training and button["action"] == "train":
                color = DARK_RED  # 训练中改变按钮颜色

            pygame.draw.rect(surface, color, button["rect"], border_radius=8)
            pygame.draw.rect(surface, WHITE, button["rect"], 2, border_radius=8)
            text = font_small.render(button["text"], True, WHITE)
            text_rect = text.get_rect(center=button["rect"].center)
            surface.blit(text, text_rect)

        # 绘制训练进度条
        if self.training:
            progress_rect = pygame.Rect(20, HEIGHT - 80, 200, 20)
            pygame.draw.rect(surface, GRAY, progress_rect)
            pygame.draw.rect(surface, GREEN,
                             (progress_rect.x, progress_rect.y,
                              progress_rect.width * self.training_progress // 100, progress_rect.height))
            pygame.draw.rect(surface, WHITE, progress_rect, 2)

            progress_text = font_tiny.render(f"训练进度: {self.training_progress}%", True, WHITE)
            surface.blit(progress_text, (20, HEIGHT - 60))

    def draw_info(self, surface):
        # 左下角布局参数（边距和行高）
        left_margin = 20  # 左边缘距离
        bottom_margin = 20  # 下边缘距离
        line_height = 30  # 行间距

        # 显示当前玩家（最下方一行）
        player_text = f"当前回合: {'红方' if self.state.current_player == 'red' else '黑方'}"
        color = RED if self.state.current_player == 'red' else BLACK
        text = font_medium.render(player_text, True, color)
        # 定位：左下角起，底部边距处
        surface.blit(text, (left_margin, HEIGHT - bottom_margin))

        # 显示游戏模式（上移一行）
        mode_texts = {
            "human_vs_human": "人人对战",
            "human_vs_ai": "人机对战",
            "ai_vs_ai": "机机对战"
        }
        mode_text = f"游戏模式: {mode_texts[self.game_mode]}"
        text = font_small.render(mode_text, True, BLACK)
        surface.blit(text, (left_margin, HEIGHT - bottom_margin - line_height))

        # 显示AI类型（再上移两行）
        if self.game_mode in ["human_vs_ai", "ai_vs_ai"]:
            ai_red_text = f"红方AI: {self.ai_type_red} ({self.ai_models[self.selected_model_red]})"
            ai_black_text = f"黑方AI: {self.ai_type_black} ({self.ai_models[self.selected_model_black]})"

            text = font_tiny.render(ai_red_text, True, RED)
            surface.blit(text, (left_margin, HEIGHT - bottom_margin - 2 * line_height))

            text = font_tiny.render(ai_black_text, True, BLACK)
            surface.blit(text, (left_margin, HEIGHT - bottom_margin - 3 * line_height))

        # 显示回合数（最上方一行）
        move_count = len(self.state.move_history)
        move_text = f"回合数: {move_count // 2 + 1}"
        text = font_small.render(move_text, True, BLACK)
        surface.blit(text, (left_margin, HEIGHT - bottom_margin - 4 * line_height))





    def draw_training_info(self, surface):
        """绘制训练信息面板"""
        panel_rect = pygame.Rect(WIDTH - 350, 20, 320, 320)
        pygame.draw.rect(surface, TRANSPARENT_WHITE, panel_rect)
        pygame.draw.rect(surface, BLACK, panel_rect, 2)

        title = font_medium.render("训练状态", True, BLACK)
        surface.blit(title, (WIDTH - 330, 30))

        # 训练状态信息
        info = [
            f"训练状态: {'进行中' if self.training else '未开始'}",
            f"当前阶段: {self.trainer.training_stats.get('current_phase', '准备中')}",
            f"已玩对局: {self.trainer.training_stats.get('games_played', 0)}",
            f"平均对局长度: {self.trainer.training_stats.get('avg_game_length', 0.0):.1f}步",
            f"最近损失: {self.trainer.training_stats.get('recent_loss', [0.0])[-1]:.4f}" if self.trainer.training_stats.get('recent_loss') else "最近损失: N/A",
            f"策略损失: {self.trainer.training_stats.get('policy_loss', [0.0])[-1]:.4f}" if self.trainer.training_stats.get('policy_loss') else "策略损失: N/A",
            f"价值损失: {self.trainer.training_stats.get('value_loss', [0.0])[-1]:.4f}" if self.trainer.training_stats.get('value_loss') else "价值损失: N/A",
            f"验证准确率: {self.trainer.training_stats.get('val_accuracy', [0.0])[-1]:.2%}" if self.trainer.training_stats.get('val_accuracy') else "验证准确率: N/A",
            f"平均价值误差: {self.trainer.training_stats.get('avg_value_error', 0.0):.4f}" if self.trainer.training_stats.get('avg_value_error_list') else "平均价值误差: N/A",
            f"当前轮次: {self.trainer.training_stats.get('current_epoch', 0)}/{self.trainer.training_params.get('epochs', 0)}",
            f"已处理批次: {self.trainer.training_stats.get('batch_processed', 0)}/{self.trainer.training_stats.get('total_batches', 0)}",
            f"最后训练时间: {self.trainer.training_stats.get('last_train_time', '未训练')}"
        ]

        # 计算最大滚动量
        self.max_scroll["training_info"] = max(0, len(info) * 25 - 270)

        # 绘制信息文本
        y_offset = 70 - self.scroll_offsets["training_info"]
        for line in info:
            if 60 < y_offset < 320:  # 只绘制可见区域内的文本
                text = font_tiny.render(line, True, BLACK)
                surface.blit(text, (WIDTH - 330, y_offset))
            y_offset += 25

    def draw_training_data(self, surface):
        """绘制训练数据面板"""
        panel_rect = pygame.Rect(WIDTH - 350, 360, 320, 280)
        pygame.draw.rect(surface, TRANSPARENT_WHITE, panel_rect)
        pygame.draw.rect(surface, BLACK, panel_rect, 2)

        title = font_medium.render("训练数据统计", True, BLACK)
        surface.blit(title, (WIDTH - 330, 370))

        # 数据统计信息
        data_stats = self.trainer.data_processor.data_stats
        info = [
            f"总样本数: {data_stats.get('total_samples', 0)}",
            f"原始样本: {data_stats.get('raw_samples', 0)}",
            f"处理样本: {data_stats.get('processed_samples', 0)}",
            f"训练样本: {data_stats.get('train_samples', 0)}",
            f"验证样本: {data_stats.get('val_samples', 0)}",
            f"平均对局长度: {data_stats.get('avg_game_length', 0.0):.1f}步",
            f"数据质量: {data_stats.get('data_quality', 0)}%",
            f"最后处理时间: {data_stats.get('last_processed', '未处理')}",
            f"处理进度: {data_stats.get('processing_progress', 0)}%"
        ]

        # 计算最大滚动量
        self.max_scroll["training_data"] = max(0, len(info) * 25 - 230)

        # 绘制信息文本
        y_offset = 410 - self.scroll_offsets["training_data"]
        for line in info:
            if 360 < y_offset < 620:  # 只绘制可见区域内的文本
                text = font_tiny.render(line, True, BLACK)
                surface.blit(text, (WIDTH - 330, y_offset))
            y_offset += 25

    def draw_training_history(self, surface):
        """绘制训练历史面板"""
        panel_rect = pygame.Rect(WIDTH - 350, 660, 320, 200)
        pygame.draw.rect(surface, TRANSPARENT_WHITE, panel_rect)
        pygame.draw.rect(surface, BLACK, panel_rect, 2)

        title = font_medium.render("最近训练结果", True, BLACK)
        surface.blit(title, (WIDTH - 330, 670))

        # 获取最近的训练历史
        history = self.trainer.data_processor.data_stats.get("training_history", [])
        # 只显示最近5条记录
        recent_history = history[-5:] if len(history) > 5 else history

        # 历史记录信息
        info = []
        for item in reversed(recent_history):
            epoch = item.get('epoch', 'N/A')
            loss = item.get('loss', 0.0)
            acc = item.get('accuracy', 0.0)
            info.append(f"轮次 {epoch}: 损失={loss:.4f}, 准确率={acc:.2%}")

        # 计算最大滚动量
        self.max_scroll["training_history"] = max(0, len(info) * 25 - 150)

        # 绘制信息文本
        y_offset = 710 - self.scroll_offsets["training_history"]
        for line in info:
            if 660 < y_offset < 840:  # 只绘制可见区域内的文本
                text = font_tiny.render(line, True, BLACK)
                surface.blit(text, (WIDTH - 330, y_offset))
            y_offset += 25

    def draw_help(self, surface):
        """绘制帮助页面"""
        # 绘制标题
        title = font_large.render("游戏帮助", True, BLUE)
        title_rect = title.get_rect(center=(WIDTH // 2, 60))
        surface.blit(title, title_rect)

        # 帮助内容
        help_texts = [
            "中国象棋规则简介:",
            "1. 棋盘由9条竖线和10条横线组成，中间有楚河汉界。",
            "2. 红方先行，双方轮流移动棋子。",
            "3. 不同棋子有不同的走法:",
            "   - 车：横、竖方向任意格数，但不能跳过其他棋子",
            "   - 马：走“日”字，马腿被绊则不能移动",
            "   - 炮：横、竖方向走法同车，但吃子需隔一个棋子",
            "   - 相/象：走“田”字，不能过河，中心被堵则不能移动",
            "   - 仕/士：在九宫格内走斜线一格",
            "   - 帅/将：在九宫格内走一步，不能照面",
            "   - 兵/卒：未过河只能前进，过河后可左右移动",
            "4. 吃掉对方的将/帅获胜。",
            "",
            "游戏操作:",
            "- 点击要移动的棋子，再点击目标位置",
            "- 提示按钮可显示推荐走法",
            "- 悔棋按钮可撤销上一步操作",
            "- 可在设置中选择不同AI对手和难度"
        ]

        # 绘制帮助文本
        y_pos = 120
        for text in help_texts:
            if y_pos > HEIGHT - 150:  # 避免超出按钮区域
                break
            render_text = font_small.render(text, True, BLACK)
            surface.blit(render_text, (WIDTH // 2 - 350, y_pos))
            y_pos += 30

        # 绘制按钮
        for button in self.buttons["help"]:
            pygame.draw.rect(surface, button["color"], button["rect"], border_radius=8)
            pygame.draw.rect(surface, WHITE, button["rect"], 2, border_radius=8)
            text = font_medium.render(button["text"], True, WHITE)
            text_rect = text.get_rect(center=button["rect"].center)
            surface.blit(text, text_rect)

    def plot_terminal_training_curves(history):
        """在终端中绘制训练曲线（损失和准确率随epoch的变化）"""
        # 添加静态变量跟踪提示状态
        if not hasattr(plot_terminal_training_curves, "warned"):
            plot_terminal_training_curves.warned = False
        if len(history) < 2:
            print("数据不足（至少需要2个epoch），无法绘制曲线")
            plot_terminal_training_curves.warned = True
            return
            # 数据足够时重置提示状态
        plot_terminal_training_curves.warned = False
        # 检查数据完整性
        required_fields = ['epoch', 'loss', 'policy_loss', 'value_loss', 'accuracy']
        for item in history:
            if not all(field in item for field in required_fields):
                print(f"数据不完整: {item} 缺少必要字段")
                return

        # 提取数据
        epochs = [item['epoch'] for item in history]
        losses = [item['loss'] for item in history]
        policy_losses = [item['policy_loss'] for item in history]
        value_losses = [item['value_loss'] for item in history]
        accuracies = [item['accuracy'] for item in history]

        # 设置终端图表大小（根据终端窗口调整）
        plt.clf()  # 清空之前的图表
        plt.canvas_size(120, 40)  # 宽度120字符，高度40字符

        # 绘制损失曲线（第一个子图）
        plt.subplot(2, 1, 1)  # 2行1列，第1个子图
        plt.plot(epochs, losses, label="总损失", marker="*")
        plt.plot(epochs, policy_losses, label="策略损失", marker="o")
        plt.plot(epochs, value_losses, label="价值损失", marker="x")
        plt.title("损失随Epoch变化曲线")
        plt.xlabel("Epoch")
        plt.ylabel("损失值")
        plt.legend()

        # 绘制准确率曲线（第二个子图）
        plt.subplot(2, 1, 2)  # 2行1列，第2个子图
        plt.plot(epochs, accuracies, label="准确率", marker="s", color="green")
        plt.title("准确率随Epoch变化曲线")
        plt.xlabel("Epoch")
        plt.ylabel("准确率（0-1）")
        plt.ylim(0, 1)  # 准确率范围固定在0-1
        plt.legend()

        # 在终端显示图表
        plt.show()

    def draw_about(self, surface):
        """绘制关于页面"""
        # 绘制标题
        title = font_large.render("关于", True, PURPLE)
        title_rect = title.get_rect(center=(WIDTH // 2, 60))
        surface.blit(title, title_rect)

        # 关于内容
        about_texts = [
            "中国象棋 - 增强版",
            "版本: 1.0.0",
            "",
            "这是一个基于AI的中国象棋游戏，支持多种对战模式:",
            "- 人人对战",
            "- 人机对战",
            "- 机机对战",
            "",
            "AI类型:",
            "- AlphaBeta剪枝算法",
            "- AlphaZero深度强化学习",
            "",
            "特色功能:",
            "- 可训练自己的AI模型",
            "- 查看训练数据和历史",
            "- 调整训练参数优化AI",
            "- 游戏提示功能",
            "",
            "© 2023 中国象棋开发团队"
        ]

        # 绘制关于文本
        y_pos = 150
        for i, text in enumerate(about_texts):
            if i == 0:  # 标题行使用较大字体
                render_text = font_medium.render(text, True, BLACK)
                render_text.set_alpha(200)
                text_rect = render_text.get_rect(center=(WIDTH // 2, y_pos))
                surface.blit(render_text, text_rect)
                y_pos += 40
            else:
                if y_pos > HEIGHT - 150:  # 避免超出按钮区域
                    break
                render_text = font_small.render(text, True, BLACK)
                text_rect = render_text.get_rect(center=(WIDTH // 2, y_pos))
                surface.blit(render_text, text_rect)
                y_pos += 30

        # 绘制按钮
        for button in self.buttons["about"]:
            pygame.draw.rect(surface, button["color"], button["rect"], border_radius=8)
            pygame.draw.rect(surface, WHITE, button["rect"], 2, border_radius=8)
            text = font_medium.render(button["text"], True, WHITE)
            text_rect = text.get_rect(center=button["rect"].center)
            surface.blit(text, text_rect)


# 主游戏循环
def main():
    game = ChineseChessGame()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键点击
                    game.handle_click(event.pos)
                elif event.button == 4 or event.button == 5:  # 鼠标滚轮
                    y = 1 if event.button == 4 else -1
                    game.handle_scroll(y)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    if game.screen_state == "game":
                        game.back_to_start()
                    elif game.screen_state in ["ai_selection", "training_settings", "help", "about"]:
                        game.back_to_game()
                    else:
                        running = False

        game.update()
        game.draw(screen)
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
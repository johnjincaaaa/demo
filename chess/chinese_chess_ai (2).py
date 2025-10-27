import pygame
import sys
import numpy as np
import random
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque, defaultdict
from pygame.locals import *
import matplotlib

matplotlib.use("Agg")  # 使用Agg后端避免GUI冲突
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 初始化pygame
pygame.init()

# 游戏常量 - 扩大棋盘尺寸，调整布局
WIDTH, HEIGHT = 1400, 900  # 窗口适当增大以容纳训练曲线按钮
GRID_SIZE = 70  # 增大棋盘格子尺寸
BOARD_WIDTH = 8 * GRID_SIZE
BOARD_HEIGHT = 9 * GRID_SIZE
# 调整边距，确保棋盘居中且不被遮挡
MARGIN_X = (WIDTH - BOARD_WIDTH) // 2  # 居中棋盘
MARGIN_Y = (HEIGHT - BOARD_HEIGHT) // 2 + 20  # 顶部边距
FPS = 240  # 提高帧率以提高响应速度

# 按钮位置和尺寸
TRAINING_CURVE_BUTTON = pygame.Rect(WIDTH - 340, 30, 160, 40)  # 训练曲线按钮位置

# 颜色定义
BACKGROUND = (240, 217, 181)
BOARD_COLOR = (210, 180, 140)
DARK_WOOD = (160, 120, 80)
BLACK = (45, 45, 45)
WHITE = (255, 253, 245)
RED = (220, 60, 60)
DARK_RED = (180, 40, 40)
GREEN = (70, 180, 80)
YELLOW = (255, 200, 50)
BLUE = (65, 105, 225)
LIGHT_BLUE = (173, 216, 230)
PURPLE = (147, 112, 219)
ORANGE = (255, 140, 0)
GOLD = (255, 215, 0)
LIGHT_GREEN = (144, 238, 144)
GRAY = (160, 160, 160)
TRANSPARENT_BLACK = (0, 0, 0, 128)
TRANSPARENT_WHITE = (255, 255, 255, 240)
CYAN = (0, 255, 255)  # 青色定义

# 创建游戏窗口
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("中国象棋 - 增强版")
clock = pygame.time.Clock()

# 加载字体
try:
    font_large = pygame.font.SysFont("simhei", 48)
    font_medium = pygame.font.SysFont("simhei", 32)
    font_small = pygame.font.SysFont("simhei", 24)
    font_tiny = pygame.font.SysFont("simhei", 20)
    font_piece = pygame.font.SysFont("simhei", 42)  # 稍大的棋子字体
except:
    font_large = pygame.font.Font(None, 48)
    font_medium = pygame.font.Font(None, 32)
    font_small = pygame.font.Font(None, 24)
    font_tiny = pygame.font.Font(None, 20)
    font_piece = pygame.font.Font(None, 42)


# 神经网络定义（AlphaZero核心）
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(14, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)

        self.policy_conv = nn.Conv2d(256, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 10 * 9, 90 * 90)

        self.value_conv = nn.Conv2d(256, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(10 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 10 * 9)
        policy = self.policy_fc(policy)

        value = self.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 10 * 9)
        value = self.relu(self.value_fc1(value))
        value = self.tanh(self.value_fc2(value))

        return policy, value


# AlphaBeta剪枝算法（传统AI）
class AlphaBetaAI:
    def __init__(self, difficulty="medium"):
        self.set_difficulty(difficulty)
        self.piece_values = {
            '帅': 10000, '将': 10000,
            '车': 900, '马': 450, '炮': 450,
            '相': 200, '象': 200, '仕': 200, '士': 200,
            '兵': 100, '卒': 100
        }

    def set_difficulty(self, difficulty):
        """设置难度等级"""
        if difficulty == "easy":
            self.depth = 1
            self.move_delay = 300  # 减少延迟，提高响应速度
        elif difficulty == "hard":
            self.depth = 5
            self.move_delay = 1000  # 减少延迟，提高响应速度
        else:  # medium
            self.depth = 3
            self.move_delay = 600  # 减少延迟，提高响应速度

    def evaluate(self, game_state):
        score = 0
        for piece in game_state.pieces:
            sign = 1 if piece.color == 'red' else -1
            base_value = self.piece_values[piece.name]
            pos_value = self.get_position_value(piece)
            total_value = (base_value + pos_value) * sign
            score += total_value

        if game_state.is_in_check('black'):
            score += 300
        if game_state.is_in_check('red'):
            score -= 300

        return score

    def get_position_value(self, piece):
        x, y = piece.position
        pos_value = 0

        if piece.name == '车':
            if y in [0, 8] or y == 4:
                pos_value += 50
            if 3 <= x <= 6:
                pos_value += 30
        elif piece.name == '马':
            if (piece.color == 'red' and 4 <= x <= 5) or (piece.color == 'black' and 4 <= x <= 5):
                pos_value += 80
        elif piece.name in ['兵', '卒']:
            if (piece.color == 'red' and x <= 4) or (piece.color == 'black' and x >= 5):
                pos_value += 50
            if (piece.color == 'red' and x <= 2) or (piece.color == 'black' and x >= 7):
                pos_value += 100

        return pos_value

    def alpha_beta(self, game_state, depth, alpha, beta, maximizing_player):
        if depth == 0 or game_state.game_over:
            return self.evaluate(game_state)

        legal_moves = game_state.get_legal_moves('red' if maximizing_player else 'black')
        if not legal_moves:
            return -float('inf') if maximizing_player else float('inf')

        if maximizing_player:
            max_eval = -float('inf')
            for move in legal_moves:
                new_state = game_state.copy()
                new_state.apply_move(move)
                eval_score = self.alpha_beta(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                new_state = game_state.copy()
                new_state.apply_move(move)
                eval_score = self.alpha_beta(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def get_move(self, game_state):
        legal_moves = game_state.get_legal_moves(game_state.current_player)
        if not legal_moves:
            return None

        best_move = None
        best_score = -float('inf') if game_state.current_player == 'red' else float('inf')

        for move in legal_moves:
            new_state = game_state.copy()
            new_state.apply_move(move)
            score = self.alpha_beta(new_state, self.depth - 1, -float('inf'), float('inf'),
                                    game_state.current_player != 'red')

            if (game_state.current_player == 'red' and score > best_score) or \
                    (game_state.current_player == 'black' and score < best_score):
                best_score = score
                best_move = move

        return best_move if best_move else (legal_moves[0] if legal_moves else None)


# AlphaZero AI（基于神经网络+蒙特卡洛树搜索）
class AlphaZeroAI:
    def __init__(self, model_path=None, difficulty="medium"):
        self.model = ChessNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.set_difficulty(difficulty)
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.P = defaultdict(float)
        self.model_name = os.path.basename(model_path) if model_path else "默认模型"

    def set_difficulty(self, difficulty):
        """设置难度等级"""
        if difficulty == "easy":
            self.c_puct = 1.5
            self.num_simulations = 15  # 减少模拟次数，提高响应速度
            self.move_delay = 300  # 减少延迟
        elif difficulty == "hard":
            self.c_puct = 0.8
            self.num_simulations = 150  # 适当减少模拟次数
            self.move_delay = 1000  # 减少延迟
        else:  # medium
            self.c_puct = 1.0
            self.num_simulations = 60  # 适当减少模拟次数
            self.move_delay = 600  # 减少延迟

    def encode_state(self, game_state):
        state = torch.zeros(14, 10, 9, device=self.device)

        for piece in game_state.pieces:
            x, y = piece.position
            if piece.color == 'red':
                if piece.name == '帅':
                    idx = 0
                elif piece.name == '仕':
                    idx = 1
                elif piece.name == '相':
                    idx = 2
                elif piece.name == '车':
                    idx = 3
                elif piece.name == '马':
                    idx = 4
                elif piece.name == '炮':
                    idx = 5
                elif piece.name == '兵':
                    idx = 6
            else:
                if piece.name == '将':
                    idx = 7
                elif piece.name == '士':
                    idx = 8
                elif piece.name == '象':
                    idx = 9
                elif piece.name == '车':
                    idx = 10
                elif piece.name == '马':
                    idx = 11
                elif piece.name == '炮':
                    idx = 12
                elif piece.name == '卒':
                    idx = 13
            state[idx, x, y] = 1.0

        return state.unsqueeze(0)

    def get_state_key(self, game_state):
        pieces_key = []
        for piece in sorted(game_state.pieces, key=lambda p: (p.color, p.name, p.position)):
            pieces_key.append(f"{piece.color}:{piece.name}:{piece.position[0]},{piece.position[1]}")
        return "|".join(pieces_key) + f"|{game_state.current_player}"

    def move_to_idx(self, move):
        (x1, y1), (x2, y2) = move
        return (x1 * 9 + y1) * 90 + (x2 * 9 + y2)

    def mcts_search(self, game_state):
        if game_state.game_over:
            if game_state.winner == 'red':
                return 1.0
            elif game_state.winner == 'black':
                return -1.0
            else:
                return 0.0

        state_key = self.get_state_key(game_state)
        legal_moves = game_state.get_legal_moves(game_state.current_player)

        if not legal_moves:
            return 0.0

        if state_key not in self.P:
            with torch.no_grad():
                state_tensor = self.encode_state(game_state)
                policy, value = self.model(state_tensor)
                policy = policy.squeeze(0).cpu().numpy()
                value = value.item()

            move_probs = {}
            total_prob = 0.0
            for move in legal_moves:
                move_idx = self.move_to_idx(move)
                move_probs[move] = policy[move_idx]
                total_prob += policy[move_idx]

            if total_prob > 0:
                for move in move_probs:
                    move_probs[move] /= total_prob
            else:
                for move in legal_moves:
                    move_probs[move] = 1.0 / len(legal_moves)

            self.P[state_key] = move_probs
            self.N[state_key] = 0
            return value

        best_move = None
        best_ucb = -float('inf')
        total_visits = sum(self.N[(state_key, move)] for move in legal_moves)

        for move in legal_moves:
            q = self.Q[(state_key, move)]
            p = self.P[state_key][move]
            n = self.N[(state_key, move)]

            ucb = q + self.c_puct * p * math.sqrt(total_visits) / (1 + n)

            if (game_state.current_player == 'red' and ucb > best_ucb) or \
                    (game_state.current_player == 'black' and ucb < best_ucb):
                best_ucb = ucb
                best_move = move

        # 确保最佳移动有效
        if best_move is None or best_move not in legal_moves:
            best_move = random.choice(legal_moves)

        new_state = game_state.copy()
        new_state.apply_move(best_move)

        v = self.mcts_search(new_state)
        self.N[(state_key, best_move)] += 1
        self.Q[(state_key, best_move)] = (self.N[(state_key, best_move)] - 1) * self.Q[(state_key, best_move)] + v
        self.Q[(state_key, best_move)] /= self.N[(state_key, best_move)]
        self.N[state_key] += 1

        return v

    def get_move(self, game_state):
        legal_moves = game_state.get_legal_moves(game_state.current_player)
        if not legal_moves:
            return None

        for _ in range(self.num_simulations):
            self.mcts_search(game_state.copy())

        state_key = self.get_state_key(game_state)
        best_move = None
        max_visits = -1

        for move in legal_moves:
            if (state_key, move) in self.N and self.N[(state_key, move)] > max_visits:
                max_visits = self.N[(state_key, move)]
                best_move = move

        # 确保返回有效移动
        if best_move is None or best_move not in legal_moves:
            best_move = random.choice(legal_moves) if legal_moves else None

        self.Q.clear()
        self.N.clear()
        self.P.clear()

        return best_move


# 棋子类
class Piece:
    def __init__(self, name, color, position):
        self.name = name
        self.color = color
        self.position = position
        self.selected = False
        self.last_move_time = 0

    def draw(self, surface):
        x, y = self.position
        center_x = MARGIN_X + y * GRID_SIZE
        center_y = MARGIN_Y + x * GRID_SIZE

        radius = GRID_SIZE // 2 - 6
        if self.color == 'red':
            pygame.draw.circle(surface, RED, (center_x, center_y), radius)
            pygame.draw.circle(surface, DARK_RED, (center_x, center_y), radius, 2)
            text_color = GOLD
        else:
            pygame.draw.circle(surface, BLACK, (center_x, center_y), radius)
            pygame.draw.circle(surface, (80, 80, 80), (center_x, center_y), radius, 2)
            text_color = WHITE

        pygame.draw.circle(surface, WHITE, (center_x, center_y), radius, 2)

        text = font_piece.render(self.name, True, text_color)
        text_rect = text.get_rect(center=(center_x, center_y))
        surface.blit(text, text_rect)

        if self.selected:
            pygame.draw.circle(surface, YELLOW, (center_x, center_y), radius + 3, 3)


# 游戏状态类
class GameState:
    def __init__(self):
        self.initialize_board()
        self.current_player = 'red'
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.last_move = None
        self.game_id = None  # 新增：用于标识游戏

    def initialize_board(self):
        self.pieces = [
            # 红方棋子
            Piece('车', 'red', (9, 0)), Piece('马', 'red', (9, 1)), Piece('相', 'red', (9, 2)),
            Piece('仕', 'red', (9, 3)), Piece('帅', 'red', (9, 4)), Piece('仕', 'red', (9, 5)),
            Piece('相', 'red', (9, 6)), Piece('马', 'red', (9, 7)), Piece('车', 'red', (9, 8)),
            Piece('炮', 'red', (7, 1)), Piece('炮', 'red', (7, 7)),
            Piece('兵', 'red', (6, 0)), Piece('兵', 'red', (6, 2)), Piece('兵', 'red', (6, 4)),
            Piece('兵', 'red', (6, 6)), Piece('兵', 'red', (6, 8)),

            # 黑方棋子
            Piece('车', 'black', (0, 0)), Piece('马', 'black', (0, 1)), Piece('象', 'black', (0, 2)),
            Piece('士', 'black', (0, 3)), Piece('将', 'black', (0, 4)), Piece('士', 'black', (0, 5)),
            Piece('象', 'black', (0, 6)), Piece('马', 'black', (0, 7)), Piece('车', 'black', (0, 8)),
            Piece('炮', 'black', (2, 1)), Piece('炮', 'black', (2, 7)),
            Piece('卒', 'black', (3, 0)), Piece('卒', 'black', (3, 2)), Piece('卒', 'black', (3, 4)),
            Piece('卒', 'black', (3, 6)), Piece('卒', 'black', (3, 8))
        ]
        # 生成唯一游戏ID
        self.game_id = hash(tuple(sorted(p.position for p in self.pieces)))

    def get_piece_at(self, position):
        for piece in self.pieces:
            if piece.position == position:
                return piece
        return None

    def get_legal_moves(self, color):
        moves = []
        for piece in self.pieces:
            if piece.color == color:
                piece_moves = self.get_legal_moves_for_piece(piece)
                for move in piece_moves:
                    moves.append((piece.position, move))
        return moves

    def get_legal_moves_for_piece(self, piece):
        moves = []
        for x in range(10):
            for y in range(9):
                if self.is_valid_move(piece.position, (x, y)):
                    test_state = self.copy()
                    test_piece = test_state.get_piece_at(piece.position)
                    target_piece = test_state.get_piece_at((x, y))

                    if target_piece:
                        test_state.pieces.remove(target_piece)
                    test_piece.position = (x, y)

                    if not test_state.is_in_check(piece.color):
                        moves.append((x, y))
        return moves

    def is_valid_move(self, from_pos, to_pos):
        piece = self.get_piece_at(from_pos)
        if not piece:
            return False

        target_piece = self.get_piece_at(to_pos)
        if target_piece and target_piece.color == piece.color:
            return False

        sx, sy = from_pos
        ex, ey = to_pos

        if piece.name in ['车']:
            return self.is_valid_chariot_move(sx, sy, ex, ey)
        elif piece.name in ['马']:
            return self.is_valid_horse_move(sx, sy, ex, ey)
        elif piece.name in ['炮']:
            return self.is_valid_cannon_move(sx, sy, ex, ey)
        elif piece.name in ['相', '象']:
            return self.is_valid_elephant_move(sx, sy, ex, ey, piece.color)
        elif piece.name in ['仕', '士']:
            return self.is_valid_advisor_move(sx, sy, ex, ey, piece.color)
        elif piece.name in ['帅', '将']:
            return self.is_valid_king_move(sx, sy, ex, ey, piece.color)
        elif piece.name in ['兵', '卒']:
            return self.is_valid_pawn_move(sx, sy, ex, ey, piece.color)

        return False

    def is_valid_chariot_move(self, sx, sy, ex, ey):
        if sx != ex and sy != ey:
            return False

        if sx == ex:
            start, end = min(sy, ey), max(sy, ey)
            for y in range(start + 1, end):
                if self.get_piece_at((sx, y)):
                    return False
        else:
            start, end = min(sx, ex), max(sx, ex)
            for x in range(start + 1, end):
                if self.get_piece_at((x, sy)):
                    return False
        return True

    def is_valid_horse_move(self, sx, sy, ex, ey):
        dx, dy = abs(ex - sx), abs(ey - sy)
        if not ((dx == 1 and dy == 2) or (dx == 2 and dy == 1)):
            return False

        if dx == 1:
            block_y = (sy + ey) // 2
            if self.get_piece_at((sx, block_y)):
                return False
        else:
            block_x = (sx + ex) // 2
            if self.get_piece_at((block_x, sy)):
                return False
        return True

    def is_valid_cannon_move(self, sx, sy, ex, ey):
        if sx != ex and sy != ey:
            return False

        target_piece = self.get_piece_at((ex, ey))
        pieces_between = 0

        if sx == ex:
            start, end = min(sy, ey), max(sy, ey)
            for y in range(start + 1, end):
                if self.get_piece_at((sx, y)):
                    pieces_between += 1
        else:
            start, end = min(sx, ex), max(sx, ex)
            for x in range(start + 1, end):
                if self.get_piece_at((x, sy)):
                    pieces_between += 1

        if target_piece:
            return pieces_between == 1
        else:
            return pieces_between == 0

    def is_valid_elephant_move(self, sx, sy, ex, ey, color):
        dx, dy = abs(ex - sx), abs(ey - sy)
        if dx != 2 or dy != 2:
            return False

        if color == 'red' and ex < 5:
            return False
        if color == 'black' and ex > 4:
            return False

        block_x, block_y = (sx + ex) // 2, (sy + ey) // 2
        if self.get_piece_at((block_x, block_y)):
            return False
        return True

    def is_valid_advisor_move(self, sx, sy, ex, ey, color):
        dx, dy = abs(ex - sx), abs(ey - sy)
        if dx != 1 or dy != 1:
            return False

        if color == 'red':
            return 7 <= ex <= 9 and 3 <= ey <= 5
        else:
            return 0 <= ex <= 2 and 3 <= ey <= 5

    def is_valid_king_move(self, sx, sy, ex, ey, color):
        dx, dy = abs(ex - sx), abs(ey - sy)

        if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
            if color == 'red':
                return 7 <= ex <= 9 and 3 <= ey <= 5
            else:
                return 0 <= ex <= 2 and 3 <= ey <= 5

        if sx == ex:
            other_king = self.get_king_position('black' if color == 'red' else 'red')
            if other_king and other_king[1] == ey:
                start, end = min(sx, other_king[0]), max(sx, other_king[0])
                for x in range(start + 1, end):
                    if self.get_piece_at((x, ey)):
                        return False
                return True
        return False

    def is_valid_pawn_move(self, sx, sy, ex, ey, color):
        dx, dy = ex - sx, ey - sy

        if color == 'red':
            if dx > 0:
                return False
            if abs(dx) + abs(dy) != 1:
                return False
            if dx == 0 and sx >= 5:
                return False
        else:
            if dx < 0:
                return False
            if abs(dx) + abs(dy) != 1:
                return False
            if dx == 0 and sx <= 4:
                return False
        return True

    def get_king_position(self, color):
        king_name = '帅' if color == 'red' else '将'
        for piece in self.pieces:
            if piece.color == color and piece.name == king_name:
                return piece.position
        return None

    def is_in_check(self, color):
        king_pos = self.get_king_position(color)
        if not king_pos:
            return False

        opponent = 'black' if color == 'red' else 'red'
        for piece in self.pieces:
            if piece.color == opponent and self.is_valid_move(piece.position, king_pos):
                return True
        return False

    def copy(self):
        new_state = GameState()
        new_state.pieces = []
        for piece in self.pieces:
            new_piece = Piece(piece.name, piece.color, piece.position)
            new_state.pieces.append(new_piece)
        new_state.current_player = self.current_player
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        new_state.move_history = self.move_history.copy()
        new_state.last_move = self.last_move
        new_state.game_id = self.game_id  # 复制游戏ID
        return new_state

    def apply_move(self, move):
        # 防止None移动导致错误
        if move is None:
            return False
        try:
            from_pos, to_pos = move
        except (TypeError, ValueError):
            return False

        piece = self.get_piece_at(from_pos)
        if not piece:
            return False

        self.last_move = move

        # 吃子
        target_piece = self.get_piece_at(to_pos)
        if target_piece:
            self.pieces.remove(target_piece)
            if target_piece.name in ['将', '帅']:
                self.game_over = True
                self.winner = self.current_player

        # 移动棋子
        piece.position = to_pos
        self.move_history.append(move)

        # 切换玩家
        self.current_player = 'black' if self.current_player == 'red' else 'red'
        return True


# 数据收集与预处理类
class DataProcessor:
    def __init__(self):
        self.raw_data = []
        self.processed_data = []
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
        for state, _, _ in self.raw_data:
            game_id = state.game_id
            if game_id not in game_lengths:
                game_lengths[game_id] = 0
            game_lengths[game_id] += 1

        if game_lengths:
            total_length = sum(game_lengths.values())
            self.data_stats["avg_game_length"] = total_length / len(game_lengths)
        else:
            self.data_stats["avg_game_length"] = 0.0

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
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  # 更精确的时间
            "loss": loss,
            "policy_loss": policy_loss,  # 新增：策略损失
            "value_loss": value_loss,  # 新增：价值损失
            "accuracy": accuracy,
            "epoch": epoch
        }
        self.data_stats["training_history"].append(result)
        # 增加历史记录容量（从10条改为50条，便于观察趋势）
        if len(self.data_stats["training_history"]) > 50:
            self.data_stats["training_history"].pop(0)


# 训练类
class ChessTrainer:
    def __init__(self):
        self.model = ChessNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion_policy = nn.CrossEntropyLoss()
        self.criterion_value = nn.MSELoss()

        # 数据处理器
        self.data_processor = DataProcessor()

        # 训练参数
        self.training_params = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "validation_split": 0.2
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
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
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
                    # 若没有访问记录，为合法移动分配均匀概率
                    prob = 1.0 / len(legal_moves) if legal_moves else 0.0
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
            self.data_processor.collect_data(game_data)
            print(f"第{game_idx + 1}局生成数据量：{len(game_data)}条")  # 调试用

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

        if progress_callback and not getattr(progress_callback, 'training_cancelled', False):
            progress_callback(30)

        return len(self.data_processor.processed_data)

    def train(self, progress_callback=None):
        """训练模型，修正损失和准确率计算"""
        processed_data = self.data_processor.processed_data
        if len(processed_data) < self.training_params["batch_size"]:
            return 0.0  # 返回浮点数

        # 分割训练集和验证集 - 修复缩进
        split_idx = int(len(processed_data) * (1 - self.training_params["validation_split"]))
        train_data = processed_data[:split_idx]
        val_data = processed_data[split_idx:]
        print(f"训练集：{len(train_data)}，验证集：{len(val_data)}")  # 调试用

        # 更新数据统计中的训练集和验证集数量
        self.data_processor.data_stats["train_samples"] = len(train_data)
        self.data_processor.data_stats["val_samples"] = len(val_data)

        # 计算总批次用于进度显示
        self.training_stats["total_batches"] = len(train_data) // self.training_params["batch_size"]
        self.training_stats["current_phase"] = "模型训练"

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for epoch in range(self.training_params["epochs"]):
            if progress_callback and getattr(progress_callback, 'training_cancelled', False):
                break

            self.training_stats["current_epoch"] = epoch + 1
            self.training_stats["batch_processed"] = 0
            random.shuffle(train_data)

            batch_loss = 0.0
            batch_policy_loss = 0.0
            batch_value_loss = 0.0
            batches = 0

            for i in range(0, len(train_data), self.training_params["batch_size"]):
                batch = train_data[i:i + self.training_params["batch_size"]]
                if len(batch) < 2:
                    continue

                # 准备批次数据
                state_tensors, policy_tensors, value_tensors, _ = zip(*batch)
                state_tensors = torch.cat(state_tensors).to(self.device)
                policy_tensors = torch.stack(policy_tensors).to(self.device)
                value_tensors = torch.stack(value_tensors).to(self.device)  # 移除多余的unsqueeze

                # 前向传播计算损失
                self.model.train()
                self.optimizer.zero_grad()  # 每次迭代都清零梯度
                pred_policies, pred_values = self.model(state_tensors)
                loss_policy = self.criterion_policy(pred_policies, policy_tensors)
                loss_value = self.criterion_value(pred_values.squeeze(), value_tensors)  # 对齐维度
                loss = loss_policy + loss_value

                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 实时记录每批次损失（修复核心：确保面板能实时读取）
                self.training_stats["recent_loss"].append(loss.item())
                self.training_stats["policy_loss"].append(loss_policy.item())
                self.training_stats["value_loss"].append(loss_value.item())

                # 限制列表长度，避免内存溢出
                if len(self.training_stats["recent_loss"]) > 100:
                    self.training_stats["recent_loss"].pop(0)
                    self.training_stats["policy_loss"].pop(0)
                    self.training_stats["value_loss"].pop(0)

                batch_loss += loss.item()
                batch_policy_loss += loss_policy.item()
                batch_value_loss += loss_value.item()
                batches += 1
                self.training_stats["batch_processed"] = batches

                # 更新进度 - 修复进度计算
                if progress_callback:
                    epoch_progress = int((i / len(train_data)) * 70)
                    progress = 30 + int((epoch / self.training_params["epochs"]) * 70) + int(
                        epoch_progress / self.training_params["epochs"])
                    progress_callback(min(100, progress))

            if batches == 0:
                continue

            # 计算epoch平均损失
            avg_loss = batch_loss / batches
            avg_policy_loss = batch_policy_loss / batches
            avg_value_loss = batch_value_loss / batches
            total_loss += avg_loss
            total_policy_loss += avg_policy_loss
            total_value_loss += avg_value_loss

            # 验证集评估
            val_acc = self.evaluate(val_data)
            self.training_stats["val_accuracy"].append(val_acc)

            # 记录训练结果
            self.data_processor.record_training_result(avg_loss, avg_policy_loss, avg_value_loss, val_acc,
                                                       epoch + 1)
            print(
                f"轮次 {epoch + 1}：总损失={avg_loss:.4f}，策略损失={avg_policy_loss:.4f}，价值损失={avg_value_loss:.4f}")  # 调试用

            if progress_callback:
                progress = 30 + int((epoch + 1) / self.training_params["epochs"] * 70)
                progress_callback(progress)

        # 最终统计
        avg_total_loss = total_loss / self.training_params["epochs"] if self.training_params["epochs"] > 0 else 0.0
        self.training_stats["last_train_time"] = time.strftime("%Y-%m-%d %H:%M")
        self.training_stats["current_phase"] = "训练完成"
        return avg_total_loss

    def evaluate(self, val_data):
        if not val_data:
            return 0.0

        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        value_errors = []

        with torch.no_grad():
            for state_tensor, policy_tensor, value_tensor, _ in val_data:
                # 修复：确保输入有批次维度（添加unsqueeze(0)）
                pred_policy, pred_value = self.model(state_tensor.unsqueeze(0))  # 关键：增加批次维度

                # 评估策略预测准确率
                pred_move = torch.argmax(pred_policy).item()
                true_move = policy_tensor.item()  # policy_tensor是单个索引，直接取item()
                if pred_move == true_move:
                    correct_predictions += 1
                total_predictions += 1

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


def draw_training_curve(trainer):
    """绘制训练曲线图"""
    history = trainer.data_processor.data_stats["training_history"]
    if len(history) < 2:
        return None  # 数据不足，无法绘图

    # 创建图形
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

    pygame_surf = pygame.image.fromstring(raw_data, size, "RGB")
    return pygame_surf


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
        if os.path.exists("chess_model.pth"):
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
                    self.scroll_offsets["training_data"] = max(0, min(
                        self.max_scroll["training_data"],
                        self.scroll_offsets["training_data"] - y * scroll_speed
                    ))
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
        if self.selected_piece and to_pos in self.valid_moves:
            move = (self.selected_piece.position, to_pos)
            self.state.apply_move(move)

            self.selected_piece.selected = False
            self.selected_piece = None
            self.valid_moves = []

            # 检查游戏结果
            self.check_game_result()

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
        import threading
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
        current_idx = difficulties.index(self.difficulty)
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
        if len(self.state.move_history) >= 2:
            moves = self.state.move_history[:-2]
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
            import threading
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
        self.trainer.data_processor.record_training_result(avg_loss, final_policy_loss, final_value_loss, final_acc,
                                                           "最终")

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
                            new_state.apply_move(move)
                            score = -temp_ai.alpha_beta(new_state, 2, -float('inf'), float('inf'), False)

                            if score > best_score:
                                best_score = score
                                best_move = move

                    self.best_hint_move = best_move
            else:
                # 如果没有选中棋子，提示应该移动哪个棋子
                temp_ai = AlphaBetaAI(difficulty="medium")
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
    def decrease_lr(self):
        current_lr = self.trainer.training_params["learning_rate"]
        new_lr = max(0.00001, current_lr * 0.5)
        self.trainer.set_training_params(lr=new_lr)

    def increase_lr(self):
        current_lr = self.trainer.training_params["learning_rate"]
        new_lr = min(0.3, current_lr * 2)  # 最大学习率调整为30%
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

    def save_training_params(self):
        """保存训练参数设置"""
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

            # 对于人机对战，补充显示玩家胜负状态
            if self.game_mode == "human_vs_ai":
                if (self.state.winner == 'red' and self.ai_type_black != "human") or \
                        (self.state.winner == 'black' and self.ai_type_red != "human"):
                    self.result_message += "\n恭喜你赢得了比赛！"
                else:
                    self.result_message += "\n再接再厉，下次一定能赢！"

            # 机机对战显示额外信息
            elif self.game_mode == "ai_vs_ai":
                self.result_message += f"\n{self.ai_type_red} vs {self.ai_type_black}"

            self.show_result_dialog = True

    # 关闭结果对话框
    def close_result_dialog(self):
        self.show_result_dialog = False

    # 重新开始游戏（从结果对话框）
    def reset_after_result(self):
        self.show_result_dialog = False
        self.reset()

    def update(self):
        """游戏状态更新"""
        if self.training or self.screen_state != "game" or self.show_result_dialog or self.show_training_curve:
            return

        # 处理AI自动移动，减少延迟
        if not self.state.game_over and not self.is_human_turn():
            current_time = pygame.time.get_ticks()
            if current_time - self.ai_move_timer > 0:
                # 获取AI移动延迟并应用
                delay = self.ai_move()
                self.ai_move_timer = current_time + delay

    def draw(self, surface):
        surface.fill(BACKGROUND)

        if self.show_training_curve:
            # 绘制训练曲线图
            curve_surf = draw_training_curve(self.trainer)
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
        lr_dec_btn = [b for b in self.buttons["training_settings"] if b["action"] == "decrease_lr"][0]
        lr_inc_btn = [b for b in self.buttons["training_settings"] if b["action"] == "increase_lr"][0]
        lr_dec_btn["rect"] = pygame.Rect(0, 0, button_width, button_height)
        lr_inc_btn["rect"] = pygame.Rect(0, 0, button_width, button_height)
        lr_dec_btn["rect"].midtop = (lr_text_x - 70, param_start_y + text_to_button_spacing + 40)
        lr_inc_btn["rect"].midtop = (lr_text_x + 70, param_start_y + text_to_button_spacing + 40)

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
        # 显示当前玩家
        player_text = f"当前回合: {'红方' if self.state.current_player == 'red' else '黑方'}"
        color = RED if self.state.current_player == 'red' else BLACK
        text = font_medium.render(player_text, True, color)
        surface.blit(text, (WIDTH // 2 - 100, 20))

        # 显示游戏模式
        mode_texts = {
            "human_vs_human": "人人对战",
            "human_vs_ai": "人机对战",
            "ai_vs_ai": "机机对战"
        }
        mode_text = f"游戏模式: {mode_texts[self.game_mode]}"
        text = font_small.render(mode_text, True, BLACK)
        surface.blit(text, (WIDTH // 2 - 100, 60))

        # 显示AI类型
        if self.game_mode in ["human_vs_ai", "ai_vs_ai"]:
            ai_red_text = f"红方AI: {self.ai_type_red} ({self.ai_models[self.selected_model_red]})"
            ai_black_text = f"黑方AI: {self.ai_type_black} ({self.ai_models[self.selected_model_black]})"

            text = font_tiny.render(ai_red_text, True, RED)
            surface.blit(text, (WIDTH // 2 - 100, 90))

            text = font_tiny.render(ai_black_text, True, BLACK)
            surface.blit(text, (WIDTH // 2 - 100, 115))

        # 显示回合数
        move_count = len(self.state.move_history)
        move_text = f"回合数: {move_count // 2 + 1}"
        text = font_small.render(move_text, True, BLACK)
        surface.blit(text, (WIDTH // 2 - 100, 145))

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
            f"平均价值误差: {self.trainer.training_stats.get('avg_value_error', 0.0):.4f}",
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
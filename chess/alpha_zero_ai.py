import torch
import math
import random
from collections import defaultdict
from neural_network import ChessNet
from game_state import GameState

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
        self.difficulty = difficulty
        # 根据难度调整MCTS迭代次数等参数
        self.mcts_iterations = {"easy": 10, "medium": 50, "hard": 100}[difficulty]
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

    # alpha_zero_ai.py 中 AlphaZeroAI 类新增方法
    def compute_policy_value(self, game_state):
        """计算当前局面的策略分布（policy）和价值评估（value）"""
        # 实际逻辑应基于MCTS搜索结果，示例：
        mcts_result = self.mcts.search(game_state)  # MCTS搜索获取决策
        policy = mcts_result["policy"]  # 策略分布（各走法的概率）
        value = mcts_result["value"]  # 局面价值（如-1到1的评分）
        return policy, value

    # 在alpha_zero_ai.py的get_move方法中添加MCTS模拟次数显示
    def get_move(self, game_state):
        legal_moves = game_state.get_legal_moves(game_state.current_player)
        if not legal_moves:
            return None

        # 打印当前MCTS模拟次数
        print(f"[MCTS] 开始模拟... 模拟次数: {self.num_simulations}")
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

        # 替换原打印总访问量的代码
        total_visits = sum(self.N.get((state_key, move), 0) for move in legal_moves)
        print(f"[MCTS] 模拟完成 | 总访问量: {total_visits} | 最佳移动访问量: {max_visits}")

        self.Q.clear()
        self.N.clear()
        self.P.clear()

        return best_move
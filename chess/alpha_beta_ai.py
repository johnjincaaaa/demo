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
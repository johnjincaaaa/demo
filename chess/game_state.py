from piece import Piece

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
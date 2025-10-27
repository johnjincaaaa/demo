import pygame
from config import MARGIN_X, MARGIN_Y, GRID_SIZE, RED, DARK_RED, BLACK, WHITE, GOLD, YELLOW, font_piece

class Piece:
    def __init__(self, name, color, position):
        self.name = name
        self.color = color
        self.position = position
        self.selected = False
        self.last_move_time = 0

    def draw(self, screen):
        # 原有绘制逻辑（棋盘、棋子等）
        self.draw_board(screen)
        self.draw_pieces(screen)

        # 原有结果对话框绘制
        if self.show_result_dialog:
            self.draw_result_dialog(screen)
            # 当需要显示训练曲线时绘制
            if self.show_training_curve:
                curve_surf, self.training_curve_warned = draw_training_curve(self.trainer, self.training_curve_warned)
                if curve_surf:
                    # 绘制曲线到屏幕（示例位置：右上角）
                    screen.blit(curve_surf, (screen.get_width() - 600, 20))
                else:
                    # 数据不足时显示提示文字
                    font = pygame.font.SysFont("SimHei", 24)
                    tip_text = font.render("训练数据不足，无法显示曲线（至少需要2轮训练）", True, (255, 0, 0))
                    screen.blit(tip_text, (50, 50))

        # 新增：绘制左下角状态信息
        self.draw_status_info(screen)

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
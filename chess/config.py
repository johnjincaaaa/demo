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
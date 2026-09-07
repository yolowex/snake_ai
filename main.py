"""
main.py
=======
A polished Snake game built on top of `snake_engine.py`.

- All game logic (movement, collisions, food, scoring) comes from
  `SnakeGame` in snake_engine.py -- this file is purely presentation
  and input handling.
- Two control modes, switchable at any time with on-screen buttons:
    * Human Control  -- arrow keys / WASD
    * Neural Network -- a small MLP that reads the engine's 22-value
      state vector and picks an action. Loads trained weights from
      best_weights.npz when available.

Run:
    pip install pygame-ce numpy
    python main.py
"""

from __future__ import annotations
import math
import sys
import os

import numpy as np
import pygame

try:
    import tkinter as tk
    from tkinter import messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from snake_engine import (
    SnakeGame,
    SnakeConfig,
    ACTION_STRAIGHT,
    ACTION_RIGHT,
    ACTION_LEFT,
)

# ---------------------------------------------------------------------------
# Layout / config
# ---------------------------------------------------------------------------

GRID_W, GRID_H = 20, 20
CELL = 28
BOARD_W, BOARD_H = GRID_W * CELL, GRID_H * CELL
SIDEBAR_W = 280
MARGIN = 18

SCREEN_W = BOARD_W + SIDEBAR_W + MARGIN * 3
SCREEN_H = BOARD_H + MARGIN * 2
BOARD_X, BOARD_Y = MARGIN, MARGIN

FPS = 60
HUMAN_MOVES_PER_SEC = 9.0
AI_MOVES_PER_SEC = 12.0

# -- palette (dark, "arcade-neon" theme) -----------------------------------

BG_TOP = (14, 16, 28)
BG_BOTTOM = (22, 12, 34)
PANEL = (26, 28, 44)
PANEL_BORDER = (54, 58, 88)
GRID_LINE = (36, 39, 60)
BOARD_BG_A = (20, 22, 36)
BOARD_BG_B = (24, 26, 42)

SNAKE_HEAD = (110, 245, 170)
SNAKE_BODY_A = (60, 200, 140)
SNAKE_BODY_B = (34, 140, 105)
SNAKE_OUTLINE = (12, 40, 30)

FOOD_CORE = (255, 90, 110)
FOOD_GLOW = (255, 90, 110)

TEXT_MAIN = (235, 236, 245)
TEXT_DIM = (150, 154, 178)
ACCENT = (124, 156, 255)
ACCENT_2 = (255, 176, 90)

BTN_IDLE = (40, 43, 66)
BTN_HOVER = (54, 58, 88)
BTN_ACTIVE_HUMAN = (58, 140, 255)
BTN_ACTIVE_AI = (255, 130, 60)
BTN_BORDER = (70, 74, 104)

GAMEOVER_OVERLAY = (10, 8, 16, 195)

# direction indices used by the engine: 0=up, 1=right, 2=down, 3=left
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ABS_KEYS = {
    pygame.K_UP: UP, pygame.K_w: UP,
    pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
    pygame.K_DOWN: DOWN, pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT, pygame.K_a: LEFT,
}


def absolute_to_relative(current_dir: int, target_dir: int):
    """Convert an absolute heading request into the engine's relative
    action space. Returns None for a 180-degree reversal (ignored, since
    that's an instant illegal self-collision anyway)."""
    if target_dir == current_dir:
        return ACTION_STRAIGHT
    if (current_dir + 1) % 4 == target_dir:
        return ACTION_RIGHT
    if (current_dir - 1) % 4 == target_dir:
        return ACTION_LEFT
    return None  # would be a reversal


# ---------------------------------------------------------------------------
# Neural Network agent (matches train_snake.py architecture)
# ---------------------------------------------------------------------------

class NeuralNetworkAgent:
    """
    2-layer MLP (22 -> 64 -> 3) with ReLU hidden activation.
    Matches the architecture and weight layout produced by train_snake.py.
    """

    def __init__(self, input_size: int = 22, hidden_size: int = 64,
                 output_size: int = 3, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (input_size, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = rng.normal(0, 0.5, (hidden_size, output_size)).astype(np.float32)
        self.b2 = np.zeros(output_size, dtype=np.float32)
        self.trained = False

    def act(self, state: np.ndarray) -> int:
        # ReLU to match the trained DQN
        h = np.maximum(0.0, state @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return int(np.argmax(logits))

    def load_weights(self, path: str):
        """Load trained weights from an .npz file with keys W1,b1,W2,b2."""
        data = np.load(path)
        self.W1 = data["W1"].astype(np.float32)
        self.b1 = data["b1"].astype(np.float32)
        self.W2 = data["W2"].astype(np.float32)
        self.b2 = data["b2"].astype(np.float32)
        self.trained = True


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

class Button:
    def __init__(self, rect, label, sublabel=""):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.sublabel = sublabel
        self.hovered = False

    def handle_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)

    def clicked(self, pos):
        return self.rect.collidepoint(pos)

    def draw(self, surf, font, subfont, active=False, active_color=BTN_ACTIVE_HUMAN):
        if active:
            color = active_color
        elif self.hovered:
            color = BTN_HOVER
        else:
            color = BTN_IDLE
        pygame.draw.rect(surf, color, self.rect, border_radius=12)
        pygame.draw.rect(surf, BTN_BORDER, self.rect, width=2, border_radius=12)

        label_color = (18, 18, 24) if active else TEXT_MAIN
        label_surf = font.render(self.label, True, label_color)
        ly = self.rect.centery - (10 if self.sublabel else 0)
        surf.blit(label_surf, label_surf.get_rect(center=(self.rect.centerx, ly)))

        if self.sublabel:
            sub_color = (30, 30, 34) if active else TEXT_DIM
            sub_surf = subfont.render(self.sublabel, True, sub_color)
            surf.blit(sub_surf, sub_surf.get_rect(center=(self.rect.centerx, self.rect.centery + 13)))


def vertical_gradient(surf, top_color, bottom_color):
    h = surf.get_height()
    for y in range(h):
        t = y / max(h - 1, 1)
        color = tuple(int(top_color[i] + (bottom_color[i] - top_color[i]) * t) for i in range(3))
        pygame.draw.line(surf, color, (0, y), (surf.get_width(), y))


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_rounded_cell(surf, x, y, color, outline=None, radius=7, inset=1):
    x, y = int(x), int(y)  # guard against numpy narrow-int scalars
    rect = pygame.Rect(BOARD_X + x * CELL + inset, BOARD_Y + y * CELL + inset,
                        CELL - inset * 2, CELL - inset * 2)
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if outline:
        pygame.draw.rect(surf, outline, rect, width=1, border_radius=radius)


def show_alert(title: str, message: str):
    """Show a system alert window reporting a load failure."""
    if HAS_TK:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        messagebox.showerror(title, message)
        root.destroy()
    else:
        # Fallback if tkinter is unavailable
        print(f"[ALERT] {title}: {message}")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class SnakeApp:
    MODE_HUMAN = "human"
    MODE_AI = "ai"

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake -- Human vs. Neural Network")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags=pygame.RESIZABLE | pygame.SCALED)
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("arial", 30, bold=True)
        self.font_h2 = pygame.font.SysFont("arial", 18, bold=True)
        self.font_body = pygame.font.SysFont("arial", 16)
        self.font_small = pygame.font.SysFont("arial", 13)
        self.font_big_score = pygame.font.SysFont("arial", 42, bold=True)
        self.font_btn = pygame.font.SysFont("arial", 17, bold=True)
        self.font_btn_sub = pygame.font.SysFont("arial", 12)

        # pre-render the background gradient once
        self.bg = pygame.Surface((SCREEN_W, SCREEN_H))
        vertical_gradient(self.bg, BG_TOP, BG_BOTTOM)

        cfg = SnakeConfig(width=GRID_W, height=GRID_H)
        self.game = SnakeGame(cfg)

        # Match the architecture used by train_snake.py
        self.agent = NeuralNetworkAgent(
            input_size=22,
            hidden_size=64,
            output_size=3,
        )

        # Attempt to load the best trained weights
        weights_path = "best_weights.npz"
        try:
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"File not found: {weights_path}")
            self.agent.load_weights(weights_path)
            # Quick sanity-check on shapes
            if self.agent.W1.shape != (22, 64) or self.agent.W2.shape != (64, 3):
                raise ValueError(
                    f"Unexpected weight shapes: W1={self.agent.W1.shape}, W2={self.agent.W2.shape}"
                )
        except Exception as exc:
            self.agent.trained = False
            show_alert(
                "Model Load Failed",
                f"Could not load '{weights_path}'.\n\n"
                f"Reason: {exc}\n\n"
                "The AI will run with random untrained weights."
            )

        self.mode = self.MODE_HUMAN
        self.paused = False
        self.game_over = False
        self.high_score = 0

        self.move_accum = 0.0
        self.pending_dir = None  # queued absolute direction from human input
        self.time_alive = 0.0

        self._build_buttons()
        self.reset_game()

    # -- setup -----------------------------------------------------------

    def _build_buttons(self):
        panel_x = BOARD_W + MARGIN * 2
        w = SIDEBAR_W - MARGIN * 2
        bx = panel_x + MARGIN

        y = 190
        self.btn_human = Button((bx, y, w, 56), "Human Control", "Arrow keys / WASD")
        y += 66

        if self.agent.trained:
            ai_sub = "Trained model loaded"
        else:
            ai_sub = "Please train first"

        self.btn_ai = Button((bx, y, w, 56), "Neural Network", ai_sub)
        y += 78
        self.btn_restart = Button((bx, y, w, 44), "Restart")
        self.panel_x = panel_x

    def reset_game(self):
        self.game.reset()
        self.game_over = False
        self.move_accum = 0.0
        self.pending_dir = None
        self.time_alive = 0.0

    # -- input -------------------------------------------------------------

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                if event.key == pygame.K_SPACE:
                    if self.game_over:
                        self.reset_game()
                    else:
                        self.paused = not self.paused
                if event.key == pygame.K_r:
                    self.reset_game()
                if self.mode == self.MODE_HUMAN and event.key in ABS_KEYS:
                    self.pending_dir = ABS_KEYS[event.key]

            elif event.type == pygame.MOUSEMOTION:
                self.btn_human.handle_hover(event.pos)
                self.btn_ai.handle_hover(event.pos)
                self.btn_restart.handle_hover(event.pos)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_human.clicked(event.pos):
                    self.mode = self.MODE_HUMAN
                elif self.btn_ai.clicked(event.pos):
                    self.mode = self.MODE_AI
                elif self.btn_restart.clicked(event.pos):
                    self.reset_game()

    # -- stepping ------------------------------------------------------

    def choose_action(self):
        if self.mode == self.MODE_AI:
            state = self.game.get_state()
            return self.agent.act(state)

        # human mode: convert queued absolute direction into a relative
        # action; if nothing queued (or it's a reversal), go straight
        if self.pending_dir is not None:
            action = absolute_to_relative(int(self.game.direction), self.pending_dir)
            self.pending_dir = None
            if action is not None:
                return action
        return ACTION_STRAIGHT

    def update(self, dt: float):
        if self.game_over or self.paused:
            return

        self.time_alive += dt
        moves_per_sec = AI_MOVES_PER_SEC if self.mode == self.MODE_AI else HUMAN_MOVES_PER_SEC
        self.move_accum += dt * moves_per_sec

        while self.move_accum >= 1.0 and not self.game_over:
            self.move_accum -= 1.0
            action = self.choose_action()
            _, _, terminated, truncated, _ = self.game.step(action)
            if terminated or truncated:
                self.game_over = True
                self.high_score = max(self.high_score, self.game.score)

    # -- drawing ---------------------------------------------------------

    def draw_board(self):
        board_rect = pygame.Rect(BOARD_X - 4, BOARD_Y - 4, BOARD_W + 8, BOARD_H + 8)
        pygame.draw.rect(self.screen, PANEL, board_rect, border_radius=16)
        pygame.draw.rect(self.screen, PANEL_BORDER, board_rect, width=2, border_radius=16)

        # checkerboard-ish subtle floor
        for gy in range(GRID_H):
            for gx in range(GRID_W):
                color = BOARD_BG_A if (gx + gy) % 2 == 0 else BOARD_BG_B
                rect = pygame.Rect(BOARD_X + gx * CELL, BOARD_Y + gy * CELL, CELL, CELL)
                pygame.draw.rect(self.screen, color, rect)

        # faint grid lines
        for gx in range(GRID_W + 1):
            x = BOARD_X + gx * CELL
            pygame.draw.line(self.screen, GRID_LINE, (x, BOARD_Y), (x, BOARD_Y + BOARD_H))
        for gy in range(GRID_H + 1):
            y = BOARD_Y + gy * CELL
            pygame.draw.line(self.screen, GRID_LINE, (BOARD_X, y), (BOARD_X + BOARD_W, y))

        # food: pulsing glow
        if self.game.food is not None:
            fx, fy = self.game.food
            cx = BOARD_X + fx * CELL + CELL // 2
            cy = BOARD_Y + fy * CELL + CELL // 2
            pulse = (math.sin(pygame.time.get_ticks() / 220.0) + 1) / 2  # 0..1
            glow_r = int(CELL * 0.55 + pulse * 4)
            glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*FOOD_GLOW, 70), (glow_r, glow_r), glow_r)
            self.screen.blit(glow_surf, (cx - glow_r, cy - glow_r))
            pygame.draw.circle(self.screen, FOOD_CORE, (cx, cy), int(CELL * 0.32))
            pygame.draw.circle(self.screen, (255, 220, 220), (cx, cy), int(CELL * 0.32), width=1)

        # snake: gradient body, distinct head
        body_list = list(self.game.body)  # tail-first .. head-last
        n = len(body_list)
        for i, (x, y) in enumerate(body_list):
            is_head = (i == n - 1)
            x, y = int(x), int(y)  # guard against numpy narrow-int scalars
            if is_head:
                draw_rounded_cell(self.screen, x, y, SNAKE_HEAD, SNAKE_OUTLINE, radius=9)
                # simple eyes to sell "head"
                cx = BOARD_X + x * CELL + CELL // 2
                cy = BOARD_Y + y * CELL + CELL // 2
                d = int(self.game.direction)
                eye_off = {
                    UP: [(-5, -4), (5, -4)],
                    DOWN: [(-5, 4), (5, 4)],
                    LEFT: [(-4, -5), (-4, 5)],
                    RIGHT: [(4, -5), (4, 5)],
                }[d]
                for ox, oy in eye_off:
                    pygame.draw.circle(self.screen, (18, 30, 24), (cx + ox, cy + oy), 2)
            else:
                t = i / max(n - 2, 1)
                color = lerp_color(SNAKE_BODY_A, SNAKE_BODY_B, t)
                draw_rounded_cell(self.screen, x, y, color, radius=6, inset=2)

        if self.paused and not self.game_over:
            self._draw_overlay("PAUSED", "press space to resume")

        if self.game_over:
            self._draw_overlay("GAME OVER", f"score {self.game.score}  --  press space or R to restart")

    def _draw_overlay(self, title, subtitle):
        overlay = pygame.Surface((BOARD_W, BOARD_H), pygame.SRCALPHA)
        overlay.fill(GAMEOVER_OVERLAY)
        self.screen.blit(overlay, (BOARD_X, BOARD_Y))

        title_surf = self.font_title.render(title, True, TEXT_MAIN)
        sub_surf = self.font_body.render(subtitle, True, TEXT_DIM)
        cx = BOARD_X + BOARD_W // 2
        cy = BOARD_Y + BOARD_H // 2
        self.screen.blit(title_surf, title_surf.get_rect(center=(cx, cy - 14)))
        self.screen.blit(sub_surf, sub_surf.get_rect(center=(cx, cy + 20)))

    def draw_sidebar(self):
        px = self.panel_x
        panel_rect = pygame.Rect(px, MARGIN, SIDEBAR_W - MARGIN, SCREEN_H - MARGIN * 2)
        pygame.draw.rect(self.screen, PANEL, panel_rect, border_radius=16)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel_rect, width=2, border_radius=16)

        pad = MARGIN
        y = MARGIN + 20

        title = self.font_title.render("SNAKE", True, TEXT_MAIN)
        self.screen.blit(title, (px + pad, y))
        y += 40
        subtitle = self.font_small.render("powered by snake_engine.py", True, TEXT_DIM)
        self.screen.blit(subtitle, (px + pad, y))
        y += 30

        score_label = self.font_h2.render("SCORE", True, TEXT_DIM)
        self.screen.blit(score_label, (px + pad, y))
        score_val = self.font_big_score.render(str(self.game.score), True, ACCENT)
        self.screen.blit(score_val, (px + pad, y + 20))

        best_label = self.font_small.render(f"BEST  {self.high_score}", True, TEXT_DIM)
        self.screen.blit(best_label, (px + SIDEBAR_W - MARGIN - pad - best_label.get_width(), y + 30))

        # control mode buttons
        self.btn_human.draw(self.screen, self.font_btn, self.font_btn_sub,
                             active=(self.mode == self.MODE_HUMAN), active_color=BTN_ACTIVE_HUMAN)
        self.btn_ai.draw(self.screen, self.font_btn, self.font_btn_sub,
                          active=(self.mode == self.MODE_AI), active_color=BTN_ACTIVE_AI)
        self.btn_restart.draw(self.screen, self.font_btn, self.font_btn_sub)

        y2 = self.btn_restart.rect.bottom + 26
        if self.mode == self.MODE_AI:
            status = "UNTRAINED" if not self.agent.trained else "TRAINED"
            status_color = ACCENT_2 if not self.agent.trained else (110, 245, 170)
            self.screen.blit(self.font_h2.render("AI STATUS", True, TEXT_DIM), (px + pad, y2))
            self.screen.blit(self.font_h2.render(status, True, status_color), (px + pad, y2 + 22))
            note = [
                "Random-weight MLP placeholder --",
                "moves aren't learned yet. Train an",
                "agent against SnakeGame/VecSnakeGame",
                "and load its weights to replace this.",
            ] if not self.agent.trained else [
                "Loaded from best_weights.npz",
                "22 -> 64 -> 3  (ReLU)",
                "Action space: straight / right / left",
            ]
            ny = y2 + 50
            for line in note:
                self.screen.blit(self.font_small.render(line, True, TEXT_DIM), (px + pad, ny))
                ny += 17
        else:
            self.screen.blit(self.font_h2.render("CONTROLS", True, TEXT_DIM), (px + pad, y2))
            note = [
                "Arrow keys or WASD to steer",
                "Space to pause / restart",
                "R to restart anytime",
            ]
            ny = y2 + 26
            for line in note:
                self.screen.blit(self.font_small.render(line, True, TEXT_DIM), (px + pad, ny))
                ny += 18

        # footer
        footer = self.font_small.render(f"grid {GRID_W}x{GRID_H}  --  esc to quit", True, TEXT_DIM)
        self.screen.blit(footer, (px + pad, SCREEN_H - MARGIN - 24))

    def draw(self):
        self.screen.blit(self.bg, (0, 0))
        self.draw_board()
        self.draw_sidebar()
        pygame.display.flip()

    # -- main loop ---------------------------------------------------------

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()


if __name__ == "__main__":
    SnakeApp().run()
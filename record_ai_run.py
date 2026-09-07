"""
record_ai_run.py
================
Launches the Snake game in AI mode, records every frame until the snake dies,
then saves the recording as an animated GIF.

Requirements:
    pip install pygame-ce numpy imageio

Usage:
    python record_ai_run.py

Output:
    snake_ai_run.gif
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pygame
import imageio.v2 as imageio

# Re-use everything from main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import SnakeApp


def capture_frame(screen: pygame.Surface) -> np.ndarray:
    """Convert the current pygame surface into an RGB numpy array (H, W, 3)."""
    frame = pygame.surfarray.array3d(screen)          # (W, H, 3)
    frame = np.transpose(frame, (1, 0, 2))             # -> (H, W, 3)
    return frame.copy()


def main():
    # Create the full application (loads best_weights.npz automatically)
    app = SnakeApp()

    # Force AI mode from the very first frame
    app.mode = app.MODE_AI
    app.paused = False
    app.game_over = False

    frames: list[np.ndarray] = []
    max_frames = 4000          # safety limit (~ few minutes at 15 fps)
    record_fps = 15            # GIF playback speed
    capture_every_n = 2        # capture every 2nd frame to keep GIF size reasonable

    print("Recording AI run... (window will close automatically when the snake dies)")

    frame_counter = 0
    while not app.game_over and len(frames) < max_frames:
        dt = app.clock.tick(30) / 1000.0          # run a bit slower for cleaner recording

        # Minimal event handling – allow clean exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit(0)

        app.update(dt)
        app.draw()

        # Capture
        if frame_counter % capture_every_n == 0:
            frames.append(capture_frame(app.screen))
        frame_counter += 1

    # Keep the GAME OVER screen for a short moment
    for _ in range(int(record_fps * 1.5)):
        app.draw()
        frames.append(capture_frame(app.screen))
        app.clock.tick(record_fps)

    pygame.quit()

    if not frames:
        print("No frames captured – nothing to save.")
        return

    out_path = "snake_ai_run.gif"
    print(f"Saving {len(frames)} frames to {out_path} ...")
    imageio.mimsave(
        out_path,
        frames,
        fps=record_fps,
        loop=0,                 # infinite loop
    )
    print(f"Done → {out_path}")
    print(f"Final score: {app.game.score}")


if __name__ == "__main__":
    main()
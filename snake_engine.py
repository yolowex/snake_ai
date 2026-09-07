"""
snake_engine.py
================
A fast, dependency-light Snake engine built for training RL agents.

Two engines are provided:

1. SnakeGame
   A single-instance engine. Uses a numpy grid for O(1) collision checks
   and a compact 11-value boolean feature vector as the default
   observation (the standard lightweight input used in most snake-RL
   work, ideal for a small MLP / DQN). Full grid observation is also
   available for CNN-style agents.

2. VecSnakeGame
   A fully vectorized engine that steps B independent games at once
   using pure numpy array ops -- there is no Python for-loop over
   environments in the hot path. This is what you want for actually
   training fast (thousands of steps/sec instead of tens of thousands
   of Python function calls/sec).

Both expose a gymnasium-style step() -> (obs, reward, terminated, truncated, info).

Dependencies: numpy only.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

EMPTY, BODY, HEAD, FOOD = 0, 1, 2, 3

# direction indices: 0=up, 1=right, 2=down, 3=left (clockwise)
DIRS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.int8)

# actions are RELATIVE to current heading: 0=straight, 1=turn right, 2=turn left
# (this action space can never cause an instant 180-degree self-collision,
#  which is standard practice for snake-RL and shrinks the action space to 3)
ACTION_STRAIGHT, ACTION_RIGHT, ACTION_LEFT = 0, 1, 2


def _turn(direction: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Vectorized: apply a relative turn to a direction array."""
    # right turn: +1, left turn: -1, straight: +0
    delta = np.where(action == ACTION_RIGHT, 1, np.where(action == ACTION_LEFT, -1, 0))
    return (direction + delta) % 4


# ---------------------------------------------------------------------------
# 1. Single-instance engine
# ---------------------------------------------------------------------------

@dataclass
class SnakeConfig:
    width: int = 12
    height: int = 12
    max_steps_without_food: Optional[int] = None  # default: width*height*4
    food_reward: float = 10.0
    death_reward: float = -10.0
    step_reward: float = -0.01          # small time penalty -> encourages efficiency
    move_toward_food_reward: float = 0.0  # optional shaping, 0 = off


class SnakeGame:
    """
    Single Snake game instance, optimized for tight RL step loops.

    - Grid stored as a uint8 numpy array -> O(1) collision lookups.
    - Snake body stored in a deque -> O(1) append/pop at both ends.
    - No object allocation in the step() hot path.
    """

    def __init__(self, config: SnakeConfig = SnakeConfig()):
        self.cfg = config
        self.w, self.h = config.width, config.height
        self.max_steps_without_food = (
            config.max_steps_without_food or self.w * self.h * 4
        )
        self.grid = np.zeros((self.h, self.w), dtype=np.uint8)
        self.reset()

    # -- lifecycle -----------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)

        self.grid.fill(EMPTY)
        cx, cy = self.w // 2, self.h // 2

        # body stored tail-first -> head-last, so body[-1] is always the head
        self.body = deque([(cx - 1, cy), (cx, cy)])
        self.direction = np.int8(1)  # facing right
        for (x, y) in list(self.body)[:-1]:
            self.grid[y, x] = BODY
        self.grid[cy, cx] = HEAD

        self.steps_since_food = 0
        self.score = 0
        self._place_food()
        return self.get_state()

    def _place_food(self) -> None:
        empties = np.argwhere(self.grid == EMPTY)
        if len(empties) == 0:
            # board is full -> the snake has won, no food to place
            self.food = None
            return
        y, x = empties[np.random.randint(len(empties))]
        self.food = (int(x), int(y))
        self.grid[y, x] = FOOD

    # -- stepping --------------------------------------------------------

    def step(self, action: int):
        """
        action: 0 = straight, 1 = turn right, 2 = turn left

        returns: (observation, reward, terminated, truncated, info)
        """
        if action == ACTION_RIGHT:
            self.direction = np.int8((self.direction + 1) % 4)
        elif action == ACTION_LEFT:
            self.direction = np.int8((self.direction - 1) % 4)

        hx, hy = self.body[-1]
        dx, dy = DIRS[self.direction]
        # DIRS is an int8 array -- cast to plain Python ints so body
        # coordinates never silently narrow to int8 (which overflows
        # once a renderer multiplies them by a pixel-size constant).
        nx, ny = int(hx) + int(dx), int(hy) + int(dy)

        terminated = False
        truncated = False
        reward = self.cfg.step_reward
        ate = False

        # wall collision
        if not (0 <= nx < self.w and 0 <= ny < self.h):
            terminated = True
            reward = self.cfg.death_reward
        else:
            cell = self.grid[ny, nx]
            # moving into the current tail cell is legal iff the tail will
            # move away this turn, i.e. we are NOT eating food this step
            will_eat = self.food is not None and (nx, ny) == self.food
            tail = self.body[0]
            is_tail_cell = (nx, ny) == tail and not will_eat

            if cell == BODY and not is_tail_cell:
                terminated = True
                reward = self.cfg.death_reward
            elif cell == HEAD:
                terminated = True
                reward = self.cfg.death_reward
            else:
                ate = will_eat

        if not terminated:
            # optional distance-based shaping, computed before moving
            if self.cfg.move_toward_food_reward and self.food is not None:
                old_dist = abs(hx - self.food[0]) + abs(hy - self.food[1])
                new_dist = abs(nx - self.food[0]) + abs(ny - self.food[1])
                reward += self.cfg.move_toward_food_reward * np.sign(old_dist - new_dist)

            # move head
            self.grid[hy, hx] = BODY
            self.body.append((nx, ny))
            self.grid[ny, nx] = HEAD

            if ate:
                self.score += 1
                self.steps_since_food = 0
                reward = self.cfg.food_reward
                self._place_food()
                if self.food is None:
                    # snake fills the whole board -> win, end episode
                    terminated = True
            else:
                tx, ty = self.body.popleft()
                self.grid[ty, tx] = EMPTY
                self.steps_since_food += 1
                if self.steps_since_food >= self.max_steps_without_food:
                    truncated = True

        info = {"score": self.score}
        return self.get_state(), float(reward), terminated, truncated, info

    # -- observations ----------------------------------------------------

    def get_state(self) -> np.ndarray:
        """
        Compact 11-value boolean feature vector (fast, MLP-friendly):
        [danger_straight, danger_right, danger_left,
         dir_up, dir_right, dir_down, dir_left,
         food_up, food_down, food_left, food_right]
        """
        hx, hy = self.body[-1]
        d = int(self.direction)

        def blocked(dir_idx):
            dx, dy = DIRS[dir_idx]
            x, y = hx + dx, hy + dy
            if not (0 <= x < self.w and 0 <= y < self.h):
                return True
            cell = self.grid[y, x]
            tail = self.body[0]
            if (x, y) == tail:
                return False
            return cell == BODY or cell == HEAD

        straight_dir = d
        right_dir = (d + 1) % 4
        left_dir = (d - 1) % 4

        state = np.zeros(11, dtype=np.float32)
        state[0] = blocked(straight_dir)
        state[1] = blocked(right_dir)
        state[2] = blocked(left_dir)
        state[3 + d] = 1.0  # one-hot current direction (indices 3..6)

        if self.food is not None:
            fx, fy = self.food
            state[7] = fy < hy   # food up
            state[8] = fy > hy   # food down
            state[9] = fx < hx   # food left
            state[10] = fx > hx  # food right

        return state

    def get_grid(self) -> np.ndarray:
        """Full grid observation (H, W), for CNN-based agents."""
        return self.grid.copy()

    def render(self) -> str:
        chars = {EMPTY: ".", BODY: "o", HEAD: "@", FOOD: "*"}
        return "\n".join(
            "".join(chars[v] for v in row) for row in self.grid
        )


# ---------------------------------------------------------------------------
# 2. Vectorized batch engine -- the fast one, for actual training
# ---------------------------------------------------------------------------

class VecSnakeGame:
    """
    Runs `batch_size` independent Snake games simultaneously using pure
    numpy vector ops. There is no per-environment Python loop in step():
    every env is advanced by the same batch of array operations at once.

    Snake bodies are stored as ring buffers (fixed-size numpy arrays per
    env with head/tail pointers), giving O(1) push/pop per env, done for
    the whole batch in a single vectorized assignment.

    Observation: full grids, shape (B, H, W), uint8.
    Actions: (B,) int array, values in {0=straight, 1=right, 2=left}.

    Finished envs (terminated or truncated) are auto-reset at the end of
    step(), matching the behavior of gymnasium's vector envs. The
    observation right before reset is stored in info["final_observation"].
    """

    def __init__(
        self,
        batch_size: int,
        width: int = 12,
        height: int = 12,
        max_steps_without_food: Optional[int] = None,
        food_reward: float = 10.0,
        death_reward: float = -10.0,
        step_reward: float = -0.01,
        seed: Optional[int] = None,
    ):
        self.B = batch_size
        self.w, self.h = width, height
        self.capacity = width * height  # max possible snake length
        self.max_steps_without_food = max_steps_without_food or width * height * 4
        self.food_reward = food_reward
        self.death_reward = death_reward
        self.step_reward = step_reward
        self.rng = np.random.default_rng(seed)

        # grid[b, y, x]
        self.grid = np.zeros((self.B, self.h, self.w), dtype=np.uint8)

        # ring buffers for snake bodies
        self.body_x = np.zeros((self.B, self.capacity), dtype=np.int16)
        self.body_y = np.zeros((self.B, self.capacity), dtype=np.int16)
        self.head_ptr = np.zeros(self.B, dtype=np.int32)
        self.tail_ptr = np.zeros(self.B, dtype=np.int32)
        self.length = np.zeros(self.B, dtype=np.int32)

        self.direction = np.ones(self.B, dtype=np.int8)  # facing right
        self.food_x = np.zeros(self.B, dtype=np.int16)
        self.food_y = np.zeros(self.B, dtype=np.int16)
        self.has_food = np.zeros(self.B, dtype=bool)
        self.steps_since_food = np.zeros(self.B, dtype=np.int32)
        self.score = np.zeros(self.B, dtype=np.int32)

        self.reset_all()

    # -- helpers -----------------------------------------------------

    def _head_xy(self):
        idx = np.arange(self.B)
        return self.body_x[idx, self.head_ptr], self.body_y[idx, self.head_ptr]

    def _tail_xy(self):
        idx = np.arange(self.B)
        return self.body_x[idx, self.tail_ptr], self.body_y[idx, self.tail_ptr]

    def _push_head(self, mask: np.ndarray, x: np.ndarray, y: np.ndarray):
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return
        self.head_ptr[idx] = (self.head_ptr[idx] + 1) % self.capacity
        self.body_x[idx, self.head_ptr[idx]] = x[idx]
        self.body_y[idx, self.head_ptr[idx]] = y[idx]
        self.length[idx] += 1

    def _pop_tail(self, mask: np.ndarray):
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return
        tx, ty = self.body_x[idx, self.tail_ptr[idx]], self.body_y[idx, self.tail_ptr[idx]]
        self.grid[idx, ty, tx] = EMPTY
        self.tail_ptr[idx] = (self.tail_ptr[idx] + 1) % self.capacity
        self.length[idx] -= 1

    def _place_food(self, mask: np.ndarray):
        """Vectorized rejection-sampling food placement for envs in `mask`."""
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            return
        remaining = idx.copy()
        for _ in range(64):  # bounded rejection-sampling passes
            if len(remaining) == 0:
                break
            xs = self.rng.integers(0, self.w, size=len(remaining))
            ys = self.rng.integers(0, self.h, size=len(remaining))
            free = self.grid[remaining, ys, xs] == EMPTY
            good = remaining[free]
            self.food_x[good] = xs[free]
            self.food_y[good] = ys[free]
            self.grid[good, ys[free], xs[free]] = FOOD
            self.has_food[good] = True
            remaining = remaining[~free]

        # fallback for any stragglers (near-full boards): explicit search
        for b in remaining:
            empties = np.argwhere(self.grid[b] == EMPTY)
            if len(empties) == 0:
                self.has_food[b] = False  # board full -> win
                continue
            y, x = empties[self.rng.integers(len(empties))]
            self.food_x[b], self.food_y[b] = x, y
            self.grid[b, y, x] = FOOD
            self.has_food[b] = True

    # -- lifecycle -----------------------------------------------------

    def reset_all(self) -> np.ndarray:
        idx = np.arange(self.B)
        self._reset_envs(idx)
        return self.get_obs()

    def _reset_envs(self, idx: np.ndarray):
        if len(idx) == 0:
            return
        self.grid[idx] = EMPTY
        cx, cy = self.w // 2, self.h // 2

        self.head_ptr[idx] = 1
        self.tail_ptr[idx] = 0
        self.length[idx] = 2
        self.body_x[idx, 0], self.body_y[idx, 0] = cx - 1, cy
        self.body_x[idx, 1], self.body_y[idx, 1] = cx, cy
        self.grid[idx, cy, cx - 1] = BODY
        self.grid[idx, cy, cx] = HEAD

        self.direction[idx] = 1  # right
        self.steps_since_food[idx] = 0
        self.score[idx] = 0
        self.has_food[idx] = False
        self._place_food(np.isin(np.arange(self.B), idx))

    # -- stepping --------------------------------------------------------

    def step(self, actions: np.ndarray):
        """
        actions: (B,) int array, 0=straight 1=right 2=left

        returns: obs (B,H,W) uint8, reward (B,) float32,
                 terminated (B,) bool, truncated (B,) bool, info dict
        """
        actions = np.asarray(actions)
        self.direction = _turn(self.direction, actions).astype(np.int8)

        hx, hy = self._head_xy()
        d = DIRS[self.direction]
        nx = (hx + d[:, 0]).astype(np.int16)
        ny = (hy + d[:, 1]).astype(np.int16)

        in_bounds = (nx >= 0) & (nx < self.w) & (ny >= 0) & (ny < self.h)
        safe_nx = np.clip(nx, 0, self.w - 1)
        safe_ny = np.clip(ny, 0, self.h - 1)

        idx = np.arange(self.B)
        cell_val = self.grid[idx, safe_ny, safe_nx]

        will_eat = in_bounds & self.has_food & (nx == self.food_x) & (ny == self.food_y)
        tx, ty = self._tail_xy()
        is_tail_cell = in_bounds & (nx == tx) & (ny == ty) & (~will_eat)

        hit_body = in_bounds & (cell_val == BODY) & (~is_tail_cell)
        hit_head = in_bounds & (cell_val == HEAD)  # only relevant if length==1, kept for safety
        terminated = (~in_bounds) | hit_body | hit_head

        reward = np.full(self.B, self.step_reward, dtype=np.float32)
        reward[terminated] = self.death_reward

        alive = ~terminated
        move_mask = alive
        eat_mask = alive & will_eat
        move_no_eat_mask = alive & (~will_eat)

        # Order matters here: when the tail vacates the exact cell the head
        # is moving into (a common, legal move), the tail-clear must happen
        # BEFORE the new head is written -- otherwise it wipes the head
        # marker back to EMPTY. So: clear old head -> pop tail -> push head
        # -> write new head, strictly in that order.
        self.grid[idx[move_mask], hy[move_mask], hx[move_mask]] = BODY
        self._pop_tail(move_no_eat_mask)
        self._push_head(move_mask, nx.astype(np.int64), ny.astype(np.int64))
        self.grid[idx[move_mask], ny[move_mask], nx[move_mask]] = HEAD

        # eating: score up, reset hunger clock, give reward, place new food
        self.score[eat_mask] += 1
        self.steps_since_food[eat_mask] = 0
        reward[eat_mask] = self.food_reward
        self._place_food(eat_mask)
        # board-full win condition after eating
        won = eat_mask & (~self.has_food)
        terminated = terminated | won

        self.steps_since_food[move_no_eat_mask] += 1

        truncated = alive & (self.steps_since_food >= self.max_steps_without_food)

        obs = self.get_obs()
        info = {"score": self.score.copy()}

        done_mask = terminated | truncated
        if done_mask.any():
            info["final_observation"] = obs[done_mask].copy()
            info["final_info"] = {"score": self.score[done_mask].copy()}
            self._reset_envs(np.nonzero(done_mask)[0])
            obs = self.get_obs()  # refresh post-reset for those envs

        return obs, reward, terminated, truncated, info

    # -- observations ----------------------------------------------------

    def get_obs(self) -> np.ndarray:
        """Full grid observation for every env: shape (B, H, W), uint8."""
        return self.grid.copy()

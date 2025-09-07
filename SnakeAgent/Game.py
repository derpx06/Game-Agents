import pygame
import sys
import random
import numpy as np


class SnakeGame:
    def __init__(self, w=800, h=600, block=20):
        self.w = w
        self.h = h
        self.BLOCK = block

        pygame.init()
        pygame.display.set_caption("Snake Eater AI")
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # speed baseline
        self.difficulty = 10  # lowered baseline for smoother start

        # palette
        self.color_bg_top = pygame.Color(16, 20, 36)
        self.color_bg_bottom = pygame.Color(28, 34, 56)
        self.color_grid = (40, 46, 68)
        self.color_snake_body = pygame.Color(80, 235, 120)
        self.color_snake_head = pygame.Color(150, 255, 180)
        self.color_food = pygame.Color(255, 120, 120)
        self.color_obstacle = pygame.Color(128, 128, 128)
        self.color_text = pygame.Color(220, 220, 220)
        self.color_hud = (0, 0, 0, 80)

        # snake init
        x0, y0 = self.BLOCK * 5, self.BLOCK * 2
        self.snake_pos = [x0, y0]
        self.snake_body = [
            [x0, y0],
            [x0 - self.BLOCK, y0],
            [x0 - 2 * self.BLOCK, y0],
        ]
        self.direction = "RIGHT"
        self.score = 0
        self.frame_iteration = 0
        self.obstacles = []

        self._place_food()

    # -------------------- DRAWING --------------------
    def _draw_gradient_bg(self):
        top = np.array(self.color_bg_top)
        bottom = np.array(self.color_bg_bottom)
        for y in range(self.h):
            t = y / max(1, self.h - 1)
            col = (top * (1 - t) + bottom * t).astype(np.uint8)
            pygame.draw.line(self.display, col, (0, y), (self.w, y))

    def _draw_grid(self):
        for x in range(0, self.w, self.BLOCK):
            pygame.draw.line(self.display, self.color_grid, (x, 0), (x, self.h), 1)
        for y in range(0, self.h, self.BLOCK):
            pygame.draw.line(self.display, self.color_grid, (0, y), (self.w, y), 1)

    def _hud(self):
        # translucent bar
        hud_h = 32
        surf = pygame.Surface((self.w, hud_h), pygame.SRCALPHA)
        surf.fill(self.color_hud)
        self.display.blit(surf, (0, 0))

        font = pygame.font.SysFont("consolas", 18, bold=True)
        txt = f"Score: {self.score}    Obstacles: {len(self.obstacles)}    Speed: {self.difficulty + self.score//2}    Frames: {self.frame_iteration}"
        ts = font.render(txt, True, self.color_text)
        self.display.blit(ts, (10, 6))

    def _place_food(self):
        while True:
            fx = random.randrange(1, (self.w // self.BLOCK)) * self.BLOCK
            fy = random.randrange(1, (self.h // self.BLOCK)) * self.BLOCK
            self.food_pos = [fx, fy]
            if self.food_pos not in self.snake_body and self.food_pos not in self.obstacles:
                break

    def _update_obstacles(self):
        # Add one obstacle every 4 points
        want = self.score // 4
        while len(self.obstacles) < want:
            ox = random.randrange(1, (self.w // self.BLOCK)) * self.BLOCK
            oy = random.randrange(1, (self.h // self.BLOCK)) * self.BLOCK
            pos = [ox, oy]
            if pos not in self.snake_body and pos != self.food_pos and pos not in self.obstacles:
                self.obstacles.append(pos)

    def _update_ui(self):
        self._draw_gradient_bg()
        self._draw_grid()

        # Food
        pygame.draw.rect(
            self.display,
            self.color_food,
            pygame.Rect(self.food_pos[0], self.food_pos[1], self.BLOCK, self.BLOCK),
            border_radius=4,
        )

        # Obstacles
        for obs in self.obstacles:
            pygame.draw.rect(
                self.display,
                self.color_obstacle,
                pygame.Rect(obs[0], obs[1], self.BLOCK, self.BLOCK),
                border_radius=4,
            )

        # Snake
        for i, pos in enumerate(self.snake_body):
            color = self.color_snake_head if i == 0 else self.color_snake_body
            pygame.draw.rect(
                self.display,
                color,
                pygame.Rect(pos[0], pos[1], self.BLOCK, self.BLOCK),
                border_radius=6,
            )

        self._hud()
        pygame.display.flip()

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.snake_pos
        if pt[0] < 0 or pt[0] > self.w - self.BLOCK:
            return True
        if pt[1] < 0 or pt[1] > self.h - self.BLOCK:
            return True
        if pt in self.snake_body[1:]:
            return True
        if pt in self.obstacles:
            return True
        return False

    # -------------------- RL interface --------------------
    def get_state(self):
        head = self.snake_pos
        dir_map = {"RIGHT": 0, "LEFT": 1, "UP": 2, "DOWN": 3}
        current_direction_int = dir_map[self.direction]

        step = self.BLOCK
        point_l = [head[0] - step, head[1]]
        point_r = [head[0] + step, head[1]]
        point_u = [head[0], head[1] - step]
        point_d = [head[0], head[1] + step]

        dir_l = current_direction_int == 1
        dir_r = current_direction_int == 0
        dir_u = current_direction_int == 2
        dir_d = current_direction_int == 3

        state = [
            # danger straight
            (dir_r and self._is_collision(point_r))
            or (dir_l and self._is_collision(point_l))
            or (dir_u and self._is_collision(point_u))
            or (dir_d and self._is_collision(point_d)),
            # danger right
            (dir_u and self._is_collision(point_r))
            or (dir_d and self._is_collision(point_l))
            or (dir_l and self._is_collision(point_u))
            or (dir_r and self._is_collision(point_d)),
            # danger left
            (dir_d and self._is_collision(point_r))
            or (dir_u and self._is_collision(point_l))
            or (dir_r and self._is_collision(point_u))
            or (dir_l and self._is_collision(point_d)),
            # direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # food location
            self.food_pos[0] < head[0],
            self.food_pos[0] > head[0],
            self.food_pos[1] < head[1],
            self.food_pos[1] > head[1],
        ]
        return np.array(state, dtype=int)

    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # turn logic (clockwise list)
        clockwise = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = clockwise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):  # keep
            change_to = clockwise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # right turn
            change_to = clockwise[(idx + 1) % 4]
        else:  # left turn
            change_to = clockwise[(idx - 1) % 4]
        self.direction = change_to

        # move
        if self.direction == "UP":
            self.snake_pos[1] -= self.BLOCK
        if self.direction == "DOWN":
            self.snake_pos[1] += self.BLOCK
        if self.direction == "LEFT":
            self.snake_pos[0] -= self.BLOCK
        if self.direction == "RIGHT":
            self.snake_pos[0] += self.BLOCK

        # update body
        self.snake_body.insert(0, list(self.snake_pos))

        reward = 0
        if self.snake_pos == self.food_pos:
            self.score += 1
            reward = 17
            self._place_food()
            self._update_obstacles()
        else:
            self.snake_body.pop()

        done = False
        # collision or timeout
        if self._is_collision() or self.frame_iteration > 150 * (self.score + 1):
            done = True
            reward = -20
            return reward, done, self.score

        if self.frame_iteration > 50 * len(self.snake_body):
            reward -= 0.3
        else:
            reward += 0

        self._update_ui()
        # speed increases gently with score
        self.clock.tick(self.difficulty + self.score*0.8)
        return reward, done, self.score

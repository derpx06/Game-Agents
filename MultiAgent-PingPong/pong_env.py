import pygame
import random
import numpy as np

pygame.init()


class PongEnv:
    def __init__(self, screen_width=1200, screen_height=800,
                 paddle_height=141, base_ball_speed=12, paddle_speed=45,
                 win_score=11):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.paddle_height = paddle_height
        self.paddle_width = 25
        self.win_score = win_score

        self.base_ball_speed = base_ball_speed
        self.ball_speed = float(base_ball_speed)
        self.base_paddle_speed = paddle_speed
        self.paddle_speed = paddle_speed

        self.screen = None
        pygame.display.set_caption("SOTA Pong AI")

        # Game counters
        self.games_played = 0

        self.action_space = [0, 1, 2]
        # STATE_SIZE is now 8 to include opponent's paddle Y position
        self.state_size = 8

        self.max_paddle_y = self.screen_height - self.paddle_height
        self.ball_y_max = self.screen_height - 16
        self.paddle1_x = 15
        self.paddle2_x = self.screen_width - 15 - self.paddle_width

        self.reset()

    def reset(self):
        self.ball_x = float(self.screen_width) / 2.0
        self.ball_y = float(self.screen_height) / 2.0
        self.ball_speed = float(self.base_ball_speed)

        angle = random.uniform(-0.4, 0.4)
        direction = random.choice([-1, 1])
        vx, vy = direction * np.cos(angle), np.sin(angle)
        norm = np.hypot(vx, vy)
        self.ball_vx = (vx / norm) * self.ball_speed
        self.ball_vy = (vy / norm) * self.ball_speed

        self.paddle1_y = (self.screen_height - self.paddle_height) / 2.0
        self.paddle2_y = (self.screen_height - self.paddle_height) / 2.0

        self.score1 = 0
        self.score2 = 0
        self.done = False
        self.rally_count = 0

        return self.get_state(is_left=True), self.get_state(is_left=False)

    def get_state(self, is_left=True):
        # UPGRADE: State now includes opponent's paddle position for strategic play.
        if is_left:
            my_paddle_y = self.paddle1_y
            opponent_paddle_y = self.paddle2_y
        else:
            my_paddle_y = self.paddle2_y
            opponent_paddle_y = self.paddle1_y

        state = np.array([
            self.ball_x / self.screen_width,
            self.ball_y / self.screen_height,
            self.ball_vx / self.ball_speed,
            self.ball_vy / self.ball_speed,
            my_paddle_y / self.screen_height,
            opponent_paddle_y / self.screen_height,  # New state component
            (self.paddle1_y + self.paddle_height / 2) / self.screen_height,  # My paddle center
            (self.ball_y - (my_paddle_y + self.paddle_height / 2)) / self.screen_height
            # Ball relative to my paddle center
        ], dtype=np.float32)

        if not is_left:
            # Invert perspective for the right-side player
            state[0] = 1.0 - state[0]
            state[2] = -state[2]
        return state

    def step(self, action1, action2):
        self.paddle1_y = max(0.0, min(self.max_paddle_y,
                                      self.paddle1_y + [-self.paddle_speed, self.paddle_speed, 0][action1]))
        self.paddle2_y = max(0.0, min(self.max_paddle_y,
                                      self.paddle2_y + [-self.paddle_speed, self.paddle_speed, 0][action2]))

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y >= self.ball_y_max or self.ball_y <= 0:
            self.ball_vy *= -1

        # --- WORLD-CLASS REWARD SYSTEM ---
        reward1 = -0.01  # Efficiency penalty to discourage passivity
        reward2 = -0.01

        ball_half_height = 8
        paddle_half_height = self.paddle_height / 2

        # Paddle 1 (Left) collision
        if self.ball_vx < 0 and self.paddle1_x <= self.ball_x <= self.paddle1_x + self.paddle_width and self.paddle1_y <= self.ball_y <= self.paddle1_y + self.paddle_height:
            hit_offset = (self.ball_y - (self.paddle1_y + paddle_half_height)) / paddle_half_height
            self._bounce_ball(offset=hit_offset, left=True)
            self.rally_count += 1
            reward1 += 2.0  # Base reward for a successful return
            # Precision Bonus: Reward for hitting with the edge of the paddle for sharper angles
            reward1 += abs(hit_offset) * 2.0
            # Offensive Bonus: Reward for creating high vertical ball speed
            reward1 += abs(self.ball_vy / self.ball_speed) * 1.5

        # Paddle 2 (Right) collision
        elif self.ball_vx > 0 and self.paddle2_x <= self.ball_x <= self.paddle2_x + self.paddle_width and self.paddle2_y <= self.ball_y <= self.paddle2_y + self.paddle_height:
            hit_offset = (self.ball_y - (self.paddle2_y + paddle_half_height)) / paddle_half_height
            self._bounce_ball(offset=hit_offset, left=False)
            self.rally_count += 1
            reward2 += 2.0
            reward2 += abs(hit_offset) * 2.0
            reward2 += abs(self.ball_vy / self.ball_speed) * 1.5

        # Point scoring
        point_scored = False
        if self.ball_x < 0:
            self.score2 += 1
            reward2 += 15.0  # Big reward for winning a point
            reward1 -= 15.0  # Big penalty for losing a point
            point_scored = True
        elif self.ball_x > self.screen_width:
            self.score1 += 1
            reward1 += 15.0
            reward2 -= 15.0
            point_scored = True

        if point_scored:
            if self.score1 >= self.win_score or self.score2 >= self.win_score:
                self.done = True
                self.games_played += 1
            self.reset_ball()

        state1 = self.get_state(is_left=True)
        state2 = self.get_state(is_left=False)
        info = {"score1": self.score1, "score2": self.score2, "rally_count": self.rally_count,
                "games_played": self.games_played}
        return state1, state2, reward1, reward2, self.done, info

    def _bounce_ball(self, offset, left=True):
        angle = offset * (np.pi / 4)  # Max angle of 45 degrees
        speed_factor = 1.0 + abs(offset) * 0.1  # Edge hits are slightly faster

        direction = 1 if left else -1
        self.ball_vx = direction * self.ball_speed * np.cos(angle)
        self.ball_vy = self.ball_speed * np.sin(angle)

        # Increase ball speed slightly after each hit
        self.ball_speed = min(self.base_ball_speed * 1.8, self.ball_speed * 1.02)

    def reset_ball(self):
        self.ball_x = self.screen_width / 2
        self.ball_y = self.screen_height / 2
        self.rally_count = 0
        self.ball_speed = self.base_ball_speed  # Reset ball speed after a point
        angle = random.uniform(-0.2, 0.2)
        direction = 1 if self.score1 > self.score2 else -1
        self.ball_vx = direction * self.ball_speed * np.cos(angle)
        self.ball_vy = self.ball_speed * np.sin(angle)

    def render(self):
        # Standard rendering code, no changes needed here.
        if self.screen is None: self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill((15, 15, 15))  # Dark gray background
        pygame.draw.line(self.screen, (100, 100, 100), (self.screen_width / 2, 0),
                         (self.screen_width / 2, self.screen_height), 4)
        pygame.draw.rect(self.screen, (200, 200, 200),
                         (self.paddle1_x, int(self.paddle1_y), self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, (200, 200, 200),
                         (self.paddle2_x, int(self.paddle2_y), self.paddle_width, self.paddle_height))
        pygame.draw.ellipse(self.screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y), 16, 16))
        font = pygame.font.Font(None, 74)
        text1 = font.render(str(self.score1), 1, (200, 200, 200))
        text2 = font.render(str(self.score2), 1, (200, 200, 200))
        self.screen.blit(text1, (self.screen_width / 2 - 100, 10))
        self.screen.blit(text2, (self.screen_width / 2 + 70, 10))
        pygame.display.flip()
        pygame.time.Clock().tick(60)

    def close(self):
        pygame.quit()
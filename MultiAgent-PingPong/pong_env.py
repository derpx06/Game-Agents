import pygame
import random
import numpy as np

pygame.init()

class PongEnv:
    def __init__(self, screen_width=1200, screen_height=800,
                 paddle_height=141, base_ball_speed=12, paddle_speed=30,
                 min_rally_sudden_death=5):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.paddle_height = paddle_height
        self.paddle_width = 25

        self.base_ball_speed = base_ball_speed
        self.ball_speed = float(base_ball_speed)
        self.base_paddle_speed = paddle_speed
        self.paddle_speed = paddle_speed

        self.screen = None
        pygame.display.set_caption("Pong")

        # Load assets safely
        try:
            self.ball_img = pygame.image.load("ball.png")
            self.paddle1_img = pygame.image.load("Paddle.png")
            self.paddle2_img = pygame.image.load("Paddle.png")
            self.icon = pygame.image.load("pinball.png")
            pygame.display.set_icon(self.icon)
        except Exception:
            self.ball_img = None
            self.paddle1_img = None
            self.paddle2_img = None
            self.icon = None

        self.score1_font = pygame.font.Font("freesansbold.ttf", 100)
        self.score2_font = pygame.font.Font("freesansbold.ttf", 100)

        self.action_space = [0, 1, 2]  # up, down, stay
        self.state_size = 7

        self.max_paddle_y = self.screen_height - self.paddle_height
        self.ball_y_max = self.screen_height - 16
        self.paddle1_x = 15
        self.paddle2_x = self.screen_width - 15 - self.paddle_width

        self.obstacle_width = 20
        self.obstacle_height = 50
        self.obstacle1_x = self.screen_width // 4
        self.obstacle2_x = 3 * self.screen_width // 4
        self.obstacle1_y = (self.screen_height - self.obstacle_height) // 3
        self.obstacle2_y = 2 * (self.screen_height - self.obstacle_height) // 3

        self.min_rally_sudden_death = min_rally_sudden_death

        self.reset()

    def reset(self):
        self.ball_x = float(self.screen_width) / 2.0
        self.ball_y = float(self.screen_height) / 2.0
        self.ball_speed = float(self.base_ball_speed)

        angle = random.uniform(-0.35, 0.35)
        direction = random.choice([-1, 1])
        vx = direction * np.cos(angle)
        vy = np.sin(angle)
        norm = np.hypot(vx, vy)
        self.ball_vx = (vx / norm) * self.ball_speed
        self.ball_vy = (vy / norm) * self.ball_speed

        self.paddle1_y = (self.screen_height - self.paddle_height) / 2.0
        self.paddle2_y = (self.screen_height - self.paddle_height) / 2.0

        self.score1 = 0
        self.score2 = 0

        self.done = False
        self.rally_count = 0  # resets on miss

        return self.get_state(True), self.get_state(False)

    def get_state(self, is_left=True):
        paddle_y = self.paddle1_y if is_left else self.paddle2_y
        state = np.array([
            self.ball_x / self.screen_width,
            self.ball_y / self.screen_height,
            self.ball_vx / max(1e-6, self.ball_speed),
            self.ball_vy / max(1e-6, self.ball_speed),
            paddle_y / self.screen_height,
            self.obstacle1_y / self.screen_height,
            self.obstacle2_y / self.screen_height
        ], dtype=np.float32)

        if not is_left:
            state[0] = 1.0 - state[0]
            state[2] = -state[2]
        return state

    def step(self, action1, action2):
        # update ball and paddle speeds dynamically
        self.ball_speed = float(min(max(self.base_ball_speed, self.base_ball_speed + 0.2 * (self.score1 + self.score2)), 20.0))
        norm = np.hypot(self.ball_vx, self.ball_vy)
        if norm > 1e-6:
            self.ball_vx = (self.ball_vx / norm) * self.ball_speed
            self.ball_vy = (self.ball_vy / norm) * self.ball_speed
        self.paddle_speed = int(min(self.base_paddle_speed + 0.2 * (self.ball_speed - self.base_ball_speed), 45))

        # move paddles
        self.paddle1_y = max(0.0, min(self.max_paddle_y, self.paddle1_y + [-self.paddle_speed, self.paddle_speed, 0][action1]))
        self.paddle2_y = max(0.0, min(self.max_paddle_y, self.paddle2_y + [-self.paddle_speed, self.paddle_speed, 0][action2]))

        # move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # wall collisions
        if self.ball_y >= self.ball_y_max:
            self.ball_y = float(self.ball_y_max)
            self.ball_vy = -abs(self.ball_vy)
        if self.ball_y <= 0:
            self.ball_y = 0.0
            self.ball_vy = abs(self.ball_vy)

        reward1 = reward2 = 0.0

        # small per-step proximity reward
        center1 = self.paddle1_y + self.paddle_height / 2
        center2 = self.paddle2_y + self.paddle_height / 2
        reward1 += 0.02 * max(0.0, 1.0 - abs(center1 - self.ball_y) / self.screen_height)
        reward2 += 0.02 * max(0.0, 1.0 - abs(center2 - self.ball_y) / self.screen_height)

        # obstacle collisions
        for ox, oy in [(self.obstacle1_x, self.obstacle1_y), (self.obstacle2_x, self.obstacle2_y)]:
            if ox <= self.ball_x <= ox + self.obstacle_width and oy <= self.ball_y <= oy + self.obstacle_height:
                self.ball_vx = -self.ball_vx
                reward1 += 0.1
                reward2 += 0.1

        ball_half = 8
        left_hit = (self.ball_x <= self.paddle1_x + self.paddle_width and
                    self.paddle1_y - ball_half <= self.ball_y <= self.paddle1_y + self.paddle_height + ball_half and
                    self.ball_vx < 0)
        right_hit = (self.ball_x >= self.paddle2_x - self.paddle_width and
                     self.paddle2_y - ball_half <= self.ball_y <= self.paddle2_y + self.paddle_height + ball_half and
                     self.ball_vx > 0)

        # hitting rewards
        hit_base = 10.0
        if left_hit:
            self._bounce_ball(left=True)
            self.rally_count += 1
            reward1 += hit_base + 0.3 * (self.rally_count - 1)
            reward2 -= 0.5
            self.score1 += 1
        elif right_hit:
            self._bounce_ball(left=False)
            self.rally_count += 1
            reward2 += hit_base + 0.3 * (self.rally_count - 1)
            reward1 -= 0.5
            self.score2 += 1

        # miss penalties + small opponent bonus
        sudden_death = (self.rally_count >= self.min_rally_sudden_death)
        if self.ball_x <= -32:
            reward1 += -20.0
            reward2 += 1.0
            if sudden_death: self.done = True
            self.reset_ball()
        elif self.ball_x >= self.screen_width + 32:
            reward2 += -20.0
            reward1 += 1.0
            if sudden_death: self.done = True
            self.reset_ball()

        # optional max score end
        if self.score1 >= 200 or self.score2 >= 200:
            self.done = True
            reward1 += 50.0 if self.score1 >= 200 else -50.0
            reward2 += 50.0 if self.score2 >= 200 else -50.0

        state1 = self.get_state(True)
        state2 = self.get_state(False)
        info = {"score1": self.score1, "score2": self.score2, "rally_count": self.rally_count, "ball_speed": self.ball_speed}
        return state1, state2, reward1, reward2, self.done, info

    def _bounce_ball(self, left=True):
        if left:
            offset = (self.ball_y - (self.paddle1_y + self.paddle_height / 2)) / (self.paddle_height / 2)
            angle = max(-0.9, min(0.9, offset * 0.7 + random.uniform(-0.05, 0.05)))
            vx, vy = abs(np.cos(angle)), np.sin(angle)
        else:
            offset = (self.ball_y - (self.paddle2_y + self.paddle_height / 2)) / (self.paddle_height / 2)
            angle = max(-0.9, min(0.9, -offset * 0.7 + random.uniform(-0.05, 0.05)))
            vx, vy = -abs(np.cos(angle)), np.sin(angle)
        norm = np.hypot(vx, vy)
        self.ball_vx = (vx / norm) * self.ball_speed
        self.ball_vy = (vy / norm) * self.ball_speed

    def reset_ball(self):
        self.ball_x = float(self.screen_width) / 2
        self.ball_y = float(self.screen_height) / 2
        angle = random.uniform(-0.35, 0.35)
        direction = random.choice([-1, 1])
        vx, vy = direction * np.cos(angle), np.sin(angle)
        norm = np.hypot(vx, vy)
        self.ball_vx = (vx / norm) * max(self.base_ball_speed, self.ball_speed * 0.9)
        self.ball_vy = (vy / norm) * max(self.base_ball_speed, self.ball_speed * 0.9)
        self.rally_count = 0  # reset rally count after miss

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        self.screen.fill((0, 0, 0))
        half_w = self.screen_width // 2
        pygame.draw.line(self.screen, (255, 255, 255), (half_w, 0), (half_w, self.screen_height), 7)

        pygame.draw.rect(self.screen, (100, 100, 100), (self.obstacle1_x, self.obstacle1_y, self.obstacle_width, self.obstacle_height))
        pygame.draw.rect(self.screen, (100, 100, 100), (self.obstacle2_x, self.obstacle2_y, self.obstacle_width, self.obstacle_height))

        if self.ball_img:
            self.screen.blit(self.ball_img, (int(self.ball_x), int(self.ball_y)))
        else:
            pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball_x), int(self.ball_y)), 8)

        if self.paddle1_img:
            self.screen.blit(self.paddle1_img, (self.paddle1_x, int(self.paddle1_y)))
            self.screen.blit(self.paddle2_img, (self.paddle2_x, int(self.paddle2_y)))
        else:
            pygame.draw.rect(self.screen, (200, 200, 200), (self.paddle1_x, int(self.paddle1_y), self.paddle_width, self.paddle_height))
            pygame.draw.rect(self.screen, (200, 200, 200), (self.paddle2_x, int(self.paddle2_y), self.paddle_width, self.paddle_height))

        t1 = self.score1_font.render(str(self.score1), True, (255, 0, 0))
        t2 = self.score2_font.render(str(self.score2), True, (0, 0, 255))
        self.screen.blit(t1, (half_w - 200, self.screen_height // 2 - 50))
        self.screen.blit(t2, (half_w + 150, self.screen_height // 2 - 50))

        pygame.display.update()
        pygame.time.Clock().tick(60)

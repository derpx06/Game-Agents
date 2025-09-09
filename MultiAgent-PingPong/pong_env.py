import pygame
import random
import numpy as np
from collections import deque


class PongEnv:
    def __init__(self, screen=None, screen_width=1200, screen_height=800, paddle_height=141,
                 base_ball_speed=14, paddle_speed=45, win_score=11, frame_skip=4):
        self.is_headless = screen is None
        if not self.is_headless:
            pygame.init()
        else:
            pygame.font.init()

        self.screen_width = screen_width;
        self.screen_height = screen_height;
        self.paddle_height = paddle_height
        self.paddle_width = 25;
        self.win_score = win_score;
        self.screen = screen
        self.base_ball_speed = base_ball_speed;
        self.ball_speed = float(base_ball_speed)
        self.base_paddle_speed = paddle_speed;
        self.paddle_speed = paddle_speed
        self.games_played = 0;
        self.state_size = 10;
        self.frame_skip = frame_skip

        self.max_paddle_y = self.screen_height - self.paddle_height
        self.paddle1_x = 15;
        self.paddle2_x = self.screen_width - 15 - self.paddle_width
        self.last_serve_direction = 1
        self.state_buffer = deque(maxlen=4)

        if not self.is_headless: self.font = pygame.font.Font(None, 74)
        self.reset()

    def reset(self):
        self.ball_x = float(self.screen_width) / 2.0;
        self.ball_y = float(self.screen_height) / 2.0
        self.paddle1_y = (self.screen_height - self.paddle_height) / 2.0
        self.paddle2_y = (self.screen_height - self.paddle_height) / 2.0
        self.score1, self.score2, self.done, self.rally_count = 0, 0, False, 0
        self.reset_ball(initial_serve=True)
        initial_state1, _ = self._get_current_state_pair()
        self.state_buffer.clear()
        for _ in range(4): self.state_buffer.append(initial_state1)
        return np.array(list(self.state_buffer)), np.array(list(self.state_buffer))

    def _get_current_state_pair(self):
        s1 = self.get_state(is_left=True);
        s2 = self.get_state(is_left=False)
        return s1, s2

    def get_state(self, is_left=True):
        my_paddle_y = self.paddle1_y if is_left else self.paddle2_y
        opp_paddle_y = self.paddle2_y if is_left else self.paddle1_y
        state = np.array(
            [self.ball_x / self.screen_width, self.ball_y / self.screen_height, self.ball_vx / self.ball_speed,
             self.ball_vy / self.ball_speed, my_paddle_y / self.screen_height, opp_paddle_y / self.screen_height,
             -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        if not is_left: state[0] = 1.0 - state[0]; state[2] = -state[2]
        return state

    def step(self, action1, action2, render=True):
        total_r1, total_r2 = 0.0, 0.0;
        done = False;
        info = {}
        info_accumulator = {'perfect_hit_p1': 0, 'perfect_hit_p2': 0, 'normal_hit_p1': 0, 'normal_hit_p2': 0}

        for _ in range(self.frame_skip):
            r1, r2, done, frame_info = self._internal_step(action1, action2)
            total_r1 += r1;
            total_r2 += r2
            for key in info_accumulator: info_accumulator[key] += frame_info.get(key, 0)
            if render: self.render()
            if done: break

        s1, s2 = self._get_current_state_pair()
        self.state_buffer.append(s1)
        next_state1 = np.array(list(self.state_buffer))
        s2_buffer = list(self.state_buffer);
        s2_buffer[-1] = s2
        next_state2 = np.array(s2_buffer)

        info.update(info_accumulator)
        info.update({"score1": self.score1, "score2": self.score2, "rally_count": self.rally_count,
                     "games_played": self.games_played})
        return next_state1, next_state2, total_r1, total_r2, done, info

    def _internal_step(self, action1, action2):
        self.paddle1_y = max(0.0, min(self.max_paddle_y,
                                      self.paddle1_y + [-self.paddle_speed, self.paddle_speed, 0][action1]))
        self.paddle2_y = max(0.0, min(self.max_paddle_y,
                                      self.paddle2_y + [-self.paddle_speed, self.paddle_speed, 0][action2]))
        self.ball_x += self.ball_vx;
        self.ball_y += self.ball_vy
        r1, r2 = -0.01, -0.01

        info = {'perfect_hit_p1': 0, 'perfect_hit_p2': 0, 'normal_hit_p1': 0, 'normal_hit_p2': 0}

        ball_rect = pygame.Rect(int(self.ball_x), int(self.ball_y), 16, 16)
        paddle_hh = self.paddle_height / 2

        if ball_rect.top <= 0 or ball_rect.bottom >= self.screen_height: self.ball_vy *= -1

        if self.ball_vx < 0 and ball_rect.colliderect(
                pygame.Rect(self.paddle1_x, self.paddle1_y, self.paddle_width, self.paddle_height)):
            offset = (ball_rect.centery - (self.paddle1_y + paddle_hh)) / paddle_hh;
            self._bounce_ball(offset, True);
            self.rally_count += 1
            r1 += 5.0 + (self.rally_count * 1.5);
            if abs(offset) > 0.7:
                info['perfect_hit_p1'] = 1;
                r1 += (abs(offset) - 0.7) * 2
            else:
                info['normal_hit_p1'] = 1
        elif self.ball_vx > 0 and ball_rect.colliderect(
                pygame.Rect(self.paddle2_x, self.paddle2_y, self.paddle_width, self.paddle_height)):
            offset = (ball_rect.centery - (self.paddle2_y + paddle_hh)) / paddle_hh;
            self._bounce_ball(offset, False);
            self.rally_count += 1
            r2 += 5.0 + (self.rally_count * 1.5)
            if abs(offset) > 0.7:
                info['perfect_hit_p2'] = 1;
                r2 += (abs(offset) - 0.7) * 2
            else:
                info['normal_hit_p2'] = 1

        point_scored = False;
        done = False
        if ball_rect.left < 0:
            self.score2 += 1;
            r2 += 20.0;
            r1 -= 15.0;
            point_scored = True
        elif ball_rect.right > self.screen_width:
            self.score1 += 1;
            r1 += 20.0;
            r2 -= 15.0;
            point_scored = True
        if point_scored:
            if self.score1 >= self.win_score or self.score2 >= self.win_score: done = True; self.games_played += 1
            self.reset_ball()

        info.update({"score1": self.score1, "score2": self.score2, "rally_count": self.rally_count,
                     "games_played": self.games_played})
        return r1, r2, done, info

    def _bounce_ball(self, offset, left=True):
        angle = offset * (np.pi / 3.5);
        direction = 1 if left else -1
        self.ball_vx = direction * self.ball_speed * np.cos(angle);
        self.ball_vy = self.ball_speed * np.sin(angle)
        self.ball_speed = min(self.base_ball_speed * 2, self.ball_speed * 1.01)

    def reset_ball(self, initial_serve=False):
        # --- MODIFICATION: New serving logic ---
        self.rally_count = 0
        self.ball_speed = self.base_ball_speed

        # 1. Start the ball at the bottom of the screen, horizontally centered
        self.ball_x = self.screen_width / 2
        self.ball_y = random.uniform(self.screen_height * 0.8, self.screen_height - 16)  # -16 for ball height

        # 2. Serve upwards at a wide, random angle (from -135 to -45 degrees)
        # In Pygame, a negative VY moves the ball up the screen.
        angle = random.uniform(-np.pi * 3 / 4, -np.pi / 4)

        # 3. The direction (left/right) still alternates
        if not initial_serve: self.last_serve_direction *= -1
        direction = self.last_serve_direction

        self.ball_vx = direction * self.ball_speed * np.cos(angle)
        self.ball_vy = self.ball_speed * np.sin(angle)
        # --- END OF MODIFICATION ---

    def render(self):
        if self.is_headless or self.screen is None: return
        self.screen.fill((21, 21, 21))
        pygame.draw.line(self.screen, (50, 50, 50), (self.screen_width / 2, 0),
                         (self.screen_width / 2, self.screen_height), 4)
        pygame.draw.rect(self.screen, (200, 200, 200),
                         (self.paddle1_x, int(self.paddle1_y), self.paddle_width, self.paddle_height))
        pygame.draw.rect(self.screen, (200, 200, 200),
                         (self.paddle2_x, int(self.paddle2_y), self.paddle_width, self.paddle_height))
        pygame.draw.ellipse(self.screen, (220, 220, 100), (int(self.ball_x), int(self.ball_y), 16, 16))
        t1 = self.font.render(str(self.score1), 1, (200, 200, 200));
        t2 = self.font.render(str(self.score2), 1, (200, 200, 200))
        self.screen.blit(t1, (self.screen_width / 2 - 100, 10));
        self.screen.blit(t2, (self.screen_width / 2 + 70, 10))
        pygame.display.flip()

    def close(self):
        pygame.quit()
import pygame
import numpy as np
from collections import deque
from pong_env import PongEnv
from PongAgent import PongAgent
from PongTrainer import PongTrainer
import time


class GameManager:
    def __init__(self):
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Advanced Pong AI Dashboard")
        self.font = pygame.font.Font(None, 74)
        self.small_font = pygame.font.Font(None, 45)
        self.stats_font = pygame.font.Font(None, 36)
        self.notification_font = pygame.font.Font(None, 50)

    def show_main_menu(self):
        """Displays the main menu and handles user selection."""
        menu_options = {1: "Interactive Mode", 2: "Spectator Mode", 3: "Advanced Training", 4: "Quit"}
        while True:
            self.screen.fill((21, 21, 21))
            title_text = self.font.render("Pong AI Trainer", True, (220, 220, 100))
            self.screen.blit(title_text, (self.screen_width / 2 - title_text.get_width() / 2, 100))
            for i, text in menu_options.items():
                option_text = self.small_font.render(f"{i}. {text}", True, (200, 200, 200))
                self.screen.blit(option_text, (self.screen_width / 2 - option_text.get_width() / 2, 250 + i * 60))
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return 4
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1: return 1
                    if event.key == pygame.K_2: return 2
                    if event.key == pygame.K_3: return 3
                    if event.key == pygame.K_4 or event.key == pygame.K_ESCAPE: return 4

    def display_notification(self, message, duration_ms=2000):
        """Displays a temporary notification on the dashboard."""
        start_time = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start_time < duration_ms:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return

            notification_text = self.notification_font.render(message, True, (21, 21, 21))
            text_rect = notification_text.get_rect(center=(self.screen_width / 2, 50))
            bg_rect = text_rect.inflate(40, 20)
            pygame.draw.rect(self.screen, (100, 220, 100), bg_rect, border_radius=15)
            self.screen.blit(notification_text, text_rect)
            pygame.display.flip()

    def draw_live_training_ui(self, episode, stats):
        self.screen.fill((21, 21, 21))
        title = self.font.render("Advanced Training Dashboard", True, (220, 220, 100))
        self.screen.blit(title, (self.screen_width / 2 - title.get_width() / 2, 100))

        # --- Progress Bar ---
        progress = episode / stats['max_episodes']
        bar_w = self.screen_width - 400;
        bar_h = 40;
        px = 200;
        py = 250;
        fill_w = int(bar_w * progress)
        pygame.draw.rect(self.screen, (50, 50, 50), (px, py, bar_w, bar_h))
        if fill_w > 0: pygame.draw.rect(self.screen, (220, 220, 100), (px, py, fill_w, bar_h))
        percent_text = self.stats_font.render(f"Episode {episode}/{stats['max_episodes']} ({progress:.1%})", True,
                                              (200, 200, 200))
        self.screen.blit(percent_text, (self.screen_width / 2 - percent_text.get_width() / 2, py + 5))

        # --- Key Performance Indicators (KPIs) ---
        kpi_labels = ["Total Games", "Max Rally", "Perfect Hits", "Avg Reward Diff", "Epsilon"]
        total_hits = stats['perfect_hits'] + stats['normal_hits']
        perfect_hit_pct = (stats['perfect_hits'] / total_hits * 100) if total_hits > 0 else 0
        kpi_values = [f"{stats['games_completed']}", f"{stats['max_rally']}", f"{perfect_hit_pct:.1f}%",
                      f"{stats['avg_reward_diff']:.2f}", f"{stats['epsilon']:.3f}"]

        # Display KPIs in two columns for a clean layout
        for i, (label, value) in enumerate(zip(kpi_labels, kpi_values)):
            col = i % 3
            row = i // 3

            x_pos = self.screen_width / 2 - 350 + col * 350
            y_pos = 400 + row * 100

            label_text = self.stats_font.render(label, True, (150, 150, 150))
            value_text = self.small_font.render(value, True, (200, 200, 200))
            self.screen.blit(label_text, (x_pos, y_pos))
            self.screen.blit(value_text, (x_pos, y_pos + 30))

        footer = self.stats_font.render("Press ESC to stop training and return to menu", True, (150, 150, 150))
        self.screen.blit(footer, (self.screen_width / 2 - footer.get_width() / 2, self.screen_height - 100))
        pygame.display.flip()

    def run_advanced_training(self, trainer):
        stats = {'max_episodes': 10000, 'games_completed': 0, 'max_rally': 0, 'perfect_hits': 0,
                 'normal_hits': 0, 'avg_reward_diff': 0, 'epsilon': 1.0, 'reward_history': deque(maxlen=100)}

        for episode in range(stats['max_episodes']):
            for event in pygame.event.get():
                if event.type == pygame.QUIT: pygame.quit(); return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return

            episode_stats = trainer.run_episode(training=True, render=False, interactive_player=None, game_speed=0)

            if episode_stats is None: return

            stats['games_completed'] = episode_stats['games_played']
            stats['max_rally'] = max(stats['max_rally'], episode_stats['max_rally'])
            stats['perfect_hits'] += episode_stats['perfect_hits']
            stats['normal_hits'] += episode_stats['normal_hits']
            stats['reward_history'].append(episode_stats['reward_diff'])
            stats['avg_reward_diff'] = np.mean(stats['reward_history'])
            stats['epsilon'] = trainer.agent1.epsilon
            self.draw_live_training_ui(episode + 1, stats)

            if trainer.check_for_save(stats['avg_reward_diff']):
                self.display_notification(f"New best model saved at Episode {episode + 1}!")

    def run(self):
        while True:
            choice = self.show_main_menu()
            if choice == 4: break

            env = PongEnv(screen=self.screen)
            agent1 = PongAgent(model_path='agent1.pth')
            agent2 = PongAgent(model_path='agent2.pth')
            trainer = PongTrainer(env, agent1, agent2)

            if choice == 1:
                trainer.run_loop(render=True, training=True, interactive_player='player2', game_speed=20)
            elif choice == 2:
                trainer.run_loop(render=True, training=True, interactive_player=None, game_speed=70)
            elif choice == 3:
                self.run_advanced_training(trainer)
        pygame.quit()
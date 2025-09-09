import pygame
import numpy as np
from collections import deque


class PongTrainer:
    def __init__(self, env, agent1, agent2, batch_size=512):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.batch_size = batch_size
        self.best_avg_reward = -np.inf
        self.reward_history_for_saving = deque(maxlen=100)
        self.clock = pygame.time.Clock()

    def get_human_action(self):
        """Checks keyboard state for human player input."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            return 0
        elif keys[pygame.K_DOWN]:
            return 1
        return 2

    def run_episode(self, training, render, interactive_player, game_speed):
        """Runs a single episode of the game with the specified configuration."""
        state1, state2 = self.env.reset()
        total_r1, total_r2 = 0, 0
        episode_stats = {'max_rally': 0, 'perfect_hits': 0, 'normal_hits': 0}
        done = False

        while not done:
            # --- FIX: Event loop now runs ALWAYS to keep the window responsive ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.env.close(); raise SystemExit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("Returning to main menu...")
                    return None  # Signal an exit

            action1 = self.agent1.choose_action(state1) if interactive_player != 'player1' else self.get_human_action()
            action2 = self.agent2.choose_action(state2) if interactive_player != 'player2' else self.get_human_action()

            # Pass the render flag to the env to control frame skipping visuals
            next_state1, next_state2, r1, r2, done, info = self.env.step(action1, action2, render)

            episode_stats['max_rally'] = max(episode_stats['max_rally'], info['rally_count'])
            episode_stats['perfect_hits'] += info['perfect_hit_p1'] + info['perfect_hit_p2']
            episode_stats['normal_hits'] += info['normal_hit_p1'] + info['normal_hit_p2']

            if training:
                self.agent1.store_experience(state1, action1, r1, next_state1, done)
                self.agent2.store_experience(state2, action2, r2, next_state2, done)
                if len(self.agent1.replay_buffer) > self.batch_size * 10:
                    self.agent1.train(self.batch_size);
                    self.agent2.train(self.batch_size)

            state1, state2 = next_state1, next_state2
            total_r1 += r1;
            total_r2 += r2

            # Drawing is still conditional
            if render:
                self.clock.tick(game_speed)

        if training:
            self.agent1.update_epsilon();
            self.agent2.update_epsilon()

        episode_stats['reward_diff'] = total_r1 - total_r2
        episode_stats['games_played'] = self.env.games_played
        return episode_stats

    def check_for_save(self, current_avg_reward):
        """Checks if the model should be saved and saves it."""
        self.reward_history_for_saving.append(current_avg_reward)
        avg_for_saving = np.mean(self.reward_history_for_saving)
        if len(self.reward_history_for_saving) >= 100 and avg_for_saving > self.best_avg_reward:
            self.best_avg_reward = avg_for_saving
            self.agent1.save_model('agent1.pth');
            self.agent2.save_model('agent2.pth')
            return True
        return False

    def run_loop(self, render, training, interactive_player, game_speed):
        """Legacy loop for simple modes (Interactive, Spectator)."""
        while True:
            episode_stats = self.run_episode(training, render, interactive_player, game_speed)
            if episode_stats is None:
                return  # Exit if ESC was pressed
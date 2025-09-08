import pygame
import numpy as np
from collections import deque


class PongTrainer:
    def __init__(self, env, agent1, agent2, max_episodes=50000, max_steps=2000, batch_size=512):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.best_avg_reward = float('-inf')
        self.sequence_length = agent1.sequence_length
        self.reward_history = deque(maxlen=100)

    def get_stacked_state(self, state_deque):
        return np.array(list(state_deque))

    def run_episode(self, training=True):
        state1_single, state2_single = self.env.reset()

        state1_hist = deque([state1_single] * self.sequence_length, maxlen=self.sequence_length)
        state2_hist = deque([state2_single] * self.sequence_length, maxlen=self.sequence_length)

        total_reward1, total_reward2, steps = 0, 0, 0
        done = False

        while not done and steps < self.max_steps:
            if not training:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: self.env.close(); raise SystemExit

            stacked_state1 = self.get_stacked_state(state1_hist)
            stacked_state2 = self.get_stacked_state(state2_hist)

            action1 = self.agent1.choose_action(stacked_state1)
            action2 = self.agent2.choose_action(stacked_state2)

            next_state1_single, next_state2_single, reward1, reward2, done, info = self.env.step(action1, action2)

            state1_hist.append(next_state1_single)
            state2_hist.append(next_state2_single)

            next_stacked_state1 = self.get_stacked_state(state1_hist)
            next_stacked_state2 = self.get_stacked_state(state2_hist)

            if training:
                self.agent1.store_experience(stacked_state1, action1, reward1, next_stacked_state1, done)
                self.agent2.store_experience(stacked_state2, action2, reward2, next_stacked_state2, done)
                if steps > self.batch_size and steps % 4 == 0:
                    self.agent1.train(self.batch_size)
                    self.agent2.train(self.batch_size)
                self.agent1.update_epsilon()
                self.agent2.update_epsilon()

            total_reward1 += reward1
            total_reward2 += reward2
            steps += 1

            if not training: self.env.render()

        return info['score1'], info['score2'], total_reward1, total_reward2, info['rally_count'], info['games_played']

    def train(self):
        for episode in range(self.max_episodes):
            score1, score2, r1, r2, rally, games = self.run_episode(training=True)

            self.reward_history.append(r1 - r2)  # Track reward difference
            avg_reward_diff = np.mean(self.reward_history)

            if episode % 20 == 0:
                print(
                    f"Ep {episode:<5} | Games: {games:<3} | Score: {score1}-{score2} | Avg Reward Diff: {avg_reward_diff:<8.2f} | "
                    f"Rally: {rally:<3} | Epsilon: {self.agent1.epsilon:.3f}")

            if avg_reward_diff > self.best_avg_reward and len(self.reward_history) >= 100:
                self.best_avg_reward = avg_reward_diff
                print(f"*** New best avg reward diff: {avg_reward_diff:.2f}! Saving models... ***")
                self.agent1.save_model('sota_agent1.pth')
                self.agent2.save_model('sota_agent2.pth')

    def test(self):
        print("\n--- Testing Trained SOTA Agents ---")
        self.agent1.epsilon = 0
        self.agent2.epsilon = 0
        score1, score2, r1, r2, rally, games = self.run_episode(training=False)
        print(f"Final Test Game -> Score: {score1}-{score2}")
import pygame
from pong_env import PongEnv
from PongAgent import PongAgent

class PongTrainer:
    def __init__(self, env, agent1, agent2, max_episodes=10000, max_steps=1000, batch_size=1024):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.best_score_diff = float('-inf')

    def run_episode(self, render=False):
        state1, state2 = self.env.reset()
        total_reward1 = total_reward2 = 0
        steps = 0
        done = False

        while not done and steps < self.max_steps:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.env.close()
                        raise SystemExit

            # agents choose actions
            action1 = self.agent1.choose_action(state1)
            action2 = self.agent2.choose_action(state2)

            # environment step
            next_state1, next_state2, reward1, reward2, done, info = self.env.step(action1, action2)

            # store experiences
            self.agent1.store_experience(state1, action1, reward1, next_state1, done)
            self.agent2.store_experience(state2, action2, reward2, next_state2, done)

            # train agents
            if steps % 2 == 0:
                self.agent1.train(self.batch_size)
                self.agent2.train(self.batch_size)

            # update states and rewards
            state1, state2 = next_state1, next_state2
            total_reward1 += reward1
            total_reward2 += reward2
            steps += 1

            if render:
                self.env.render()

        # epsilon decay
        self.agent1.update_epsilon()
        self.agent2.update_epsilon()

        # save best model
        score_diff = info['score1'] - info['score2']
        if score_diff > self.best_score_diff:
            self.best_score_diff = score_diff
            self.agent1.save_model('best_agent1.pth')
            self.agent2.save_model('best_agent2.pth')

        return info['score1'], info['score2'], total_reward1, total_reward2, info['rally_count'], info['ball_speed']

    def train(self):
        for episode in range(self.max_episodes):
            score1, score2, reward1, reward2, rally_count, ball_speed = self.run_episode(render=True)
            if episode % 100 == 0:
                print(f"Episode {episode}, Score: {score1}-{score2}, "
                      f"Rewards: {reward1:.1f}, {reward2:.1f}, "
                      f"Rally Count: {rally_count}, Ball Speed: {ball_speed:.1f}")

    def test(self):
        print("Testing trained agents...")
        score1, score2, reward1, reward2, rally_count, ball_speed = self.run_episode(render=True)
        print(f"Test Episode, Score: {score1}-{score2}, "
              f"Rewards: {reward1:.1f}, {reward2:.1f}, "
              f"Rally Count: {rally_count}, Ball Speed: {ball_speed:.1f}")


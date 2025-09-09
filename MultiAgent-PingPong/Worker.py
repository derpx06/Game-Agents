import torch.multiprocessing as mp
import numpy as np
from collections import deque
from pong_env import PongEnv
from PongAgent import PongAgent
import pygame


class Worker(mp.Process):
    def __init__(self, worker_id, central_agent, experience_queue, stats_queue):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.central_agent_state = central_agent.agent.state_dict()
        self.experience_queue = experience_queue
        self.stats_queue = stats_queue
        self.sequence_length = central_agent.sequence_length
        self.state_size = central_agent.state_size

    def run(self):
        # --- WORKER RUNS HEADLESS (NO SCREEN) ---
        # Pygame is still initialized for fonts, but no display is created.
        pygame.font.init()
        self.env = PongEnv(screen=None)  # Explicitly pass None for screen
        self.local_agent1 = PongAgent(state_size=self.state_size, action_size=3, sequence_length=self.sequence_length)
        self.local_agent2 = PongAgent(state_size=self.state_size, action_size=3, sequence_length=self.sequence_length)

        try:
            while True:
                self.local_agent1.agent.load_state_dict(self.central_agent_state)
                self.local_agent2.agent.load_state_dict(self.central_agent_state)

                state1_single, state2_single = self.env.reset()
                state1_hist = deque([state1_single] * self.sequence_length, maxlen=self.sequence_length)
                state2_hist = deque([state2_single] * self.sequence_length, maxlen=self.sequence_length)

                done = False
                steps = 0
                max_rally_in_episode = 0

                while not done and steps < 2500:
                    stacked_state1 = np.array(list(state1_hist))
                    stacked_state2 = np.array(list(state2_hist))
                    action1 = self.local_agent1.choose_action(stacked_state1)
                    action2 = self.local_agent2.choose_action(stacked_state2)

                    next_state1_single, next_state2_single, reward1, reward2, done, info = self.env.step(action1,
                                                                                                         action2)

                    # Send live, per-step stats
                    self.stats_queue.put((self.worker_id, 'rally', info['rally_count']))
                    self.stats_queue.put((self.worker_id, 'score', (info['score1'], info['score2'])))

                    max_rally_in_episode = max(max_rally_in_episode, info['rally_count'])
                    state1_hist.append(next_state1_single)
                    state2_hist.append(next_state2_single)
                    next_stacked_state1 = np.array(list(state1_hist))
                    next_stacked_state2 = np.array(list(state2_hist))

                    self.experience_queue.put((stacked_state1, action1, reward1, next_stacked_state1, done))
                    self.experience_queue.put((stacked_state2, action2, reward2, next_stacked_state2, done))
                    steps += 1

                # After episode ends, send summary stats
                self.stats_queue.put((self.worker_id, 'game_completed', 1))
                self.stats_queue.put((self.worker_id, 'max_rally', max_rally_in_episode))

        except KeyboardInterrupt:
            pass  # Allow graceful exit
        finally:
            self.env.close()
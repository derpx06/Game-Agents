import pygame
import numpy as np
import matplotlib.pyplot as plt
from Game import SnakeGame
import torch
import sys
from Agent import Agent

def main():
    pygame.init()
    check_errors = pygame.init()
    if check_errors[1] > 0:
        print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
        sys.exit(-1)

    # score tracking
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # loss tracking
    plot_losses = []
    plot_mean_losses = []

    agent = Agent()
    game = SnakeGame()

    # ---- Figure 1: Scores ----
    plt.ion()
    fig1 = plt.figure(figsize=(8, 4))
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('Game')
    ax1.set_ylabel('Score')
    ax1.set_title('Learning Curve (Scores)')
    line1, = ax1.plot([], [], label='Score')
    line2, = ax1.plot([], [], label='Mean Score')
    ax1.legend()

    # ---- Figure 2: Losses ----
    fig2 = plt.figure(figsize=(8, 4))
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel('Game')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss')
    line3, = ax2.plot([], [], label='Loss')
    line4, = ax2.plot([], [], label='Mean Loss')
    ax2.legend()

    # ---- Training Loop ----
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # short memory step (we don't plot its loss to keep noise down)
        _ = agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1

            long_loss = agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            if long_loss is not None:
                loss_value = float(long_loss)  # trainer returns a float already
            else:
                loss_value = None

            print(
                f'Game {agent.n_games:4d} | Score: {score:3d} | Record: {record:3d} '
                f'| Epsilon: {agent.epsilon:.3f} | Loss: {loss_value if loss_value is not None else "N/A"}'
            )

            plot_scores.append(score)
            total_score += score
            plot_mean_scores.append(total_score / agent.n_games)

            line1.set_xdata(range(len(plot_scores)))
            line1.set_ydata(plot_scores)
            line2.set_xdata(range(len(plot_mean_scores)))
            line2.set_ydata(plot_mean_scores)
            ax1.relim()
            ax1.autoscale_view()
            fig1.canvas.draw()
            fig1.canvas.flush_events()



            plt.pause(0.01)

    # unreachable normally, kept for completeness
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()

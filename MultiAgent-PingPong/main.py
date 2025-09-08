from pong_env import PongEnv
from PongAgent import PongAgent
from PongTrainer import PongTrainer


def main():
    print("Welcome to the SOTA Pong AI Trainer!")
    print("Architecture: Conv1D Dueling DDQN + PER + Advanced Rewards")

    env = PongEnv(screen_width=1200, screen_height=800, paddle_height=141, base_ball_speed=15, paddle_speed=45)

    # STATE_SIZE is now 8 to include the opponent's paddle position.
    STATE_SIZE = 8
    ACTION_SIZE = 3
    SEQUENCE_LENGTH = 4  # Number of frames to stack for temporal context

    # Optimized hyperparameters for the new architecture
    agent1 = PongAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        lr=0.0001,  # Lower LR for more stable learning with a complex net
        gamma=0.99,
        buffer_capacity=200000,  # Larger buffer for more diverse experience
        model_path='sota_agent1.pth'
    )
    agent2 = PongAgent(
        state_size=STATE_SIZE,
        action_size=ACTION_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        lr=0.0001,
        gamma=0.99,
        buffer_capacity=200000,
        model_path='sota_agent2.pth'
    )

    trainer = PongTrainer(env, agent1, agent2, max_episodes=50000, max_steps=2000, batch_size=512)
    trainer.train()
    trainer.test()
    env.close()


if __name__ == "__main__":
    main()
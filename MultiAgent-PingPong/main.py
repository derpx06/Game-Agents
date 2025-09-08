from pong_env import PongEnv
from PongAgent import PongAgent
from PongTrainer import PongTrainer

def main():
    print("Hello from multiagent-pingpong with Dueling DDQN + PER!")
    env = PongEnv(screen_width=1200, screen_height=800, paddle_height=141, base_ball_speed=15, paddle_speed=15)

    agent1 = PongAgent(state_size=7, action_size=3, lr=0.0005, gamma=0.99, buffer_capacity=100000, model_path='best_agent1.pth')
    agent2 = PongAgent(state_size=7, action_size=3, lr=0.0005, gamma=0.99, buffer_capacity=100000, model_path='best_agent2.pth')

    trainer = PongTrainer(env, agent1, agent2, max_episodes=10000, max_steps=1000, batch_size=1024)
    trainer.train()
    trainer.test()
    env.close()

if __name__ == "__main__":
    main()


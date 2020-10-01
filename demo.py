from dqn import DQN

player = DQN(
    environment='Seaquest-ram-v0',
    num_iterations=50000,
    learning_rate=0.001, 
    log_interval=1000, 
    num_eval_episodes=10, 
    eval_interval=5000)

player.train()
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from env import TicTacEnv


def main():
    env = TicTacEnv()
    check_env(env)
    
    env = DummyVecEnv([lambda: env for _ in range(4)])
    
    model = DQN("MlpPolicy", env, learning_rate=1e-4, batch_size= 32, verbose=1)
    model.learn(2000000)
    model.save('x_model')
    
    print("model saved")
    
if __name__ == '__main__':
    main()
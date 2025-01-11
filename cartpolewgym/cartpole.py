import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time

envName = 'CartPole-v1'
env = gym.make(envName, render_mode='human')

(state, _) = env.reset()


############ Random Sampling ##################

# episodeNumber = 5
# timeSteps = 100

# for episodeIndex in range(episodeNumber):
#    initial_state = env.reset()
#    print(episodeIndex)
#    env.render()
#    appendedObservations = []
#    for timeIndex in range(timeSteps):
#        print(timeIndex)
#        random_action = env.action_space.sample()
#        observation, reward, terminated, truncated, info = env.step(
#            random_action)
#        appendedObservations.append(observation)
#        time.sleep(0.1)
#        if (terminated):
#            time.sleep(1)
#            break
# env.close()


env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose=1)

# let the model learn and save it
# model.learn(total_timesteps= 20000)
# model.save('PPO')

# load the model
model = model.load('PPO_model.zip')

# evaluate with render
evaluate_policy(model, env, n_eval_episodes=10, render=True)

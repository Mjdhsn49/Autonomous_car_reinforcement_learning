import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from agent_model import DQNAgent
from carla_env import CarlaEnvironment
import random, time
import numpy as np
from threading import Thread
import utils
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


"""Code taken/implemented from https://pythonprogramming.net/reinforcement-learning-self-driving-autonomous-cars-carla-python/"""

if __name__ == '__main__':
    FPS = 60
    ep_rewards = []
    random.seed(1)
    np.random.seed(1)
    episodes = 5000
    epsilon_min = 0.0001
    epsilon_dec = 0.96
    epsilon = 1
    model_name = 'carla'
    total_epsilon,total_episode = [],[]

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    env = CarlaEnvironment()
    agent = DQNAgent()

    trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
    trainer_thread.start()
    while not agent.training_initialized:
        time.sleep(0.01)
    agent.get_qs(np.ones((1, 600, 400, 3)))

    # Iterate over episodes
    for episode in range(episodes):
        env.collision_sensor_list = []
        agent.tensorboard.step = episode
        episode_reward = 0
        step = 1
        state = env.reset()
        done = False
        episode_start = time.time()
        while True:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(state))
            else:
                action = agent.random_action()
                time.sleep(1/FPS)

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            agent.add_memory((state, action, reward, new_state, done))
            state = new_state
            step += 1

            if done:
                break
        for actor in env.actor_list:
            actor.destroy()

        ep_rewards.append(episode_reward)

        if epsilon > epsilon_min:
            epsilon *= epsilon_dec
            epsilon = max(epsilon_min, epsilon)

        print('Episode: {}'.format(episode),
              'Reward: {}'.format((sum(ep_rewards)/len(ep_rewards))),
              'Epsilon: {}'.format(epsilon))

        total_epsilon.append(epsilon)
        total_episode.append(episode)

        if episode % 50 == 0:
            min_reward = min(ep_rewards[-10:])
            """Plot epsilon over episode"""
            #print(total_epsilon,total_episode,ep_rewards)
            #utils.plot_overtime(total_episode, ep_rewards,total_epsilon) #Reward over time
            agent.model.save('models/model{}.model'.format(episode))

    agent.terminate = True
    trainer_thread.join()
    agent.model.save('model{}.model'.format(episode))
    utils.plot_overtime(total_episode, ep_rewards,total_epsilon)

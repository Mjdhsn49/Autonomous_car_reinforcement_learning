from keras.models import load_model
import numpy as np 
from carla_env import CarlaEnvironment
import random
import cv2

env = CarlaEnvironment()

#https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Deep%20Q%20Learning/Doom/Deep%20Q%20learning%20with%20Doom.ipynb

model = load_model('models/model450.model')
model.predict(np.ones((1,600,400,3))) #Start prediction to make it faster
while True:
	print('Starting Episode')
	ep_reward = []
	state = env.reset()
	env.collision_sensor_list = []
	done = False
	while True:
		cv2.imshow('Agent-Playing', state)
		cv2.waitKey(1)
		Qs = model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]
		action = np.argmax(Qs)
		new_state, reward, done, info = env.step(action)
		state = new_state
		ep_reward.append(reward)
		if done:
			print('Episode Ended')
			break

		print('Reward: {}'.format(sum(ep_reward) / len(ep_reward)))

	#Destroy each actor that is inside the actor list
	for actor in env.actor_list:
		actor.destroy()
		print('Vehicle destroyed')

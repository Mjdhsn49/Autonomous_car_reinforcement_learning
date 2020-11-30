from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Flatten, Dropout, Dense, GlobalAveragePooling2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import tensorflow as tf 
import keras
import numpy as np
import time
import random
from collections import deque

"""Code taken/implemented from https://pythonprogramming.net/reinforcement-learning-self-driving-autonomous-cars-carla-python/"""

from keras.callbacks import TensorBoard
# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DQNAgent:
    def __init__(self):
        self.model = load_model('models/model450.model')
        #self.model = self.create_model()
        #self.target_model = self.create_model()
        self.target_model = load_model('models/model450.model')

        self.replay_memory = deque(maxlen=1000000)

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{'carla'}-{int(time.time())}")
        self.target_update_counter = 0
        self.graph = tf.get_default_graph()

        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        model = Sequential()

        model.add(Dense(64, input_shape=(600,400,3), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Flatten())
        model.add(Dense(3))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])

        return model
    def add_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def random_action(self):
        action = np.random.randint(0, 3)
        return action

    def train(self):
        #Maybe change this to 1000 if anything goes bad

        if len(self.replay_memory) < 64:
            return

        minibatch = random.sample(self.replay_memory, 16)

        current_states = np.array([transition[0] for transition in minibatch])/255
        with self.graph.as_default():
            current_qs_list = self.model.predict(current_states, 1)

        new_current_states = np.array([transition[3] for transition in minibatch])/255
        with self.graph.as_default():
            future_qs_list = self.target_model.predict(new_current_states, 1)

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + 0.99 * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        with self.graph.as_default():
            self.model.fit(np.array(X)/255, np.array(y), batch_size=4, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)


        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > 5:
            self.target_model = self.model
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(1,600,400,3)/255)[0]

    def train_in_loop(self):
        X = np.random.uniform(size=(1, 600, 400, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)
        with self.graph.as_default():
            self.model.fit(X,y, verbose=0, batch_size=1)

        self.training_initialized = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)



from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from GlobalVariables import GlobalVariables
from keras.layers import Dense, Conv2D, Flatten,Conv1D, MaxPooling2D,Convolution2D,GlobalAveragePooling2D
from keras import optimizers
import random
import numpy as np
from keras.layers import MaxPooling1D,GlobalAveragePooling1D,Dropout,LSTM,TimeDistributed,AveragePooling1D,Embedding,Activation

parameter=GlobalVariables
grid_size=GlobalVariables
options=GlobalVariables

class DQNAgent:
    def __init__(self,env):
        if(options.use_samples):
            self.state_dim=parameter.sample_state_size
        elif(options.use_pitch):
            self.state_dim = parameter.pitch_state_size
        elif(options.use_spectrogram):
            self.state_dim = parameter.spectrogram_state_size
        else:
            self.state_dim = parameter.raw_data_state_size
        self.action_dim=parameter.action_size
        self.memory = deque(maxlen=2000)
        self.discount_factor = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        if(options.use_dense):
            self.model = self._build_model()
        else:
            self.model=self._build_CNN_model()
        print(self.model.summary())

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_dim,), kernel_initializer='uniform', activation='relu'))
        model.add(Dense(24, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(parameter.action_size, activation='softmax'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
        return model

    def _build_CNN_model(self):
        model = Sequential()
        model.add(Conv1D(64, 3, activation='relu', input_shape=(self.state_dim, 1)))
        model.add(Conv1D(32, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(parameter.action_size, activation='softmax'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='sgd')
        return model

    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, env):
        # Epsilon-greedy agent policy
        if random.uniform(0, 1) < self.epsilon:
            # explore
            return np.random.choice(env.allowed_actions())
        else:
            # exploit on allowed actions
            state = env.state;
            actions_allowed = env.allowed_actions()
            Q_s = self.Q[state[0], state[1], actions_allowed]
            actions_greedy = actions_allowed[np.flatnonzero(Q_s == np.max(Q_s))]
            return np.random.choice(actions_greedy)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(parameter.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.discount_factor * np.max(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
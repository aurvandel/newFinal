import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Conv3D
from keras import backend


class DQNetwork:
    def __init__(self, actions, input_shape,
                 minibatch_size=32,
                 learning_rate=0.00025,
                 discount_factor=0.99,
                 dropout_prob=0.1,
                 load_path=None,
                 logger=None,
                 n_stack=4):

        # Parameters
        backend.set_image_data_format('channels_first')
        self.actions = actions  # Size of the network output
        self.discount_factor = discount_factor  # Discount factor of the MDP
        self.minibatch_size = minibatch_size  # Size of the training batches
        self.learning_rate = learning_rate  # Learning rate
        self.dropout_prob = dropout_prob  # Probability of dropout
        self.logger = logger
        self.training_history_csv = 'training_history.csv'
        self.stack_count = n_stack
        self.kernel_size = (self.stack_count, 4, 4)
        
        print(input_shape)
        if self.logger is not None:
            self.logger.to_csv(self.training_history_csv, 'Loss,Accuracy')

        self.model = Sequential()

        # First convolutional layer
        self.model.add(Conv3D(32, kernel_size=self.kernel_size,
                              padding='same',
                              activation='relu',
                              input_shape=input_shape))

        # Second convolutional layer
        self.kernel_size = (self.stack_count, 3, 3)
        print(input_shape)
        self.model.add(Conv3D(64, kernel_size=self.kernel_size,
                             padding='same',
                             activation='relu',
                             input_shape=input_shape))

        # Third convolutional layer
        self.kernel_size = (self.stack_count, 2, 2)
        self.model.add(Conv3D(128, kernel_size=self.kernel_size,
                             padding='same',
                             activation='relu',
                             input_shape=input_shape))

        # Flatten the convolution output
        self.model.add(Flatten())

        # First dense layer
        self.model.add(Dense(512, activation='relu'))

        # Output layer
        self.model.add(Dense(self.actions, activation="softmax"))

        # Load the network weights from saved model
        if load_path is not None:
            self.load(load_path)

        self.model.compile(loss='mean_squared_error',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

        print(self.model.summary())

    def train(self, batch, DQN_target):
        x_train = []
        t_train = []

        # Generate training inputs and targets
        for datapoint in batch:
            # Inputs are the states
            x_train.append(datapoint['source'].astype(np.float64))

            # Apply the DQN or DDQN Q-value selection
            next_state = datapoint['dest'].astype(np.float64)
            next_state_pred = DQN_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_pred)

            # The error must be 0 on all actions except the one taken
            t = list(self.predict(datapoint['source'])[0])
            if datapoint['final']:
                t[datapoint['action']] = datapoint['reward']
            else:
                t[datapoint['action']] = datapoint['reward'] + \
                                         self.discount_factor * next_q_value
            t_train.append(t)

        # Prepare inputs and targets
        x_train = np.asarray(x_train).squeeze()
        t_train = np.asarray(t_train).squeeze()

        # Train the model for one epoch
        h = self.model.fit(x_train,
                           t_train,
                           batch_size=self.minibatch_size,
                           epochs=1)

        # Log loss and accuracy

        if self.logger is not None:
            self.logger.to_csv(self.training_history_csv,
                               [h.history['loss'][0], h.history['accuracy'][0]])

    def predict(self, state):
        state = state.astype(np.float64)
        return self.model.predict(state, batch_size=1)

    def save(self, filename=None, append=''):
        f = ('model%s.h5' % append) if filename is None else filename
        if self.logger is not None:
            self.logger.log('Saving model as %s' % f)
        self.model.save_weights(self.logger.path + f)

    def load(self, path):
        if self.logger is not None:
            self.logger.log('Loading weights from file...')
        self.model.load_weights(path)

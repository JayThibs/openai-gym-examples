import gym
from gym import wrappers
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
# import keras
# from keras.layers.core import input_data, dropout, fully_connected
# from keras.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3  # learning rate
env = gym.make('CartPole-v0').env  # environment
goal_steps = 500
# The score requirement threshold is the top 10% of all scores.
score_requirement = 60  # learn from game runs that at least get a score of 50
initial_games = 50000

# Let's start off by defining some random game runs

print('random_games')
def random_games_for_testing(isThereAModel=0, numberOfEpisodes=5, render=0):
    scores = []
    choices = []
    for episode in range(numberOfEpisodes):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for t in range(goal_steps):
            if render == 1:
                env.render()  # to see what is going on in game, remove for faster code
            if len(prev_obs) > 0 and isThereAModel == 1:
                action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
            else:
                action = env.action_space.sample()  # this generates random actions in any environment,
                                                    # it's sometimes a good place to start

            # A lot of times what can happen is that your neural network will converge towards
            # once thing. In order to understand what is going on, we are going to save all
            # the choices taken by the agent in case we want understand what the ratio our
            # neural network is predicting at. This could help with debugging.

            choices.append(action)

            # Let's save the information we get from each step in the game.

            # Observation can be, in many games, the pixel data, but
            # in CartPole-v0 it is pull position, cart position, etc.

            # Reward will give us a 1 or 0 based on whether the CartPole
            # was balanced (1) or not (0).

            # Done tells you whether the game run is over or not.

            # Info contains any other info provided. You can usually use
            # this during debugging or fine-tuning.

            observation, reward, done, info = env.step(action)
            prev_obs = observation

            # game_memory is important if we want to retrain our neural network. Our model
            # will get stronger as we retrain.
            game_memory.append([observation, action])
            score += reward

            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break

        scores.append(score)

    print('Average Score', sum(scores) / len(scores))
    print('Choice 1: {}, Choice 0: {}'.format(choices.count(1) / len(choices),
                                              choices.count(0) / len(choices)))

# random_games_for_testing()
print('init_pop')

def initial_population():
    # Training data contains observation and move made, the moves will all be random.
    # We will only append training data to training_data if score is higher than
    # score_requirement (50).
    training_data = []
    scores = []
    accepted_scores = []
    env.reset()
    for _ in range(initial_games):
        score = 0
        # We will store all the movements and such in game_memory because we will
        # only know at the end of the game run whether we beat the score requirement.
        # Then, we can place the accepted runs in training_data and accepted_scores.
        game_memory = []
        prev_observation = []

        # The following loop runs 1 game of CartPool-v0
        for _ in range(goal_steps):
            action = env.action_space.sample()  # replace with env.action_space.sample()

            observation, reward, done, info = env.step(action)

            # Here we will basically connect the action taken with the previous observation
            # instead of the current observation (like in the following commented line).

            # game_memory.append([observation, action])

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward  # score only increases by 1 if CartPole is balanced
            if done:
                break

        # We can now save the score in training_data and accepted_scores that passed
        # the score_requirement threshold for 1 particular game run.

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))
    print(len(training_data))
    return training_data

initial_population()

# The following is the neural network using a tflearn implementation.

# In TensorFlow, we can save models and use them later. To use a saved model, we actually
# need to have a model already defined. We also need to make sure that our saved model
# and the model we've defined are the same input size. Therefore, the neural_network_model
# function has an input that describes the size of our models.
# Generally, we want to seperate out the model, the training of the model and usage of the model.

print('NN')
def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    # fully_connected(Incoming, number of units, activation function)
    # dropout(Incoming, keep_prob : A float representing the probability that each element is kept.)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    # output layer
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

print('train_NN')
def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = np.array([i[1] for i in training_data])

    # if we don't already have a model
    if not model:
        model = neural_network_model(input_size=len(X[0]))

    # we choose 5 epochs for the fit since too many epochs can lead to overfitting.
    # I will further comment if I decide to change the value based on results.
    model.fit({'input':X}, {'targets':y}, n_epoch=2, snapshot_step=500, show_metric=True,
              run_id='CartPole-v0-tflearn')

    return model

print('training_data save')
training_data = initial_population()
print('model')
model = train_model(training_data)

print('final run')
# env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
random_games_for_testing(isThereAModel=1, numberOfEpisodes=10, render=0)

model.save('test4.tflearn')

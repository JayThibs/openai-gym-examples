import gym
from gym import wrappers
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
env = gym.make('CartPole-v1')
LR = 1e-3  # learning rate
goal_steps = 500

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

model = neural_network_model(4)
model.load('test3.tflearn')
env = wrappers.Monitor(env, '/tmp/cartpole-v1-experiment-1', force=True)
random_games_for_testing(isThereAModel=1, numberOfEpisodes=100, render=1)
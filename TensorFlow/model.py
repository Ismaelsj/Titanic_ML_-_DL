import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def random_batches(X, Y, batch_size):
    # Concat X & Y inputs to shuffle
    m, _ = X.shape
    big_batch = []
    data = pd.concat([X, Y], axis=1)
    data = data.sample(frac=1)
    # Divide data into batch_size batches
    nb = 0
    for i in range(m):
        if i % batch_size == 0 and i != 0:
            batch_X = data[nb : i].drop(['Survived'], axis=1).values
            batch_Y = data[nb : i].Survived.values.reshape(-1, 1)
            nb = i;
            batches = (batch_X, batch_Y)
            big_batch.append(batches)
    # Handling the end case: last batch < batch_size
    if nb < m:
        batch_X = data[nb : i].drop(['Survived'], axis=1).values
        batch_Y = data[nb : i].Survived.values.reshape(-1, 1)
        batches = (batch_X, batch_Y)
        big_batch.append(batches)
    return big_batch


def make_model(parameters):
    # Network parmaeters
    n_features = parameters['n_features']
    n_hidden = parameters['n_hidden'] # 2
    hidden_dim = parameters['hidden_dim'] # 10
    n_class = parameters['n_class'] # 1
    learning_rate = parameters['learning_rate']

    X = tf.placeholder(tf.float32, shape=[None, n_features], name="x_input")
    Y = tf.placeholder(tf.float32, shape=[None, 1], name="y_input")

    # Layer; Weights & Biases
    Thetas = {
        'Theta1': tf.Variable(tf.random_uniform([n_features,hidden_dim], -1, 1)),
        'Theta2': tf.Variable(tf.random_uniform([hidden_dim,hidden_dim], -1, 1)),
        'Theta3': tf.Variable(tf.random_uniform([hidden_dim,n_class], -1, 1))
        }
    Biases = {
        'Bias1': tf.Variable(tf.zeros([1])),
        'Bias2': tf.Variable(tf.zeros([1])),
        'Bias3': tf.Variable(tf.zeros([1]))
        }

    # Model
    layer2 = tf.sigmoid(tf.add(tf.matmul(X, Thetas['Theta1']), Biases['Bias1']))
    layer3 = tf.sigmoid(tf.add(tf.matmul(layer2, Thetas['Theta2']), Biases['Bias2']))
    hypothesis = tf.sigmoid(tf.add(tf.matmul(layer3, Thetas['Theta3']), Biases['Bias3']))

    # Cost & Trainer
    cost = tf.reduce_mean(((Y * tf.log(hypothesis)) + ((1 - Y) * tf.log(1 - hypothesis))) * -1)
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Accuracy
    prediction = tf.round(hypothesis)
    correct_prediction = tf.equal(prediction, Y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    model = {'X': X, 'Y': Y, 'hypothesis': hypothesis, 'cost': cost, 'train_op': train_op, 'prediction': prediction, 'accuracy': accuracy}
    return model

def neural_network(x_train, y_train, parameters, model):
    # Parameters
    training_epochs = parameters['training_epochs']
    batch_size = parameters['batch_size']
    n_input = parameters['n_input']

    # Cost per epoch saver
    epoch_list = []
    cost_list = []

    # Save model 
    saver = tf.train.Saver()

    # Init variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            epoch_cost = 0.
            total_batches = int(n_input / batch_size)
            batches = random_batches(x_train, y_train, batch_size)
            # Training over all batches
            for batch in batches:
                (batch_x, batch_y) = batch
                # Backprpagation & Cost
                _, c = sess.run([model['train_op'], model['cost']], feed_dict={model['X']: batch_x, model['Y']: batch_y})
                # Compute average loss & save in list
                epoch_cost += c / total_batches
            if (epoch % 10 == 0):
                epoch_list.append(epoch)
                cost_list.append(epoch_cost)
            if (epoch % 100 == 0):
                print ("Cost after epoch {0}: {1}".format(epoch, epoch_cost)) 
                #print("Training output:\n{}\n".format(sess.run(model['prediction'], feed_dict={model['X']: batch_x, model['Y']: batch_y})))
                #print("Real output: \n{}".format(batch_y))
        print("Accuracy : {}%".format(sess.run(model['accuracy'], feed_dict={model['X']: batch_x, model['Y']: batch_y}) * 100))
        #save_path = saver.save(sess, 'Titanic')
        #print("Model saved in path: %s" % save_path)
    if parameters['visualize'] == True:
        plt.plot(epoch_list, cost_list)
        plt.show()


import tensorflow as tf
import pandas as pd

def Accuracy(parameters, model, X_train, Y_train, X_test, Y_test):
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, parameters['model_path'])
        print ("Train Accuracy: {}".format(sess.run(model['accuracy'], feed_dict={model['X']: X_train, model['Y']: Y_train}) * 100))
        print ("Test Accuracy: {}\n".format(sess.run(model['accuracy'], feed_dict={model['X']: X_test, model['Y']: Y_test}) * 100))

def Estimation(parameters, model, X_test, Id_test):
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, parameters['model_path'])
        prediction = pd.DataFrame(sess.run(model['prediction'], feed_dict={model['X']: X_test})).astype(int)
        output = pd .DataFrame({'PassengerId': Id_test, 'Survived': prediction[0]})
        output.to_csv("estimation.csv", index=False)
        print("Writting output to estimation.csv\n")

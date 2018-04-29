import tensorflow as tf

def Accuracy(parameters, model, X_train, Y_train, X_test, Y_test):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        saver.restore(sess, parameters['model_name'])
        print ("Train Accuracy: {}".format(sess.run(model['accuracy'], feed_dict={model['X']: X_train, model['Y']: Y_train})))
        print ("Test Accuracy: {}".format(sess.run(model['accuracy'], feed_dict={model['X']: X_test, model['Y']: Y_test})))

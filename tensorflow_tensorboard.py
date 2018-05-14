# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import tensorflow as tf

leanring_rate = 0.001
train_epochs = 20
batch_size=64
logs_path = "tensorboard/mlp_mnist"

n_input = 784
n_hidden1 = 100
n_hidden2 = 100
n_classes = 10

x = tf.placeholder(tf.float32,shape=[None,n_input],name="InputData")
y = tf.placeholder(tf.float32,shape=[None,n_classes],name="LabelData")
weights = {'w1':tf.Variable(tf.random_normal([n_input,n_hidden1]),name="W1"),
           'w2':tf.Variable(tf.random_normal([n_hidden1,n_hidden2]),name="W2"),   
           'w3':tf.Variable(tf.random_normal([n_hidden2,n_classes]),name="W3"),   
        }
biases = {'b1':tf.Variable(tf.random_normal([n_hidden1]),name="b1"),
          'b2':tf.Variable(tf.random_normal([n_hidden2]),name="b2"),   
          'b3':tf.Variable(tf.random_normal([n_classes]),name="b3"),   
        }

def inference():
    layer_1 = tf.nn.relu(tf.matmul(x,weights['w1'])+biases['b1'])
    tf.summary.histogram('layer_1',layer_1)  #记录变量直方图
    layer_2 = tf.nn.relu(tf.matmul(layer_1, weights['w2']) + biases['b2'])
    tf.summary.histogram('layer_2', layer_2)   #记录变量直方图
    out_layer = tf.matmul(layer_2, weights['w3']) + biases['b3']
    return out_layer

with tf.name_scope('Inference'):
    logits = inference()
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
with tf.name_scope('Optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate = leanring_rate)
    train_op = optimizer.minimize(loss)
with tf.name_scope('Accuracy'):
    pre_correct = tf.equal(tf.argmax(y, 1), tf.argmax(tf.nn.softmax(logits), 1))
    accuracy = tf.reduce_mean(tf.cast(pre_correct,tf.float32))
    
tf.summary.scalar("Loss",loss)
tf.summary.scalar("Accuracy",accuracy)

init = tf.global_variables_initializer()
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())   #创建写对象
    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(train_epochs):
        for batch in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, loss_, summary = sess.run([train_op, loss, merged_summary_op], feed_dict={x:batch_x, y:batch_y})   #执行记录运算
            summary_writer.add_summary(summary, epoch * total_batch + batch)     #将日志写入文件
        if epoch % 5 == 0:
            loss_, acc = sess.run([loss, accuracy], feed_dict={x:batch_x, y:batch_y})
            print("epoch {},  loss {:.4f}, acc {:.3f}".format(epoch, loss_, acc))

    print("optimizer finished!")

    #计算测试集的准确度
    test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
    print('test accuracy', test_acc)
    #命令行输入tensorboard --logdir==C:/Users/PALM/.spyder-py3/tensorboard/mlp_mnist
    #http://127.0.1.1:6006 本地浏览
    #http://223.3.95.83:6006 访问远程服务器
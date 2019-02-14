import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def genGauss(p,n=1,r=1):
    # Load the dataset
    x = []
    y = []
    for k in range(n):
        x_t, y_t = np.random.multivariate_normal([math.sin(2*k*math.pi/n), math.cos(2*k*math.pi/n)], [[0.0125, 0], [0, 0.0125]], p).T
        x.append(x_t)
        y.append(y_t)

    x=np.array(x).flatten()[:,None]
    y=np.array(y).flatten()[:,None]
    x-=np.mean(x)
    y-=np.mean(y)
    train=np.concatenate((x,y),axis=1)

    return train/(np.max(train)*r)

nb_neurons_h1 = 512
nb_neurons_h2 = 512
nb_neurons_h3 = 512
learning_rate =  5e-5
batch_size = 128

z_dim = 2
x_dim = 2
img_dim = 2


z_input_layer = tf.placeholder(tf.float32, shape = [None, z_dim]) # z dimension of 2
x_input_layer = tf.placeholder(tf.float32, shape = [None, x_dim]) # z dimension of 2

## Build the Generator
# Generator weights and biasis
g_w1 = tf.Variable(xavier_init([z_dim, nb_neurons_h1]), name = 'generator_weights1')
g_b1 = tf.Variable(xavier_init([nb_neurons_h1]), name = 'generator_biases1')
g_w2 = tf.Variable(xavier_init([nb_neurons_h1, nb_neurons_h2]), name = 'generator_weights2')
g_b2 = tf.Variable(xavier_init([nb_neurons_h2]), name = 'generator_biases2')
g_w3 = tf.Variable(xavier_init([nb_neurons_h2, nb_neurons_h3]), name = 'generator_weights3')
g_b3 = tf.Variable(xavier_init([nb_neurons_h3]), name = 'biases')
g_w4 = tf.Variable(xavier_init([nb_neurons_h3, x_dim]), name = 'generator_weights4')
g_b4 = tf.Variable(xavier_init([x_dim]), name = 'generator_biases4')


def Generator(z_input):
    g_y1 = tf.nn.relu((tf.matmul(z_input, g_w1) + g_b1), name = 'generator_activation_layer1')
        
    # Weihgts and biases for the second layer:  
    g_y2 = tf.nn.relu((tf.matmul(g_y1, g_w2) + g_b2) , name = 'generator_activation_layer2')
    
    # Weihgts and biases for the third layer:
    g_y3 = tf.nn.relu((tf.matmul(g_y2, g_w3) + g_b3), name = 'generator_activation_layer3')
    
    # Generator output layer
    g_y4 = tf.matmul(g_y3, g_w4) + g_b4
    return g_y4

theta_g = [g_w1, g_w2, g_w3, g_w4, g_b1, g_b2, g_b3, g_b4]

#### Build the critic
# critic Variables
d_w1 = tf.Variable(xavier_init([x_dim, nb_neurons_h1]), name = 'critic_weights1')
d_b1 = tf.Variable(xavier_init([nb_neurons_h1]), name = 'critic_biases1')
d_w2 = tf.Variable(xavier_init([nb_neurons_h1, nb_neurons_h2]), name = 'critic_weights2')
d_b2 = tf.Variable(xavier_init([nb_neurons_h2]), name = 'critic_biases2')
d_w3 = tf.Variable(xavier_init([nb_neurons_h2, nb_neurons_h3]), name = 'critic_weights3')
d_b3 = tf.Variable(xavier_init([nb_neurons_h3]), name = 'critic_biases3')
d_w4 = tf.Variable(xavier_init([nb_neurons_h3, 1]), name = 'critic_weights4')
d_b4 = tf.Variable(xavier_init([1]), name = 'critic_biases4')
theta_d = [d_w1, d_w2, d_w3, d_w4, d_b1, d_b2, d_b3, d_b4]


def Critic(x_input):

    # first layer
    d_y1 = tf.nn.relu((tf.matmul(x_input, d_w1) + d_b1), name = 'critic_activation_layer1')
    # second layer
    d_y2 = tf.nn.relu((tf.matmul(d_y1, d_w2) + d_b2) , name = 'critic_activation_layer2')
    # third layer
    d_y3 = tf.nn.relu((tf.matmul(d_y2, d_w3) + d_b3), name = 'critic_activation_layer3')
    # critic output layer
    d_y4 = tf.matmul(d_y3, d_w4) + d_b4
    
    return d_y4



fake_img = Generator(z_input_layer)
d_real = Critic(x_input_layer)
d_fake = Critic(fake_img)

clip_weights = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_d]



g_loss = tf.reduce_mean(d_fake)
d_loss = tf.reduce_mean(d_real - d_fake)
# alternatice d_loss
# d_loss = tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)



d_optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(d_loss, var_list = theta_d)
g_optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(g_loss, var_list = theta_g)


sess = tf.Session()
sess.run(tf.initialize_all_variables())

epochs = 10000

X_train = genGauss(100, 5, 1)
np.random.shuffle(X_train)
epoch_critic_loss = []
epoch_generator_loss = []

fig, axarr = plt.subplots(1, 2, figsize=(12,4 ))
nb_batches = int(X_train.shape[0] / batch_size)
for epoch in range(10000):
    print('Epoch :', epoch)
    batch_critic_loss = []
    batch_generator_loss = []
    print('nb_batches: ', nb_batches)
    i = 0
    while i < nb_batches:
        d_iters = 5
        critic_loss = []
        d_iter = 0
        while i < nb_batches and d_iter < d_iters:
            image_batch = X_train[i * batch_size:(i + 1) * batch_size]
            _, d_loss_curr = sess.run([d_optimizer, d_loss], feed_dict = {x_input_layer: image_batch, z_input_layer: np.random.uniform(-1, 1, (batch_size, z_dim))})
            critic_loss.append(d_loss_curr)
            sess.run(clip_weights) # weights clipping
            d_iter += 1
        batch_critic_loss.append(sum(critic_loss)/len(critic_loss))        
        _, g_loss_curr = sess.run([g_optimizer, g_loss], feed_dict = {z_input_layer: np.random.uniform(-1, 1, (batch_size, z_dim))})
        batch_generator_loss.append(g_loss_curr)
        i +=1
    epoch_generator_loss.append(np.mean(batch_generator_loss))
    epoch_critic_loss.append(np.mean(batch_critic_loss))
    print('critic loss: ', np.mean(batch_generator_loss))
    print('generator loss: ', np.mean(batch_critic_loss))

    samples = sess.run(fake_img, feed_dict = {z_input_layer: np.random.uniform(-1, 1, (500, z_dim))})
    # plt.title('Wasserstein GAN alg: - Epoch: {}'.format(epoch))
    fig.suptitle('Wasserstein GAN alg: - Epoch: {}'.format(epoch))
    axarr[0].set_title('Real Data vs. Generated Data')
    axarr[0].scatter(X_train[:, 0], X_train[:, 1], c = 'red', label = 'Real data', marker = '.')
    axarr[0].scatter(samples[:, 0], samples[:, 1], c = 'green', label = 'Fake data', marker = '.')
    axarr[0].legend(loc='upper left')
    axarr[1].set_title('Generator & Critic error functions')
    axarr[1].plot(epoch_critic_loss, color='red', label = 'Critic loss')
    axarr[1].plot(epoch_generator_loss, color='blue', label = 'Generator loss')
    axarr[1].legend(loc='upper left')
    fig.savefig('tf_wgan_results/frame.jpg')
    fig.savefig('tf_wgan_results/frame' + str(epoch) + '.jpg')
    axarr[0].clear()
    axarr[1].clear()





































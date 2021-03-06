# wgan-gaussian
An implementation for Wasserstein Generative Adversarial Network to generate different 5 gaussian distributions.
![wgan_frames](https://user-images.githubusercontent.com/26183913/52799603-7646b700-307a-11e9-8ff6-9789668ce8b0.gif)

Unlike Vanilla GAN, Wasserstein GAN is able to learn all distributions in the training data.
The loss function is Wasserstein distance (Earth movement distance) for minimizing the distance between the generated data and the real data.

# Content:
- WGAN Architecture
- Generator Architecture
- Critic Architecture
- Wasserstein distance vs JS and KL divergences
- Wassersteain distance as GAN loss function
- How to train the model


# Model Overview
Similar to gan we have two neural networks: generative model and discriminative model, but we called the discriminative model Critic instead of Discriminator because we use another error function.
![image](https://user-images.githubusercontent.com/26183913/52799910-0dac0a00-307b-11e9-995b-dce1ceb4fd72.png)
# Generator Architecture
It consists of an input layer of 2 neurons for the z vector, 3 hidden layers of 512 neurons and an output layer of 2 neurons
activation functions of the 3 hidden layers are Relus and linear for the output layer
<p><img width="460" alt="image" src="https://user-images.githubusercontent.com/26183913/52800140-74c9be80-307b-11e9-8222-692e31989c8e.png"></p>

# Critic Architecture
it consists of an input layer of 2 neurons for the training data, 3 hidden layers of 512 neurons of Relu activation function and an output layer of 1 neuron of linear activation function
<p><img width="460" alt="image" src="https://user-images.githubusercontent.com/26183913/52800200-9034c980-307b-11e9-9f16-6461a8266432.png"></p>

# Wasserstein distance vs Jensen–Shannon divergence & Kullback–Leibler divergence
The Wasserstein Distance is a well-known distance metric for probability distributions. It is sometimes called EarthMover’s Distance and is studied in the field of optimal transportation. It measures the optimal cost of transporting one distribution to another (Solomon et al., 2014). Actually, even when two distributions are located in a lower dimensional manifolds without overlaps, the Wasserstein distance can still provide a meaningful and smooth representation of the distance in-between. Meanwhile, other distance functions suffer from issues related to continuity. For example, the Kullback-Leibler divergence is infinity for two fully disjoint distributions. Another example is the Jensen-Shannon, which is not differentiable for fully overlapped cases i.e.: it has a sudden jump at zero distance. Thus, only the Wasserstein distance provides a smooth measure, which makes it really helpful for stable learning. Therefore, it is predestinated to solve the stability issue, which appears in normal GANs.

# Wasserstein as GAN loss function
<p><img width = "350" src="https://user-images.githubusercontent.com/26183913/52800359-e4d84480-307b-11e9-98e4-6b430dda39b9.png"></p>

```
d_loss = tf.reduce_mean(d_real - d_fake)
g_loss = tf.reduce_mean(d_fake)
```
We clipp the Critic weights to enforce a Lipschitz constraint.
<p><img width = "390" src="https://user-images.githubusercontent.com/26183913/52800447-0f2a0200-307c-11e9-9d62-c914c8d95d94.png"></p>

```
clip_weights = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_d]
```
We use RMSprop optimizer instead of Adam because Adam could cause instability in the training process.

# How to train the model
Write in the console `python wgan.py` to train the model for generating 5 Gaussian distributions. The results will be saved for each epoch in the `tf_wgan_results` folder

# License

MIT License

Copyright (c) 2019

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
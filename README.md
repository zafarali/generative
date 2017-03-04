# generative

Repository to explore Generative Models and in particular image completion methods.


## Variational Autodencoders

References: 

0. [Image Completion with Deep Learning in TensorFlow](https://bamos.github.io/2016/08/09/deep-completion/)
1. [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
2. [Tutorial On Variational Autoencoders](https://arxiv.org/abs/1606.05908)
3. [What is a Variational Autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)
4. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
5. [Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models)

### Variational Autoencoders (VAE)

Found in `./vae.keras.py`. After about 15 epochs the latent encodings looks like this: (apologies for the lack of a legend.)

![image](https://cloud.githubusercontent.com/assets/6295292/23576231/9893d016-006d-11e7-8570-f5143a5370b6.png)


and we can visualize the latent space manifold:

![image](https://cloud.githubusercontent.com/assets/6295292/23576235/a463153c-006d-11e7-975d-97a9ae9a60d0.png)


### Conditional Variational Autoencoders (CVAE)

Found in `./cvae.keras.py`. After about two epochs I am able to generate samples:

![image](https://cloud.githubusercontent.com/assets/6295292/23576118/300547b2-006a-11e7-918a-7522748b5397.png)

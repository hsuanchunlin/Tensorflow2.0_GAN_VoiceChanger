# Tensorflow2.0_GAN_VoiceChanger
This is my small project for fun for deep voice changer. The original codes is from [Lei Mao's work in TensorFlow 1.8](https://github.com/leimao/Voice-Converter-CycleGAN) \
To construct CycleGAN, I have referenced the [keras tutorial](https://keras.io/examples/generative/cyclegan/).\
The codes are upgraded to tensorflow 2.0 and modified to adapt Mac-optimized TensorFlow 2.0 to optimize training on my 15' Macbook Pro 2019.\

For Mac-optimized TensorFlow, use
```python
import tensorflow.compat.v2 as tf
```
instead of
```python
import tensorflow as tf
```
I did not turn the eager execution off according to the error when use model.fit .\
[Mac-optimized TensorFlow](https://github.com/apple/tensorflow_macos)

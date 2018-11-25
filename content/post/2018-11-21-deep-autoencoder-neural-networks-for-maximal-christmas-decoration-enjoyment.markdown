---
title: Deep Autoencoder Neural Networks for Maximal Christmas Decoration Enjoyment
author: Michael C Johnson
date: '2018-11-21'
slug: deep-autoencoder-neural-networks-for-maximal-christmas-decoration-enjoyment
categories: []
tags:
  - Deep Learning
  - Raspberry Pi
  - python
  - Keras
  - tensorflow
type: ''
subtitle: ''
image: ''
---

My wife loves decorations. From Valentines Day to Easter to Thanksgiving, our house is adorned with interesting festive items. Her favorite season by far is Christmas. Every December we drive around trying to find the best christmas lights. Inevitably, their are a few houses with lights that blink with the music, something I've always been fascinating with.   
Two years ago I embarked on a mission to build my own Christmas Tree Light Show. Roughly this could be acomplished by figuring out what frequencies are playing at a given time, and blinking different lights for a given frequency. Specically this is what I had in mind:
1. Break music up into even sized chunks
2. Convert each chunk into frequency components using [FFT](https://en.wikipedia.org/wiki/Discrete_Fourier_transform)  
3. Bucketize frequencies into the number of light channels
4. Activate a light channel when the decibles reached a certain threshold:   

![Light Show Pipeline](/img/LightShowPipeline.jpg)

A brief Google investigation revealed I had been (thankfully) beaten to the chase. The open source [lightshowpi](http://lightshowpi.org/) project already had a full pipeline, took like 10 minutes to set up, and already had the FFT parralellized on the Raspberry Pi GPU (so you can apply it to streaming music!). Here is a video of our lightshow the first year (christmas tree only):

<!--html_preserve-->{{% youtube "aAiQ7VSyrno" %}}<!--/html_preserve-->

What I've found over the past few years is that while the light show is certainly entertaining (After every song, every year, my kids have a deep expression of joy and clap and congratulate me on my show) I've been interested in ways to capture more of the structure of the sound in the blinking lights. The light blinking can be sporadic (especially during very fast parts of the song), and vocals are not always captured very well accross octave range's.
## Enter: Deep Autoencoder Neural Networks 
Autoencoders are deeply intuitive networks that have a simple premise: reconstruct an input from an output, and in the mean time learn a compressed representation of the data. This is accomplished by squezing the network in the middle, forcing the network to compress x inputs into y intermediate outputs, where x>>y. [Here](https://blog.keras.io/building-autoencoders-in-keras.html) is a nice blog post from the Keras blog that goes into some detail on the mnist dataset.

![Generic AutoEncoder](/img/Autoencoder.jpg)

So what exactly are we going to squeeze? Once you've processed the signal through the FFT, you can make a spectrogram (check out [this](https://www.youtube.com/watch?v=_FatxGN3vAM) super psyched youtube video)... Here time is on the x-axis, frequency is on the y, and the color represents the amplitude for that time/frequency. I'd really watch the video mentioned above, but if you could sing a pure [A4 note](http://pages.mtu.edu/~suits/notefreqs.html) you'd have a solid (in this case red) line at 440Hz. The song we'll study for the rest of the post is "[A Mad Russian's Christmas](https://www.youtube.com/watch?v=6P9xxJ4V7no)" by The Trans Siberian Orchestra, check out the spectrogram:  
![A Mad Russian's Christmas](/img/madRussianOrig.jpg)

You can tell quite a bit about what is going on in the song. The intro starts slow with several repeating sections, quieting down and building up to the fastest section (around 70s). This builds up, faster and louder until it breaks into another rythmic section. From here you can see a few more fast/repeating sections, and a general quieting at the end. I encourage you to look at the spectrogram while you listen to the song, see if you can pick out the different sections. But don't go crazy, there is plenty more fun to be had below!  

Since the ultimate goal is to light up different christmas lights at different rythms in the song, what we need to do is to learn a compressed representation of the audio input *at each time step*. The spectrogram works perfectly for this. There are 513 frequency 'features' at each timestep, so the goal of our autoencoder will be to build an 8-channel representation of each time step (The same number of light channels we have. Note that this is essentially a dimensionality reduction technique, and the standard set of algorithms could be applied here (PCA and Tsne are my favorite, perhaps we'll do another post on those). Something like this:

![Autoencoder For Spectrogram](/img/AutoencoderSpectrogram.jpg)

We'll train "A Mad Russian's Christmas" and test on "Christmas Serajevo". First, lets get some setup out of the way:



```python

from spectrogramUtils import * #(Check Bitbucket link for this file. Or use pyplot.specgram....)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from keras.layers import Input, Dense, Conv1D, MaxPool1D, UpSampling1D, Flatten, Dropout
```

```
## C:\Users\mj514\ANACON~1\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
##   from ._conv import register_converters as _register_converters
## Using TensorFlow backend.
```

```python
from keras.models import Model

from keract import get_activations
```
Keract is a nice package that allows you to get the intermediate activations from any Keras-based network. Notice I import both Dense and 1D Convolution layers here, we'll use the latter down below.
Now, create extract the spectrograms out of the the data. [This notebook](https://bitbucket.org/mj514316/christmasautoencoder/src/master/MakeSpectrogram.ipynb) shows how to convert from .mp3 to .wav, using ffmpeg. I'm not as familiar with powershell as I'd like to be, so I just used python to loop through the files. The plotstft function (from spectrogramUtils) breaks our song up into frequency components, and returns a .npy array with 513 features and some number of time step features.


```python
madIms = plotstft('C:/Users/mj514/Documents/dataSci/christmasAutoencoder/maddrussian.wav')
```

![](2018-11-21-deep-autoencoder-neural-networks-for-maximal-christmas-decoration-enjoyment_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

```python
christmasSerajevoIms = plotstft('C:/Users/mj514/Documents/dataSci/christmasAutoencoder/christmasSerajevo.wav')
```

![](2018-11-21-deep-autoencoder-neural-networks-for-maximal-christmas-decoration-enjoyment_files/figure-html/unnamed-chunk-4-2.png)<!-- -->

```python
print(f"Class: {madIms.__class__}, shape: {madIms.shape}") # I recently fell in love with fstrings!!!
```

```
## Class: <class 'numpy.ndarray'>, shape: (24330, 513)
```

Need to scale the inputs. I use the train to set the scale, then apply the scale to the test. This isn't strictly necessary (could set min max based on all of the songs in my library) but I wanted to see how well it could support online use... in other words you don't know what the rest of the song is going to look like. Also, since our final softmax layer will only output between 0 and 1, that is good range to scale to.


```python
scaler = MinMaxScaler()
x_train = scaler.fit_transform(madIms) 
x_test = scaler.transform(christmasSerajevoIms)
```
Next, let's set up our autoencoder. Notice thise one is pretty simple, I found that the final softmax stage was enough to capture most of the complexity. You can get slightly better if you do the stepdown and up(commented out), but considering this will run on a raspberry pi and it doesn't get much better, it's reasonable to skip.


```python
def denseAutoencoder(internalActivation = 'tanh'):
    input_img = Input(shape=(513,))
    input_dropped = Dropout(0.2)(input_img)
    # encoded = Dense(128, activation=internalActivation, name = 'encode_128')(input_img)
    # encoded = Dense(64, activation=internalActivation, name = 'encode_64')(encoded)
    # encoded = Dense(32, activation=internalActivation, name = 'encode_32')(encoded)
    # encoded = Dense(16, activation=internalActivation, name = 'encode_16')(encoded)
    encoded = Dense(8, activation=internalActivation, name = 'encode_8')(input_dropped)
    # # encoded = Dense(4, activation='relu')(encoded)
    # encoded = Dense(2, activation='relu')(encoded)
    # encoded = Dense(1, activation='relu')(encoded)
    #decoded = Dense(16, activation=internalActivation, name = 'decode_16')(encoded)
    # decoded = Dense(32, activation=internalActivation, name = 'decode_32')(decoded)
    # decoded = Dense(64, activation=internalActivation, name = 'decode_64')(decoded)
    # decoded = Dense(128, activation=internalActivation, name = 'decode_128')(decoded)
    decoded = Dense(513, activation='sigmoid', name = 'output')(encoded)

    autoencoderDense = Model(input_img, decoded)
    autoencoderDense.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoderDense
    
autoencoderDense = denseAutoencoder()
autoencoderDense.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True, #important
                validation_data=(x_test, x_test))
```

Couple of things to note, I played with all of the internal activations and tanh was by far the best. I suspect tanh was better than relu in this case because of the inherent non linear nature of sound. Just goes to show that you should craft your architecture to fit the problem at hand. Also, the network is very sensative to overfitting. This could be partially resolved by just adding more songs, and for the production version of this I'll train on every song in my library. Imagine the different structure that would need to be captured for songs that are mainly vocals vs these which are highly instrumental.  
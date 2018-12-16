---
title: 'Autoencoder For Christmas 2: Tuning and Convolutions'
author: Michael Johnson
date: '2018-12-02'
slug: autoencoder-for-christmas-2-tuning-and-convolutions
categories: []
tags:
  - Autoencoder
  - Deep Learning
  - Raspberry Pi
  - python
  - Keras
  - tensorflow
type: ''
subtitle: ''
image: '/img/LightShowPipelineDeep.jpg'
---
I've found that while typically people are exposed to machine learning on clean datasets with with clear objectives, doing machine learning in real busness environments isn't often that straightforward. You have to think more creatively about what you use for a training set and how you go about training your model. This is most prevelant in Natural Language Processing, where there isn't typically clean labeled dataset for what you want (IMDB sentiment doesn't work very well for the defense industry...).
  
The model we built in the previous blog post is an example of this: We are using a proxy objective (minimize reconstruction error of an autoencoder) to achieve our target objective (blink 8 channels in some pleasing fashion). Because of this, it is important to do some sanity checking of the result to see if it did as we intended.
## The Pipeline
Our origonal pipeline for light show magic was the following: 

![Origonal Light Show Pipeline](/img/LightShowPipeline.jpg)

Here is the pipeline we'd like to create:

![Origonal Light Show Pipeline](/img/LightShowPipelineDeep.jpg)

The objective is to capture a rich representation of sound with only 8 channels so we can entertain our neighbors with maximal efficiency. Here is the result we got last time:
{{< gallery >}}
{{< figure link="/img/bigAutoEncoded.jpg" caption="Large Autoencoded" >}}
{{< /gallery >}}
## What's Wrong?
While it's interesting and certainly begins to capture some of the structure of the music, there are a couple problems with the reproduction. If you listen closely to the first part of [the song](https://www.youtube.com/watch?v=MHioIlbnS_A), there is a clear repeating pattern of the violin drawing up and down. Take a look at our first reproduction of the intro:

![Origonal Light Show Pipeline](/img/encodedIntro.jpg)

For the most part, this looks like noise. Some of the channels change levels on the time span we are studying, but the structure of the song doesn't seem well captured = __unhappy neighbors__. There are lots of potential reasons for why this might be. Lets take a look at a plot of the magnitude of each channel over time. Turns out we can hack the function we used before to make a nice representation of this:

```python
from keras import Model
from keract import get_activations
import matplotlib.pylpot as plt
activeDense = Model.load('denseAutoencoderRound1.hdf5')
activeDense = get_activations(autoencoderDense, x_test)

intermediateLayer = list(activeDense.keys())[5] #grab the layer with 8

plt.plot(activeDense[intermediateLayer][0:1500], linewidth=0.54)
```
![8 Channel Bad Ending](/img/8ChannelBadEnding.png)
This shows the magnitude of the 8 intermediate channels of our network, and how they change over time. You can imagine the channel being __on__ when the magnitude is above zero (yellow,orange,red) and __off__ when the magnitude is below zero (blue). From this plot, one of the training problems is clear: two of the channels seem to be dedicated to minimizing reproduction error at the very end. If you listen to the last 10 seconds of the song, you'll notice that it's very quiet and contains almost nothing of value. This is a waste of our reproduction.
## How do we fix it?
Because this end doesn't do much for our light show (it's fine to have the lights dark or doing whatever they want for the last 10 seconds), it makes sense to try to remove it from the training round. There are a few ways that we could approach the problem:

1. Guess how long the average end section is, and remove that length from every song we train on. (Could remove interesting endings from some songs)
2. Set a threshold for minimum noise levels. (Could remove bits from the intro and quite parts in the middle of the song, or songs that have many quite sections)
3. Some interesting combination of the above (remove bits that fail a threshold in the last 20 seconds of the song)
4. __Train a sweet machine learning model__.

While in this case a heuristic/rules based approach might be more effective, lets see if we can train __classification__ model to decide whether a given segment is in the end or not. 

## A Breif Introduction to Classification
In machine learning, classification is the process of mapping an input to a categorical output. This could be mapping the pixes of an image to there proper classification fo cat or dog, or the words in a product review to posotive or negative.  
{{< gallery >}}
{{< figure link="/img/puppy.jpg" caption="Puppy" >}}
{{< figure link="/img/kitten.png" caption="Kitten" >}}
{{< /gallery >}}

In this case, we need to map the frequency representation of our song (the 513 frequency buckets) to the labels isEnd or isNotEnd, ideally to build something that can recognize the end of any song that we give it. First, lets use our training set to pick the end of the our training song (A Mad Russian's Christmas):

```python
plt.plot(activeTrain[intermediateLayer][23600:], linewidth=0.5)
plt.axvline(x = 335)
```
![Madd Russian](/img/madRussionEnd.png)

Now build our objective variable. We'll build an array of ones that goes to just before it ends, and an array of zeros that fills in the rest:


```python
regularSong = np.ones(23935)
endOfSong = np.zeros(len(x_train)-23935)
y_train = np.append(regularSong,endOfSong)
print(f"Just before end: {y_train[23934]}, just after: {y_train[23935]}, length: {len(full)}, length orig: {len(x_train)}")
## Just before end: 1.0, just after: 0.0, length: 24330, length orig: 24330
```
Do the same for the test set (Christmas Sarajevo):

```python
y_test  = np.append(np.ones(16505),np.zeros(len(x_test)-16505))
```
Now lets try a simple [logistic regression](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc), which uses a _linear_ mapping between the inputs (frequency buckets) and outputs (the class). This linear mapping is pushed through a sigmoid function which restricts the outcome to between 0 and 1, and allows us to interpret the results as a probability. Thus, it approxomates the function _p(endOfSong|frquencyBuckets)_ "What is the probability this is the end of the song given the frequency distribution":


```python
from sklearn.linear_model import LogisticRegression
endClassifier = LogisticRegression(random_state=42).fit(x_train, y_train)
```
Fit on the training set, check accuracy on the test set:

```python
testAccuracy = np.mean(y_hat==y_test)
print(f"test accuracy: {testAccuracy.round(3)}")
## test accuracy: 0.944
```
94% accuracy! This looks good, but lets see what accuracy we would get if we just guessed 1 every time (assume that the whole song is what we want, and there is no end [^2] )

```python
naivey_hat = np.ones(len(y_test)) # always guess ones
naiveAccuracy = np.mean(naivey_hat==y_test)
print(f"Niave accuracy: {testAccuracy.round(3)}")
## test accuracy: 0.933
```
So if we randomly guess, we get 93% accuracy, and our model gets 94% accuracy. Not a big improvement
## Can we improve on the model? 
Lets add some simple features to the model to give it some more hints on what might make the end. At this point, the model is taking the magnitude of the frequency buckets in and trying to predict weather or not we are at the end. Intuitively, the end is probably near the __actual__ end of the song, so perhaps we can let the model know when we are getting close. 

[^2]: In this case we could have simply taken the mean of y_test, but explicitly guessing 1's is a tiny bit more clear


```python
percentThroughSong_train = np.expand_dims(np.arange(len(x_train))/len(x_train),1) #Expand dims simply makes it a 2d Array so we can append it to our train/test set
percentThroughSong_test =np.expand_dims(np.arange(len(x_test))/len(x_test),1)
```
Retraining gives us a better result:

```python
x_train_improved = np.concatenate((x_train,percentThroughSong_train), axis = 1)
improvedEndClassifier = LogisticRegression(random_state=42).fit(x_train_improved, axis = 1), y_train)
```
Because logistic regression represents a linear mapping between input and output, __and__ because we normalized our inputs to a fixed range, we can use the coefficients magnitude and sign to judge what our algorithm picked as the most important variable:

```python
plt.plot(improvedEndClassifier.coef_.T)
print(np.argmin(improvedEndClassifier.coef_.T))
##513
```
![Log Reg Coefs](/img/logisticRegressionCoeffs.png)

This plot reveals a couple of interesting things about how the model arrived at it's conclusion. First off, by far the highest magnitude comes at the very end (feature 513). Because our features are 0 indexed, this is actually the 514th variable, which is the % through song we added manually! So the logistic regression, rightly so, has decided the best way to find the end of the song is to look at how close to the end we are. You can also immediately see that this is not far from some of the rule/heuristic based approaches we had suggested before. Trained on many songs, this would simply pick the average distance from the end of the song and use that as the best guess.  

## Issues With Logistic Regression
There are a couple of other issues with the use of logistic regression. Logistic regression doesn't work well when the variables are have high colliniearty. Lets check the correlation plot for our 513 features and see if this could be makingthings more difficult:

```python
x_trainDF = pd.DataFrame(data=x_train, columns=range(0,x_train.shape[1]))
x_trainDF.shape
corr = x_trainDF.sample(10000).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap,  center=0,
            square=True, linewidths=.0, cbar_kws={"shrink": .5})
```

![ Corrplot](/img/correlationPlot.png)

This plot shows that __every feature between 0 and ~360 is nearly perfectly correllated with every other feature between 0 and ~360__. It also shows that there is a negative correlation between 462-464 and 0-100, presumably indicating that when there are lots of high frequency notes (462-464) there are fewer low frequency notes (0-100).  

Additionally, logilstic regression makes strong assumptions about the world. In this case, it makes the assumption that the relationship between the input frequency buckets and the output (is it the end) is linear. You can fool it by transforming the variables using a non-linear operator (log, exp, ()^2, etc...), but you have to put it explicitly into the function. Finally, logistic regression does not naturally take into account interaction terms. While you can include interaction terms, you get deeper and deeper into trouble with high collinearity. I suspect these are some of the reasons Jeremy Howard reccomends starting with Random Forest as your first machine learning model in his excellent introduction to machine learning course at [fast.ai](https://course.fast.ai/lessonsml1/lesson1.html).

You can check out my [bitbucket repo](https://bitbucket.org/mj514316/christmasautoencoder/src/master) where I applied RF and a simple Feed Forward Neural network, as well as tried to use my autoencoder as a dimensionality reduction technique.

## Is Pulling Out The End Worth It?
Lets take a look at how the model does after we take out the end, and if it makes the outcome any better. First, we'll set up the training set with the end carefully removed (based on the experience above, we should probably just clip off the last 10 seconds of each song):

```python
endOfSong = 23000
x_trainMini = x_train[0:endOfSong]
```
Then train as in the first blog post:

```python
autoencoderDense = denseAutoencoder()
autoencoderDense.fit(x_trainMini, x_trainMini,
                epochs=10,
                batch_size=256,
                shuffle=True, #important
                validation_data=(x_test, x_test))
```

Grabbing the intermediate interactions again, and plotting the intro:

```python
active = get_activations(autoencoderDense, x_trainMini)
intermediateLayer = list(active.keys())[5]
plt.figure(figsize=(200, 7.5))
plt.plot(active[intermediateLayer], linewidth=0.5)
```
![Better Intro](/img/FixedIntro.png)




---
title: ' How Many Boys is Too Many?

'
author: Michael Johnson
date: '2018-12-30'
slug: statistically-verifying-claims-made-in-childrens-stories-web-scraping-nlp-vectorized-simulation
categories:
  - simulation
  - data science
tags:
  - wikipedia
  - python
  - SpaCy
  - NLP
  - pandas
type: ''
subtitle: 'Statistical Evaluation of Having Lots of Boys'
image: 'kids.jpg'
math: true
---
We have many, many children. 
![kids](/img/kids.jpg)
Most days I consider this to be an incredible blessing and they often tend to be the root of statistical curiosity. For example, when we had our third boy in a row[^1] I started to wonder how many children do you have to have of a given gender before you can conclude that there is something other than random chance influencing to the outcome? In other words, if you have 3 boys in a row does that mean you got "lucky" or that you are only capable of having boys (or pre-disposed to having boys)?  
[^1]: If you know our family you'll realize I'm cheating a bit here. Get over it ;-). 

We can cast this problem as a classic coin flip probability, and use a statistical measure (the p value) to make a judgment on if there could be something going on beyond chance. To do this, we'll have to set up a logical framework to evaluate the situation.

# Probability Distributions and The Null Hypothesis
Hypothesis testing is one of the most important and most foundational concepts to Data Science. Almost every problem involving data can be cast into this framework:

1. Develop a *null hypothesis*.
  * There is an equal probability of having boys and girls.
3. Identify the outcome you'd like to test (or have observed).
  * 5 children, all girls
2. Calculate/simulate the probability of the outcome given the null hypothesis.
  * If there is an equal probability of having boys and girls, how likely is it that one family with 5 children will have 5 girls?
3. Compare the answer you got above with a predetermined criteria (the p value) to decide if you can reject the null hypothesis. (typically taken to be 0.05, or 5%)

Once you've formulated your null hypothesis, how do you go about calculating the result? For our simple case above, where we can define binary outcomes (boy/girl, heads/tails, pass/fail, etc...), and we're interested in calculating the probability of an outcome that meets a specified criteria (5 girls out of 5 children), the binomial distribution allows us to calculate this directly. 

The binomial distribution is given by: 

`\(\mathrm{P}(X=k) = \binom{n}{k} p^k(1-p)^{n-k}\)` 

where `\(k\)` is equal to the number of success (in this case the number of girls), `\(n\)` is the number of trials (in this case, the number of children) and `\(p\)` is the probability of boys or girls (our null hypot hesis puts this at 50%). Our case is a simple version of the above,  where `\(k = n\)`, and `\(p = 1-p = 0.5)\)` so the equation above simplifies to:

`\(\mathrm{P}(X=k| k = n, p = 0.5) = \ p^k\)` 

Lets calculate this probability for families with 1-10 children, and see where the cuttoff is:


```r
library(tidyverse)
data_frame(numChildren = 1:10) %>%
  mutate(probabilityAllGirls = 0.5^numChildren) %>%
  ggplot(aes(numChildren,probabilityAllGirls)) +
  geom_point() + 
  geom_text(aes(label = round(probabilityAllGirls,3)*100),nudge_y = .015,nudge_x = -0.2) + 
  geom_hline(yintercept=.05, linetype="dashed", color = "red") + 
  scale_y_continuous(labels = scales::percent) + 
  ggtitle("Probability That All Children Will be Girls by Family Size")
```

<img src="/post/2018-12-30-statistically-verifying-claims-made-in-childrens-stories-web-scraping-nlp-vectorized-simulation_files/figure-html/unnamed-chunk-1-1.png" width="672" />
This chart shows that if the probability of boy and girl are equal, and you sampled 100 families of 5, just over 3 of them would be expected to have all girls (there is another 3 that has all boys). The threshold typically held for rejecting the null hypothesis is 5 out of 100, or 5%. This suggests that if your family with 5 children has only girls, you can statistically conclude that it may not be coming from random chance, there could be something else going on! Or maybe not... (see P Hacking below).

Often the problems I encounter connot be easily cast into an existing distribution, or are more complicated and thus developing the analytical solution is difficult or impossible. In this case we can employ my favorite method for doing hypothesis testing: Simulation.

## Simple Simulation
For simplicity lets say we didn't know (or were too lazy) to calculate the equation above. We could imagine a set of pretend families who had pretend children in a way that aligns to the null hypothesis (50/50 chance of boy/girl), and figure out how often those familes had all girls. 

![100 Families](/img/100Families.gif)

(OK so this is not a pretend family and they are all the same, but I'm sure you can use your imagination...)  
In R we can use sample to draw from 0s or 1s (boy or girl) with equal probability

```r
set.seed(1)
numberOfChildren = 5
probabilityOfGirl = 0.5

oneFamily = sample(c(0,1),numberOfChildren,probabilityOfGirl) #simulate one family
oneFamily
```

```
## [1] 0 0 1 1 0
```
Our first family with 5 children had two boys (the 0's), then two girls(the 1's), then another boy. What we care about is the number of girls, so we can just take the sum of our family:

```r
sum(oneFamily)
```

```
## [1] 2
```
And we have the number of girls for this family.

To get the p value from our plot above, we need to do this proceedure for 10,000 or so pretend families, and then calculate the percentage of those trials that match our observation (sum to a total of 5):

It turns out R gives us a shortcut to this, allowing us to get the number of girls from successive bernoulli trials quite easily:


```r
set.seed(1)
numberOfChildren =5
numberOfFamilies = 1
probabilityOfGirl = 0.5
rbinom(n=numberOfFamilies, size = numberOfChildren, prob = probabilityOfGirl)
```

```
## [1] 2
```
The 2 here represents the same 2 that we got when we did sum(oneFamily)... Our first family has 2 girls. Lets do it now for 10,000 families:

```r
set.seed(1)
numberOfChildren =5
numberOfFamilies = 10000
probabilityOfGirl = 0.5
tenkFamilies = rbinom(n=numberOfFamilies, size = numberOfChildren, prob = probabilityOfGirl)
print(paste0("First Family: ", tenkFamilies[1], " girls, 5010th Family: ", tenkFamilies[5010]," girls"))
```

```
## [1] "First Family: 2 girls, 5010th Family: 4 girls"
```
Lets check for those that have all 5 girls:

```r
mean(tenkFamilies == 5)
```

```
## [1] 0.0328
```
3.2% is close to the number we arrived at above (3.125) but not exactly. There are two reasons for this. **First**, because we are simulating randomness, the outcome of any given experiment (in this case 10 families) will drift around 3.125, but won't be precisely 3.125. You can reduce [the standard error](https://www.youtube.com/watch?v=BwYj69LAQOI) in the estimate by sampling more families (try 100000000). **Second**, even with more samples, there is likely to be some error in our estimate due to the fact that the random number generator on our computer is not 'truly' random. It is a reasonable approxomation of randomness, but it turns out randomness is not so easy to pull out of an algorithm.

## Some Problems With Our Conclusion
While it is nice to have a conclusive line drawn in the sand about statistical significance, there are several problems with this conclusion. First, note that even with random chance, a non trivial number of families out of 100 are likely to have *all girls* or *all boys*. Given the number of familes with children, we must expect at least *some* of them will have children of only one sex. We could consider the entire distribution of families and discover if it deviates from our expectation, but drawing conclusions based on one family seems difficult in this case

It turns out that this issue has importance beyond simple curiosity. A p-value of 0.05 has been used as the standard accross many scientific disciplines as a threshold for declaring a finding statistically siginficant and therefore publishable. Setting the bar to low means that we don't have reproduceable results (in other words, some papers will print results that arent actually true). Setting the bar too high means that important results won't be able to be published. [This](https://www.nature.com/news/1-500-scientists-lift-the-lid-on-reproducibility-1.19970) article shows that scientists failed to reproduce a surprising number of results accross many fields of study including engineering, medicine, and psychology.  

It would appear that the evidence is coming out on the side that the 0.05 threshold is too low, and thus, there has been a [push recently](https://jamanetwork.com/journals/jama/article-abstract/2676503) to lower the threshold for the p value from 0.05 to 0.005, or 0.5%. Partially derivative did a nice podcast on this, check it out [here](http://partiallyderivative.com/podcast/2017/08/08/p-values).  

## P Hacking
To get a sense of why this may be a problem we can do a simple thought experiment. The scientific process typically goes something like this: 

1. Come up with a hypothesis
  * X chemical will produce Y result...
2. Design and execute an experement to test the hypothesis
  * X chemical had Y result with p statistical certainty.
3. If the result was successful, write a fancy paper to tell other people.

The formulation of the p-value is purposfully skeptical. It says, "OK, your trying to prove that this or that happens. What if nothing really is going on, how many times would we magically arrive at the conclusion you just observed?". Lets say we wanted to make the argument that the actual ratio of boys/girls for some populations is not 50/50. To do that we decided to study the distribution of children from 100 famililes. Turns out if you look at enough groups of 100 families, you'd find one that you could use to support your hypothesis:

![Lots of Families](/img/manyFamilies.gif)

In practice, it doesn't 



---
title: Statistically Verifying Claims Made in Childrens Stories
author: Michael Johnson
date: '2019-01-25'
slug: statistically-verifying-claims-made-in-childrens-stories
categories:
  - data science
tags:
  - web scraping
  - spacy
  - nlp
  - python
  - simulation
  - family
type: ''
subtitle: 'Using web scraping, SpaCy, and simulation to check basic logic'
image: ''
---
Bedtime has historically been a battle for our family. Kids have this impressive ability to fall asleep when you want them awake and vigorously stay away when you want them to sleep. To combat the insanity, we read a few books every night before bed. There is a great series of books for young children called "[The Magic Tree House](https://www.magictreehouse.com/)". The premise of the whole series two children (Jack and Annie) who transport to other times and places for some mystery or adventure.  

One night, while reading one of the books, I stumbled upon an interesting passage. The children's friend, Morgan Le Fay, is trapped someplace in time and space and to find her they need to locate four items. They have already found two (a Moonstone and a Mango). Here is the passage that caught my attention:

>"Hey Guess what?" said Jack. "**Moonstone** and **mango** start with the letter **M**. Just like **Morgan**." 
> " You're right," said Annie. "**I bet all four things start with an M**," said Jack.

I immediatly thought, *I wonder if that is a statistically valid proposition?*. The rest of this blog post contains the plan and execution to rigorously validate this as a reasonable hypothesis.
# The Plan
In order to verify the assertion Jack makes, we need to attempt to determine the probability that you would have a friend and then happen upon two items that all start with the same letter *assuming you found these things at random*. For example, if we have 1,000 universes (or 100,000) and Jack and Annie made random friends and found random items, what is the probability that these random friends and random items would all of these would start with the same letter? This is our null hypothesis, discussed in detail in my [previous post](https://minimizeuncertainty.com/post/how-many-is-too-many/).

What do I mean by random? Intuitively, these universes should have approxomately the same distribution of possible friend names and things you can find as our universe. One way to accomplish this is to do the following:

1. Buy two urns
2. Wander around randomly
3. For every person you come up to that you could be friends with, write their name down and put it in the first urn.
4. For every item that you could pick up, write it down and put it in the second urn.

Then you have an empirical probability distribution of possible friends and possible things and you can start to simulate universes. 

To do the simulation, you pull a friend name out of the first urn, write it down under the row for universe 1 in your spreadsheet, and put it back in the urn (You could be lucky enough to have the same friend in many universes!). Then draw the name of an item out of the second urn, write that down, put it back, and repeat so you have drawn 2 items.
Here is a possible outcome:
![One Simulation](/img/nonMatchUrn.gif)
To figure out if the hypothesis that Annie makes is feasable, you could do this 1000 times, making note of every time all the draws have the same letter. If that number is smaller than our p-value(also discussed in my last post), you are well on your way to rejecting the null hypothesis.

# The Problem: Filling Urns
The **one** hitch in our plan is that filling those urns will take way more time than it's worth. (Lets be honest, this has already taken more time than its worth...). Instead of wandering around in the world to fill the buckets, what if we pulled names of people  and things  from the written world instead of the real world? 

We could (and will) scrape the wikipedia web page for every childrens author and use entity extraction and part of speech tagging to find people and things (nouns).

While it will certainly reduce the amount of time until the end of this blog post, it comes at the price of having to make several assumptions. Namely, you have to assume that the distribution of people names in the wikipedia articles we are scraping is similar to the names of people in the real world (in other words, we have to assume you are quite friendly... Additionally, not every noun is something you could pick up (can you pick up the internet?). Follow along below and let me know if you think there is a better way to identify an item you could pick up...

So our final approach will be something like this:
![The Master Plan](/img/SimulationPlan.jpg)
# The Code
First things first, lets get to web scraping. One possible (and irrisponsible) way to do this is to simply point python's request library at a wikipedia url and parse the output with beautiful soup. This is taxing on the website and could result in any number of things including the web host blocking your IP address. Instead, its best to look for an API you can use to grab content from the sites.

Wikipedia, because it is awesome, as a great and [full featured api)[https://www.mediawiki.org/wiki/REST_API]. Even better, plenty of people have provided wrappers for the api so you don't have to manage and format your api requests. One that I have found useful (but hasn't been updated in a while is [here](https://pypi.org/project/wikipedia/)).

Lets import what we'll need, and search wikipedia for 'Childrens Author'


```python
import wikipedia
import numpy as np
import matplotlib.pyplot as plt
result = wikipedia.search('Childrens Author')
result
##["Children's literature",
##[" 'Sam Angus (writer)',
##[" "Alvin Schwartz (children's author)",
##[" 'Author! Author! (film)',
##[" "Douglas Evans (children's author)",
##[" "Ruth White (children's author)",
##[" 'Alicia Previn',
##[" "Al Perkins (children's author)",
##[" 'Plymouth Marjon University',
##[" "Margaret Mayo (children's author)"]
```

We could use the Childrens Literature page as a jumping off point, lets see what's in there:

```python
lit = wikipedia.page("Children's literature")
lit.links[1:15]

# ['ALA Editions',
#   'A Book of Giants',
#   'A Little Pretty Pocket-Book',
#   'Abanindranath Tagore',
#   'Acronym and initialism',
#   'Adventure book',
#   'Adventurer',
#   'Africa',
#   'African American',
#   'After the First Death',
#   'Alan Garner',
#   'Aleksandr Afanasyev',
#   'Alex Rider',
#   'Alexander Belyayev']
```

There are some random links, but many of the links appear to be to childrens authors or childrens books.


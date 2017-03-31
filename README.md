
# Data Exploration of Quora Questions
## Using Word2Vec and t-SNE 


Below, I explore a dataset of Quora questions. I train word2vec on the dataset to generate vector representations of words found in training data set questions, and then I use t-SNE to visualize some of the high-dimensional data stored in the word2vec model and understand how the word2vec model looks.

The data comes from a dataset that Quora recently posted on Kaggle. The aim of the competition they are hosting is to identify questions which have the same intent, but may be phrased or worded differently. Although the test set includes some computer-generated data to discourage cheating, the training set includes only valid question data from Quora - so my exploration below focuses exclusively on the training set data.

- __Load and Preview Data__
- __Formatting and Processing Data__
- __Train Word2Vec Model__
- __Visualize Word2Vec Model with t-SNE__
- __Conclusion__


# Load and Preview Data

First, we'll load the libraries necessary for manipulating and exploring this dataset. These include:
- __Word2Vec__: [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) is one of many implementation of Word2Vec, which allows us to generate high-dimensional vectors to represent each of the words in our corpus of Quora questions. Using high-dimensional vectors to represent each word allows us to view how they relate to other words in the corpus. We can use this representation of data to learn and store contextual meaning of the words in our datasets. 
- __NumPy__: The [NumPy](http://www.numpy.org/) library allows us to manipulate arrays and perform linear algebra operations.
- __Pandas__: The [Pandas]() library is useful for loading and manipulating our CSV dataset.
- __string__ and __nltk__: These libraries contain tools that allow us to prepare and process our text to ready it for __Word2Vec__ and to load it into the model for analysis.



```python
from gensim.models import Word2Vec
# from gensim.models.doc2vec import LabeledSentence
# from gensim.matutils import unitvec
import numpy as np
import pandas as pd
import nltk
import string
from nltk.corpus import stopwords

import logging
logging.root.handlers = []
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
```

Next, we will load and preview our dataset. We can see that our dataset includes the following fields:

- __id__: Unique numerical identifier for pair of questions
- __qid1__: Unique numerical identifier for first question in each pair
- __qid2__: Unique numerical identifier for second question in each pair
- __question1__: Text of first question in each pair
- __question2__: Text of second question in each pair
- __is_duplicate__: Training set label as to whether the two questions in a pair are duplicates. 0 means they are not duplicates, 1 means they are duplicates.



```python
# Read in and show the first few examples from our training set
q_train = pd.read_csv('train.csv')

q_train.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>qid1</th>
      <th>qid2</th>
      <th>question1</th>
      <th>question2</th>
      <th>is_duplicate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>What is the step by step guide to invest in sh...</td>
      <td>What is the step by step guide to invest in sh...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>What is the story of Kohinoor (Koh-i-Noor) Dia...</td>
      <td>What would happen if the Indian government sto...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>6</td>
      <td>How can I increase the speed of my internet co...</td>
      <td>How can Internet speed be increased by hacking...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>7</td>
      <td>8</td>
      <td>Why am I mentally very lonely? How can I solve...</td>
      <td>Find the remainder when [math]23^{24}[/math] i...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>9</td>
      <td>10</td>
      <td>Which one dissolve in water quikly sugar, salt...</td>
      <td>Which fish would survive in salt water?</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




# Formatting and Processing Data

After confirming how our data is structured, we need to prepare the text of the questions in our training set into a format suitable for training a word2vec model. This means that, instead of keeping our questions in a string format, we need to [tokenize](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html) each of them. Tokenization involves splitting up each question into a list of words.

While we are tokenizing, we also need to clean the dataset of words so that our word2vec model has a better understanding of overlapping contextual topics found in questions. Since we are more interested in the topics of these questions than in their grammatical format, we can go ahead and filter out all of the [stop words](https://en.wikipedia.org/wiki/Stop_words) (such as __the, is, at__ and __which__) while we are tokenizing each question.

We will store the tokenized format of each question into a new Panda series called __filteredQuestions__.



```python
# Create a series of all questions in the training data set
q_train_series = pd.Series(q_train['question1'].tolist() + q_train['question2'].tolist()).astype(str)

# We imported stopwords from the ntlk library, as well as string, above
# Here, we create sets of English stopwords and punctuation marks
s = set(stopwords.words('english'))
exclude = set(string.punctuation)

# This method allows us to tokenize and filter each question to prepare for word2vec
def filterStopWordsAndPunct(queryWords):
    
    # First, we remove all punctuation marks from the question
    queryWords = ''.join(ch for ch in queryWords if ch not in exclude)
    
    # Then, we split up the words into a list (tokenization)
    qwords = queryWords.split()
    
    # Finally, we filter out any words found in our set of stopwords
    resultwords  = [word for word in qwords if word.lower() not in s]
    
    return resultwords

# Let's create filteredQuestions, a new set to store all of our tokenized questions inside of
filteredQuestions = q_train_series.copy()

# Now let's tokenize each of our questions and store it inside of filteredQuestions
for index, value in q_train_series.items():
    filteredQuestions[index] = filterStopWordsAndPunct(value)

```


# Train Word2Vec Model

Below, we can see what some of our filtered and tokenized questions look like, by previewing the Series we created above. It is easy to see how, by filtering out stop words, we have kept only the necessary key words that are necessary to reconstruct and understand the meaning of each of these intriguing questions.

This preparation will ensure that our word2vec model is trained in a manner that allows the high-dimensional vector for each word to store only the most important contextual information.



```python
filteredQuestions.values[:20]
```




    array([['step', 'step', 'guide', 'invest', 'share', 'market', 'india'],
           ['story', 'Kohinoor', 'KohiNoor', 'Diamond'],
           ['increase', 'speed', 'internet', 'connection', 'using', 'VPN'],
           ['mentally', 'lonely', 'solve'],
           ['one', 'dissolve', 'water', 'quikly', 'sugar', 'salt', 'methane', 'carbon', 'di', 'oxide'],
           ['Astrology', 'Capricorn', 'Sun', 'Cap', 'moon', 'cap', 'risingwhat', 'say'],
           ['buy', 'tiago'], ['good', 'geologist'],
           ['use', 'シ', 'instead', 'し'],
           ['Motorola', 'company', 'hack', 'Charter', 'Motorolla', 'DCX3400'],
           ['Method', 'find', 'separation', 'slits', 'using', 'fresnel', 'biprism'],
           ['read', 'find', 'YouTube', 'comments'],
           ['make', 'Physics', 'easy', 'learn'],
           ['first', 'sexual', 'experience', 'like'],
           ['laws', 'change', 'status', 'student', 'visa', 'green', 'card', 'US', 'compare', 'immigration', 'laws', 'Canada'],
           ['would', 'Trump', 'presidency', 'mean', 'current', 'international', 'master’s', 'students', 'F1', 'visa'],
           ['manipulation', 'mean'],
           ['girls', 'want', 'friends', 'guy', 'reject'],
           ['many', 'Quora', 'users', 'posting', 'questions', 'readily', 'answered', 'Google'],
           ['best', 'digital', 'marketing', 'institution', 'banglore']], dtype=object)




The Gensim Word2Vec tool makes training our model quite easy. We can pass in the values of the Series of tokenized and filtered questions, and specify parameters such as:

- `size`: specifies how high-dimensional the feature vectors for each word should be.
- `window`: specifies maximum possible distance between current word and predicted word in a sentence.
- `sg`: specifies which algorithm is used to train the neural network. If set to 1, skip-gram is used. Otherwise, by default, the continuous bag-of-words algorithm is used.

The log below shows the steps involved in collecting the words in the full corpus and training the Word2Vec model on them.



```python
model = Word2Vec(filteredQuestions.values, size=100, window=5)
```

    2017-03-28 18:34:27,750 : INFO : collecting all words and their counts
    2017-03-28 18:34:27,751 : INFO : PROGRESS: at sentence #0, processed 0 words, keeping 0 word types
    2017-03-28 18:34:27,775 : INFO : PROGRESS: at sentence #10000, processed 55066 words, keeping 13545 word types
    2017-03-28 18:34:27,807 : INFO : PROGRESS: at sentence #20000, processed 110321 words, keeping 20153 word types
    2017-03-28 18:34:27,848 : INFO : PROGRESS: at sentence #30000, processed 165268 words, keeping 25268 word types
    2017-03-28 18:34:27,902 : INFO : PROGRESS: at sentence #40000, processed 219826 words, keeping 29579 word types
    2017-03-28 18:34:27,944 : INFO : PROGRESS: at sentence #50000, processed 275098 words, keeping 33599 word types
    2017-03-28 18:34:27,997 : INFO : PROGRESS: at sentence #60000, processed 330139 words, keeping 37137 word types
    2017-03-28 18:34:28,126 : INFO : PROGRESS: at sentence #70000, processed 385134 words, keeping 40234 word types
    2017-03-28 18:34:28,162 : INFO : PROGRESS: at sentence #80000, processed 440258 words, keeping 43055 word types
    2017-03-28 18:34:28,198 : INFO : PROGRESS: at sentence #90000, processed 495286 words, keeping 45801 word types
    2017-03-28 18:34:28,253 : INFO : PROGRESS: at sentence #100000, processed 550281 words, keeping 48447 word types
    2017-03-28 18:34:28,279 : INFO : PROGRESS: at sentence #110000, processed 604652 words, keeping 50905 word types
    2017-03-28 18:34:28,318 : INFO : PROGRESS: at sentence #120000, processed 659485 words, keeping 53295 word types
    2017-03-28 18:34:28,347 : INFO : PROGRESS: at sentence #130000, processed 714703 words, keeping 55625 word types
    2017-03-28 18:34:28,403 : INFO : PROGRESS: at sentence #140000, processed 769684 words, keeping 57740 word types
    2017-03-28 18:34:28,487 : INFO : PROGRESS: at sentence #150000, processed 825035 words, keeping 59933 word types
    2017-03-28 18:34:28,522 : INFO : PROGRESS: at sentence #160000, processed 880416 words, keeping 62073 word types
    2017-03-28 18:34:28,548 : INFO : PROGRESS: at sentence #170000, processed 935645 words, keeping 64074 word types
    2017-03-28 18:34:28,595 : INFO : PROGRESS: at sentence #180000, processed 991077 words, keeping 66029 word types
    2017-03-28 18:34:28,629 : INFO : PROGRESS: at sentence #190000, processed 1046201 words, keeping 67892 word types
    2017-03-28 18:34:28,667 : INFO : PROGRESS: at sentence #200000, processed 1100917 words, keeping 69734 word types
    2017-03-28 18:34:28,713 : INFO : PROGRESS: at sentence #210000, processed 1155599 words, keeping 71533 word types
    2017-03-28 18:34:28,769 : INFO : PROGRESS: at sentence #220000, processed 1210471 words, keeping 73211 word types
    2017-03-28 18:34:28,819 : INFO : PROGRESS: at sentence #230000, processed 1266067 words, keeping 74941 word types
    2017-03-28 18:34:28,850 : INFO : PROGRESS: at sentence #240000, processed 1321076 words, keeping 76560 word types
    2017-03-28 18:34:28,902 : INFO : PROGRESS: at sentence #250000, processed 1376715 words, keeping 78169 word types
    2017-03-28 18:34:28,935 : INFO : PROGRESS: at sentence #260000, processed 1432248 words, keeping 79770 word types
    2017-03-28 18:34:28,968 : INFO : PROGRESS: at sentence #270000, processed 1487831 words, keeping 81322 word types
    2017-03-28 18:34:29,025 : INFO : PROGRESS: at sentence #280000, processed 1542831 words, keeping 82825 word types
    2017-03-28 18:34:29,074 : INFO : PROGRESS: at sentence #290000, processed 1598092 words, keeping 84316 word types
    2017-03-28 18:34:29,101 : INFO : PROGRESS: at sentence #300000, processed 1652861 words, keeping 85823 word types
    2017-03-28 18:34:29,156 : INFO : PROGRESS: at sentence #310000, processed 1707937 words, keeping 87267 word types
    2017-03-28 18:34:29,241 : INFO : PROGRESS: at sentence #320000, processed 1762863 words, keeping 88631 word types
    2017-03-28 18:34:29,323 : INFO : PROGRESS: at sentence #330000, processed 1818330 words, keeping 90076 word types
    2017-03-28 18:34:29,403 : INFO : PROGRESS: at sentence #340000, processed 1873266 words, keeping 91371 word types
    2017-03-28 18:34:29,491 : INFO : PROGRESS: at sentence #350000, processed 1928414 words, keeping 92652 word types
    2017-03-28 18:34:29,585 : INFO : PROGRESS: at sentence #360000, processed 1983438 words, keeping 93949 word types
    2017-03-28 18:34:29,617 : INFO : PROGRESS: at sentence #370000, processed 2038566 words, keeping 95244 word types
    2017-03-28 18:34:29,669 : INFO : PROGRESS: at sentence #380000, processed 2093920 words, keeping 96559 word types
    2017-03-28 18:34:29,719 : INFO : PROGRESS: at sentence #390000, processed 2149934 words, keeping 97893 word types
    2017-03-28 18:34:29,776 : INFO : PROGRESS: at sentence #400000, processed 2205563 words, keeping 99104 word types
    2017-03-28 18:34:29,839 : INFO : PROGRESS: at sentence #410000, processed 2261345 words, keeping 100244 word types
    2017-03-28 18:34:29,879 : INFO : PROGRESS: at sentence #420000, processed 2316960 words, keeping 101339 word types
    2017-03-28 18:34:29,930 : INFO : PROGRESS: at sentence #430000, processed 2372924 words, keeping 102448 word types
    2017-03-28 18:34:29,994 : INFO : PROGRESS: at sentence #440000, processed 2428503 words, keeping 103432 word types
    2017-03-28 18:34:30,057 : INFO : PROGRESS: at sentence #450000, processed 2484583 words, keeping 104369 word types
    2017-03-28 18:34:30,112 : INFO : PROGRESS: at sentence #460000, processed 2540386 words, keeping 105355 word types
    2017-03-28 18:34:30,221 : INFO : PROGRESS: at sentence #470000, processed 2596088 words, keeping 106367 word types
    2017-03-28 18:34:30,254 : INFO : PROGRESS: at sentence #480000, processed 2651995 words, keeping 107373 word types
    2017-03-28 18:34:30,300 : INFO : PROGRESS: at sentence #490000, processed 2707983 words, keeping 108331 word types
    2017-03-28 18:34:30,363 : INFO : PROGRESS: at sentence #500000, processed 2763996 words, keeping 109296 word types
    2017-03-28 18:34:30,394 : INFO : PROGRESS: at sentence #510000, processed 2818862 words, keeping 110238 word types
    2017-03-28 18:34:30,438 : INFO : PROGRESS: at sentence #520000, processed 2874451 words, keeping 111157 word types
    2017-03-28 18:34:30,504 : INFO : PROGRESS: at sentence #530000, processed 2930602 words, keeping 112047 word types
    2017-03-28 18:34:30,559 : INFO : PROGRESS: at sentence #540000, processed 2986730 words, keeping 113056 word types
    2017-03-28 18:34:30,612 : INFO : PROGRESS: at sentence #550000, processed 3043060 words, keeping 114025 word types
    2017-03-28 18:34:30,673 : INFO : PROGRESS: at sentence #560000, processed 3099088 words, keeping 114936 word types
    2017-03-28 18:34:30,701 : INFO : PROGRESS: at sentence #570000, processed 3155285 words, keeping 115805 word types
    2017-03-28 18:34:30,744 : INFO : PROGRESS: at sentence #580000, processed 3211557 words, keeping 116683 word types
    2017-03-28 18:34:30,812 : INFO : PROGRESS: at sentence #590000, processed 3268029 words, keeping 117547 word types
    2017-03-28 18:34:30,863 : INFO : PROGRESS: at sentence #600000, processed 3323219 words, keeping 118375 word types
    2017-03-28 18:34:30,916 : INFO : PROGRESS: at sentence #610000, processed 3379258 words, keeping 119270 word types
    2017-03-28 18:34:30,959 : INFO : PROGRESS: at sentence #620000, processed 3435028 words, keeping 120132 word types
    2017-03-28 18:34:30,997 : INFO : PROGRESS: at sentence #630000, processed 3490724 words, keeping 121045 word types
    2017-03-28 18:34:31,044 : INFO : PROGRESS: at sentence #640000, processed 3545867 words, keeping 121879 word types
    2017-03-28 18:34:31,070 : INFO : PROGRESS: at sentence #650000, processed 3601886 words, keeping 122735 word types
    2017-03-28 18:34:31,119 : INFO : PROGRESS: at sentence #660000, processed 3657562 words, keeping 123641 word types
    2017-03-28 18:34:31,176 : INFO : PROGRESS: at sentence #670000, processed 3713757 words, keeping 124525 word types
    2017-03-28 18:34:31,212 : INFO : PROGRESS: at sentence #680000, processed 3770087 words, keeping 125468 word types
    2017-03-28 18:34:31,264 : INFO : PROGRESS: at sentence #690000, processed 3825909 words, keeping 126304 word types
    2017-03-28 18:34:31,299 : INFO : PROGRESS: at sentence #700000, processed 3881645 words, keeping 127173 word types
    2017-03-28 18:34:31,344 : INFO : PROGRESS: at sentence #710000, processed 3937012 words, keeping 127976 word types
    2017-03-28 18:34:31,411 : INFO : PROGRESS: at sentence #720000, processed 3993057 words, keeping 128795 word types
    2017-03-28 18:34:31,474 : INFO : PROGRESS: at sentence #730000, processed 4048365 words, keeping 129592 word types
    2017-03-28 18:34:31,530 : INFO : PROGRESS: at sentence #740000, processed 4104115 words, keeping 130380 word types
    2017-03-28 18:34:31,647 : INFO : PROGRESS: at sentence #750000, processed 4159707 words, keeping 131149 word types
    2017-03-28 18:34:31,724 : INFO : PROGRESS: at sentence #760000, processed 4215635 words, keeping 131919 word types
    2017-03-28 18:34:31,807 : INFO : PROGRESS: at sentence #770000, processed 4271595 words, keeping 132711 word types
    2017-03-28 18:34:31,869 : INFO : PROGRESS: at sentence #780000, processed 4327378 words, keeping 133473 word types
    2017-03-28 18:34:31,944 : INFO : PROGRESS: at sentence #790000, processed 4383334 words, keeping 134355 word types
    2017-03-28 18:34:32,001 : INFO : PROGRESS: at sentence #800000, processed 4440048 words, keeping 135126 word types
    2017-03-28 18:34:32,037 : INFO : collected 135755 word types from a corpus of 4488531 raw words and 808580 sentences
    2017-03-28 18:34:32,038 : INFO : Loading a fresh vocabulary
    2017-03-28 18:34:32,384 : INFO : min_count=5 retains 36680 unique words (27% of original 135755, drops 99075)
    2017-03-28 18:34:32,385 : INFO : min_count=5 leaves 4334597 word corpus (96% of original 4488531, drops 153934)
    2017-03-28 18:34:32,592 : INFO : deleting the raw counts dictionary of 135755 items
    2017-03-28 18:34:32,601 : INFO : sample=0.001 downsamples 22 most-common words
    2017-03-28 18:34:32,602 : INFO : downsampling leaves estimated 4182229 word corpus (96.5% of prior 4334597)
    2017-03-28 18:34:32,604 : INFO : estimated required memory for 36680 words and 100 dimensions: 47684000 bytes
    2017-03-28 18:34:32,857 : INFO : resetting layer weights
    2017-03-28 18:34:33,327 : INFO : training model with 3 workers on 36680 vocabulary and 100 features, using sg=0 hs=0 sample=0.001 negative=5 window=5
    2017-03-28 18:34:33,328 : INFO : expecting 808580 sentences, matching count from corpus used for vocabulary survey
    2017-03-28 18:34:34,356 : INFO : PROGRESS: at 2.52% examples, 511938 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:35,357 : INFO : PROGRESS: at 5.97% examples, 612669 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:36,359 : INFO : PROGRESS: at 9.73% examples, 667914 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:37,364 : INFO : PROGRESS: at 12.70% examples, 656437 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:38,381 : INFO : PROGRESS: at 16.29% examples, 674076 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:39,390 : INFO : PROGRESS: at 20.09% examples, 694348 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:40,397 : INFO : PROGRESS: at 23.82% examples, 704565 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:41,417 : INFO : PROGRESS: at 27.50% examples, 709997 words/s, in_qsize 6, out_qsize 1
    2017-03-28 18:34:42,428 : INFO : PROGRESS: at 31.15% examples, 715022 words/s, in_qsize 6, out_qsize 0
    2017-03-28 18:34:43,430 : INFO : PROGRESS: at 34.51% examples, 714306 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:44,437 : INFO : PROGRESS: at 38.01% examples, 715962 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:45,439 : INFO : PROGRESS: at 41.53% examples, 717546 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:46,450 : INFO : PROGRESS: at 44.90% examples, 715378 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:47,462 : INFO : PROGRESS: at 46.82% examples, 692438 words/s, in_qsize 6, out_qsize 0
    2017-03-28 18:34:48,485 : INFO : PROGRESS: at 48.84% examples, 673250 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:49,495 : INFO : PROGRESS: at 51.06% examples, 660002 words/s, in_qsize 6, out_qsize 0
    2017-03-28 18:34:50,530 : INFO : PROGRESS: at 53.76% examples, 653394 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:51,534 : INFO : PROGRESS: at 56.42% examples, 648115 words/s, in_qsize 6, out_qsize 0
    2017-03-28 18:34:52,536 : INFO : PROGRESS: at 58.77% examples, 640056 words/s, in_qsize 6, out_qsize 0
    2017-03-28 18:34:53,582 : INFO : PROGRESS: at 61.71% examples, 637345 words/s, in_qsize 4, out_qsize 1
    2017-03-28 18:34:54,596 : INFO : PROGRESS: at 64.45% examples, 633608 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:55,606 : INFO : PROGRESS: at 67.14% examples, 629918 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:56,647 : INFO : PROGRESS: at 69.87% examples, 626111 words/s, in_qsize 6, out_qsize 1
    2017-03-28 18:34:57,648 : INFO : PROGRESS: at 72.80% examples, 625666 words/s, in_qsize 5, out_qsize 1
    2017-03-28 18:34:58,654 : INFO : PROGRESS: at 76.11% examples, 628465 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:34:59,662 : INFO : PROGRESS: at 79.91% examples, 634904 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:35:00,682 : INFO : PROGRESS: at 83.55% examples, 638759 words/s, in_qsize 6, out_qsize 0
    2017-03-28 18:35:01,691 : INFO : PROGRESS: at 87.32% examples, 643550 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:35:02,698 : INFO : PROGRESS: at 91.02% examples, 647796 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:35:03,710 : INFO : PROGRESS: at 94.64% examples, 651410 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:35:04,713 : INFO : PROGRESS: at 98.41% examples, 655887 words/s, in_qsize 5, out_qsize 0
    2017-03-28 18:35:05,118 : INFO : worker thread finished; awaiting finish of 2 more threads
    2017-03-28 18:35:05,130 : INFO : worker thread finished; awaiting finish of 1 more threads
    2017-03-28 18:35:05,135 : INFO : worker thread finished; awaiting finish of 0 more threads
    2017-03-28 18:35:05,136 : INFO : training on 22442655 raw words (20912202 effective words) took 31.8s, 657712 effective words/s



# Visualize Word2Vec Model with t-SNE

Once our Word2Vec model is trained, we have a series of vectors for each word in the Quora question corpus. Each of these vectors has a hundred dimensions that supply data about the context a specific word is found in, relative to other words.

How can we better understand the relationships between words that our model has identified? We can use the t-SNE (t-distributed stochastic neighbor embedding) algorithm to embed our high-dimensional data in a two-dimensional scatter plot.

Below, we import __pyplot__ from __matplotlib__ to show our visualizations, and __TSNE__ from __sklearn__ to create our t-SNE model. We also import some helper tools from __mpl_toolkits__ so that we can create some inset graphs to better inspect our t-SNE visualizations.



```python
import sys
import codecs
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

```


First, we'll need to extract our matrix of word vectors from the word2vec model, and then feed it to the t-SNE algorithm to fit X to a 2-dimensional space. We need to keep this chunk of code separate from the upcoming visualization code, because t-SNE's cost function is non-convext - meaning that the fitting method has different output everytime it runs. If you run this code, chances are your visualizations will look different from mine below.

We'll take only the first thousand word vectors from our matrix, so that our visualization is manageable for this iPython notebook.



```python
# Extract matrix of word vectors from word2vec model
X = model[model.wv.vocab]

# Transform data using t-SNE to fit to 2 dimensions
# We're only taking the first 1000 word-vectors for now
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X[:1000,:])

```


## Visualization 1: Social Media and Online Interactions

Finally, we can create a scatter plot to visualize the two-dimensional output data from t-SNE. We'll also create an inset chart in the upper left corner of this scatter plot, so that we can observe any interesting clusters of words in greater detail.



```python
# Create a new plot, split into figure and axes elements
fig, ax = plt.subplots(figsize=[20,10])

# Add our data to the new plot axes
ax.scatter(X_tsne[:, 0], X_tsne[:, 1])

# Create a new set of in-set axes, and populate with the same data
axins = zoomed_inset_axes(ax, 3, loc=2) # zoom-factor: 3, location: upper-left
axins.scatter(X_tsne[:, 0], X_tsne[:, 1])

# Define region that we want to show in our in-set chart
x1, x2, y1, y2 = 33, 38, -5, 0 # specify the limits of our inset graph
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits

# Turn off tick-marks for axes
plt.yticks(visible=False)
plt.xticks(visible=False)

# Add labeling for scatter plot points in both our large and in-set charts
vocab_word_names = list(model.wv.vocab.keys());

for label, x, y in zip(vocab_word_names, X_tsne[:, 0], X_tsne[:, 1]):
        ax.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        axins.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        
# Add lines to show where inset is focused in larger plot
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

# Draw and show chart
plt.draw()
plt.show()


```


![png](output_16_0.png)



In the chart above, we see a two-dimensional representation of an approximation of how the first thousand word vectors are spatially arranged, from our matrix from the word2vec model. Many interesting clusters of similar kinds of words show up, highlighting areas where our word2vec model was able to learn broader topics that Quora questions commonly focused on.

The in-set chart above highlights one such cluster of words related to social media and online interactions. We see Twitter, Skype and Instagram located close together, along with words about interactions on these platforms, such as "followers", "pictures", "comments", "block" and "spam."



## Visualization 2: Measuring Time

Let's create the same chart again, but this time we will focus our in-set zoom on another interesting cluster which is less "topical" but more "functional" in the types of words it identifies as being similar.



```python
# Create a new plot, split into figure and axes elements
fig, ax = plt.subplots(figsize=[20,10])

# Add our data to the new plot axes
ax.scatter(X_tsne[:, 0], X_tsne[:, 1])

# Create a new set of in-set axes, and populate with the same data
axins = zoomed_inset_axes(ax, 3, loc=2) # zoom-factor: 3, location: upper-left
axins.scatter(X_tsne[:, 0], X_tsne[:, 1])

# Define region that we want to show in our in-set chart
x1, x2, y1, y2 = -14, -9, -19, -14 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits

# Turn off tick-marks for axes
plt.yticks(visible=False)
plt.xticks(visible=False)

# Add labeling for scatter plot points in both our large and in-set charts
vocab_word_names = list(model.wv.vocab.keys());

for label, x, y in zip(vocab_word_names, X_tsne[:, 0], X_tsne[:, 1]):
        ax.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        axins.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        
# Add lines to show where inset is focused in larger plot
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

# Draw and show chart
plt.draw()
plt.show()
```


![png](output_19_0.png)



What we see here is an interesting linear cluster of words representing different periods of time, including "minutes", "hours", "days", "months" and "years." Perhaps the word2vec model was able to pick up on contextual clues such as the words "for" and "long" in questions to identify that these words fulfilled similar functions.

If we use the "most_similar" function of our word2vec model for the word "years", we see many of the same words visible in this cluster show up.



```python
model.most_similar("years")
```




    [('yrs', 0.8314576148986816),
     ('months', 0.7631711959838867),
     ('year', 0.7351706027984619),
     ('yr', 0.7314525842666626),
     ('days', 0.7190325260162354),
     ('weeks', 0.6372518539428711),
     ('decades', 0.6279939413070679),
     ('semesters', 0.618726372718811),
     ('LPA', 0.5738800168037415),
     ('minutes', 0.5627063512802124)]



# Conclusion

As is visible from the above exploration, word2vec is a powerful tool, even with a fairly limited dataset. The model we've created could conceivably be used to evaluate similarity between words and questions, or even to create new questions. By creating vectorized representations of the words in our corpus, we are able to train our model to learn not just about word frequency, but also about contextual meanings.

t-SNE allows us to take our resulting high-dimensional matrix and create an approximation of that matrix in two dimensions. We can quickly pick out interesting topical and functional clusters of words whose distances aren't too far from each other in vector-space. This gives us a sense of the usefulness of modelling context when it comes to understanding meaning in language.


```python

```

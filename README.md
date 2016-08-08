
---
**Recurrent Neural Network - Final Project**
---
***Authors:*** Ronit Yoari & Orel Swisa  
***Date:*** August 8, 2016  
***[Dataset](http://www.gutenberg.org/cache/epub/8001/pg8001.txt)***  

---
The Code in python with RNN algorithm used to generate the data recovery model documented and explained in the ipython notebook: "Report".  

# Installations:
**Anaconda for Windows**  
Windows 64-BIT  
Anaconda Version: Anaconda 2  
including Python 2.7  
**NLTK**  
Version: 64-bit Version
**NUMPY**  
**THEANO**  
For using GPU : **NVIDIA Nsight HUD Launcher 4.7**
  
---
# 1. Choosing the data collection sequences
---
## - Collection of data that we have chosen for this project is:  
We were looking to investigate a subject that interests us both, and we decided on a ***biblical writing style***.  
The first book in the bible - **Genesis** is the one that we choose to explore.  
  
## - Link to dataset:  
[Gensis](http://www.gutenberg.org/cache/epub/8001/pg8001.txt)
  
## - Data Example - Reading first data records  
1. In the beginning God created the heaven and the earth.  
  
2. And the earth was without form, and void; and darkness was  
upon the face of the deep. And the Spirit of God moved upon  
the face of the waters.  
  
3. And God said, Let there be light: and there was light.  
  
4. And God saw the light, that it was good: and God divided the  
light from the darkness.  
  
5. And God called the light Day, and the darkness he called  
Night. And the evening and the morning were the first day.  
  
6. And God said, Let there be a firmament in the midst of the  
waters, and let it divide the waters from the waters.  
  
7. And God made the firmament, and divided the waters which were  
under the firmament from the waters which were above the  
firmament: and it was so.  
  
8. And God called the firmament Heaven. And the evening and the  
morning were the second day.  
  
9. And God said, Let the waters under the heaven be gathered  
together unto one place, and let the dry land appear: and it  
was so.  
  
10. And God called the dry land Earth; and the gathering together  
of the waters called he Seas: and God saw that it was good.

---
# 2. Data Description
---
## Description of data collection chosen:  
The Book of Genesis is the first book of the Hebrew Bible (the Tanakh) and the Christian Old Testament.  
The basic narrative expresses the central theme: God creates the world (along with creating the first man and woman) and appoints man as his regent, but man proves disobedient and God destroys his world through the Flood. The new post-Flood world is equally corrupt, but God does not destroy it, instead calling one man, Abraham, to be the seed of its salvation. At God's command Abraham descends from his home into the land of Canaan, given to him by God, where he dwells as a sojourner, as does his son Isaac and his grandson Jacob. Jacob's name is changed to Israel, and through the agency of his son Joseph, the children of Israel descend into Egypt, 70 people in all with their households, and God promises them a future of greatness. Genesis ends with Israel in Egypt, ready for the coming of Moses and the Exodus. The narrative is punctuated by a series of covenants with God, successively narrowing in scope from all mankind (the covenant with Noah) to a special relationship with one people alone (Abraham and his descendants through Isaac and Jacob).  
## The main challenges of working with this data were:  
### The main difficulties that we faced with them during the work were:  
- Finding a source that contains the text.  
- Pre-processing is required (done in Notepad ++) to adjust the text to the form that we could work with it.  
## Why the study interesting and where he can contribute?  
- Research on this data is interesting because it deals with an ancient text written thousands of years ago, and we wonder what features it has and whether we can although the ancient writing style to create a meaningful new information logically.  
- This research can contribute to understanding the biblical writing style - whether it is similar to current writing style? What has changed? And to answer many other questions that arise in this topic.  

---
# 3. Data Preprocessing
---
## Stage 1: TOKENIZE TEXT  
We have raw text, but we want to make predictions on a per-word basis.  
This means we must tokenize our comments into sentences, and sentences into words.   
At first, we want to choose the dictionary that we'll work with him.  
We choose the collection of the most common words in the text.   
Tokens:  


```
_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '2500'))
vocabulary_size = _VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
```

Read the data, get raw text


```
print "Reading file..."
path = '../Genesis.txt'
f = open(path)
raw = f.read()
```

Split the text file into sentences

Comments into sentences:


```
from nltk import sent_tokenize
sentences = sent_tokenize(raw)
```

## Stage 2: PREPEND SPECIAL START AND END TOKENS
We also want to learn which words tend start and end a sentence.  
To do this we prepend a special ***SENTENCE_START*** token, and append a special ***SENTENCE_END*** token to each sentence:  


```
sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))
```

**Output:**  
Parsed 1467 sentences.  
## Stage 3: REMOVE INFREQUENT WORDS  
Most words in our text will only appear one or two times.  
Itâ€™s a good idea to remove these infrequent words. Having a huge vocabulary will make our model slow to train, and because we donâ€™t have a lot of contextual examples for such words we wouldnâ€™t be able to learn how to use them correctly anyway.  

Tokenize the sentences into words


```
from nltk import word_tokenize
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
print 'after tokenized_sentences'
```

Count the word frequencies


```
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())
```

**Output:**  
Found 2633 unique words tokens. 
 
Get the most common words and build index_to_word and word_to_index vectors
limit our vocabulary to the vocabulary_size most common words (which was set to 2500)
The word UNKNOWN_TOKEN will become part of our vocabulary and we will predict it just like any other word.


```
vocab = word_freq.most_common(vocabulary_size-1)
```

replace all words not included in our vocabulary by **UNKNOWN_TOKEN**


```
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
```

**Output:**  
Using vocabulary size 2500.  
The least frequent word in our vocabulary is 'friends' and appeared 1 times.


Replace all words not in our vocabulary with the **unknown token** 


```
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
```

## Stage 4: TRAINING DATA MATRICES  
The input to our Recurrent Neural Networks are vectors, not strings. So we create a mapping between words and indices.  
**Create the training data:**


```
import numpy as np
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
```

---
# 4. The Code in python with RNN algorithm used to generate the data recovery model
---  

**The training function:**  


```
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print "Setting learning rate to %f" % learning_rate
            sys.stdout.flush()
            # ADDED! Saving model oarameters
            save_model_parameters_theano("../data/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            print i
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
```

---
# 5. Data generating code sequences according to the model being studied
---

**The code that generate a new sentence according to the model:**


```
## GENERATING TEXT
def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str
```

---
# 6. Building and operating model, using it for creating information
---


```
from rnn import RNNTheano

model = RNNTheano(vocabulary_size, hidden_dim=_HIDDEN_DIM)
tStart = time.time()
model.sgd_step(X_train[10], y_train[10], _LEARNING_RATE)
tEnd = time.time()
print "SGD Step time: %f milliseconds" % ((tEnd - tStart) * 1000.)

if _MODEL_FILE != None:
    load_model_parameters_theano(_MODEL_FILE, model)

train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)

print 'end traing'

##GENERATING TEXT
num_sentences = 10
senten_min_length = 7

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while operator.gt(len(sent),senten_min_length):
        sent = generate_sentence(model)
    print " ".join(sent)
```

## Output:    
- and god said unto him to you shall be his servant  
- of the sons of jacob's firstborn  
- these are the days of heaven were covered  
- god called the lord said i will give it came to drink  
- and the morning were the fifth day  
- and they beginning of every one speech  
- then he begat salah lived after thee  
- and i pray thee a little food  
- and every living creature that the lord be clear ourselves  
- and take to keep me for ever  

---
# 7. Evaluation of quality information reconstructed by comparing the sequences synthesized
---

## This function check similarity between the generated sentences and the original text:


```
# Import the necessary packages for preforming similarity between texts.
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.feature_extraction.text import TfidfVectorizer as tv

# Load the texts - The original & The generated
gensis = open('../dataset/Genesis.txt','r').read().split('\r')
model_gensis = open('../outputFiles/Model_Genesis.txt','r').read().split('\n')

# Initialize
TfV = tv()
TfV.fit(gensis)
Y = TfV.transform(gensis)

# Check for every sentence the similarity
similaritySum = 0
for sentence in model_gensis:
    X = TfV.transform([sentence])
    print(sentence)
    print(gensis[cs(X, Y).argmax()])
    print(' ')
    similaritySum = cs(X, Y).max()

# Calculate the similarity
similarity = similaritySum/len(model_gensis)
print('The similarity between the original text - Genesis -  and the model is: ' , similarity)
```

## Output:  
  
**Generated:** and god said unto him to you shall be his servant  
**Original:**  And he said, Blessed be the LORD God of Shem; and Canaan shall be his servant.  
   
**Generated:** of the sons of jacob's firstborn  
**Original:**  And these are the names of the children of Israel, which came into Egypt, Jacob and his sons: Reuben, Jacob's firstborn.  
   
**Generated:** these are the days of heaven were covered  
**Original:**  And the waters prevailed exceedingly upon the earth; and all the high hills, that were under the whole heaven, were covered.   
   
**Generated:** god called the lord said i will give it came to drink  
**Original:**  And, behold, the LORD stood above it, and said, I am the LORD God of Abraham thy father, and the God of Isaac: the land whereon thou liest, to thee will I give it, and to thy seed;  
   
**Generated:** and the morning were the fifth day   
**Original:**  And the evening and the morning were the fifth day.  
   
**Generated:** and they beginning of every one speech  
**Original:**  And the whole earth was of one language, and of one speech.  
   
**Generated:** then he begat salah lived after thee  
**Original:**  And Arphaxad begat Salah; and Salah begat Eber.  
   
**Generated:** and i pray thee a little food  
**Original:**  And our father said, Go again, and buy us a little food.  
   
**Generated:** and every living creature that the lord be clear ourselves  
**Original:**  And with every living creature that is with you, of the fowl, of the cattle, and of every beast of the earth with you; from all that go out of the ark, to every beast of the earth.   
   
**Generated:** and take to keep me for ever  
**Original:**  For all the land which thou seest, to thee will I give it, and to thy seed for ever.  
   
('The similarity between the original text - Genesis -  and the model is: ', 0.057482169031696125)  

---
# 8. Analysis of the study results and conclusions
---


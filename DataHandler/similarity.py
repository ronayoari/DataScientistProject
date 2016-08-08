# Import the necessary packages for preforming similarity between texts.
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.feature_extraction.text import TfidfVectorizer as tv

# Load the texts - The original & The generated
gensis = open('../Genesis.txt','r').read().split('\r')
model_gensis = open('../Model_Genesis.txt','r').read().split('\n')

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
similarity = similaritySum/7
print('The similarity between the original text - Genesis -  and the model is: ' , similarity)
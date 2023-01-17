import tensorflow as tf
import numpy as np
import nltk

# Download the NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Read the text file
with open("data.txt", "r") as file:
    data = file.readlines()

# Extract the prompts and responses
prompts = []
responses = []
for line in data:
    line = line.strip().split("\t")
    prompts.append(line[0])
    if len(line) > 1:
        responses.append(line[1])

# Tokenize the prompts and responses
prompts_tokens = [nltk.word_tokenize(prompt) for prompt in prompts]
responses_tokens = [nltk.word_tokenize(response) for response in responses]

# Remove stop words
stop_words = nltk.corpus.stopwords.words("english")
prompts_tokens = [[word for word in prompt if word.lower() not in stop_words] for prompt in prompts_tokens]
responses_tokens = [[word for word in response if word.lower() not in stop_words] for response in responses_tokens]

# Stem the tokens
stemmer = nltk.stem.PorterStemmer()
prompts_tokens = [[stemmer.stem(word) for word in prompt] for prompt in prompts_tokens]
responses_tokens = [[stemmer.stem(word) for word in response] for response in responses_tokens]

# Convert the prompts and responses to numerical data
x_train = np.array(prompts)
y_train = np.array(responses)

# Quickly check data
if len(prompts) == len(responses):
    # Train the model if the data passes the check
    model.fit(x_train, y_train, epochs=10)
else:
    print("Data cardinality is ambiguous: x sizes:",len(prompts),", y sizes:",len(responses))
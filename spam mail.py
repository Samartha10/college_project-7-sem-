import math

# Sample dataset
dataset = [
    (['buy', 'now', 'limited', 'offer'], 'spam'),
    (['hello', 'how', 'are', 'you'], 'not_spam'),
    (['exclusive', 'deal', 'for', 'you'], 'spam'),
    (['meeting', 'tomorrow', 'at', '10', 'am'], 'not_spam'),
    (['claim', 'your', 'prize', 'today'], 'spam'),
    # Add more examples as needed
]

# Split dataset into training and testing sets
train_set = dataset[:int(0.8 * len(dataset))]
test_set = dataset[int(0.8 * len(dataset)):]

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.word_probs = {}

    def fit(self, training_data):
        # Calculate class probabilities
        total_messages = len(training_data)
        spam_messages = [message for message, label in training_data if label == 'spam']
        not_spam_messages = [message for message, label in training_data if label == 'not_spam']

        self.class_probs['spam'] = len(spam_messages) / total_messages
        self.class_probs['not_spam'] = len(not_spam_messages) / total_messages

        # Calculate word probabilities
        all_words = set([word for message, _ in training_data for word in message])

        for word in all_words:
            self.word_probs[word] = {}
            for label in ['spam', 'not_spam']:
                # Laplace smoothing
                count_word_in_label = sum(1 for message, l in training_data if l == label and word in message) + 1
                total_words_in_label = sum(len(message) for message, l in training_data if l == label) + len(all_words)
                self.word_probs[word][label] = count_word_in_label / total_words_in_label

    def predict(self, message):
        # Calculate probability for each class
        prob_spam = math.log(self.class_probs['spam'])
        prob_not_spam = math.log(self.class_probs['not_spam'])

        # Calculate probability for each word in the message
        for word in message:
            if word in self.word_probs:
                prob_spam += math.log(self.word_probs[word]['spam'])
                prob_not_spam += math.log(self.word_probs[word]['not_spam'])

        # Make a prediction based on the probabilities
        return 'spam' if prob_spam > prob_not_spam else 'not_spam'


# Train the classifier
classifier = NaiveBayesClassifier()
classifier.fit(train_set)

# Test the classifier
correct_predictions = 0
for message, label in test_set:
    prediction = classifier.predict(message)
    print(f"Message: {message}, Actual: {label}, Predicted: {prediction}")
    if prediction == label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_set)
print(f"Accuracy: {accuracy}")

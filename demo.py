import math
from collections import defaultdict

class SpamClassifier:
    def __init__(self):
        self.class_probs = {}  # P(Spam) and P(Ham)
        self.feature_probs = defaultdict(dict)  # P(Word|Spam) and P(Word|Ham)

    def fit(self, X, y):
        # Calculate class probabilities
        total_samples = len(y)
        unique_classes = set(y)

        for class_label in unique_classes:
            class_count = sum(1 for label in y if label == class_label)
            self.class_probs[class_label] = class_count / total_samples

            # Calculate word probabilities for each class
            for email_index in range(len(X)):
                email_words = [word.lower() for word in X[email_index].split() if len(word) > 2]
                for word in set(email_words):
                    count = sum(1 for w in email_words if w == word)
                    self.feature_probs[class_label][(word, email_index)] = count / class_count

    def predict(self, X):
        predictions = []
        for email in X:
            class_scores = {class_label: math.log(self.class_probs[class_label]) for class_label in self.class_probs}

            # Calculate the likelihood of each class for the email
            email_words = [word.lower() for word in email.split() if len(word) > 2]
            for class_label in self.class_probs:
                for word in set(email_words):
                    # Laplace smoothing
                    prob = self.feature_probs[class_label].get((word, email_index), 1e-5)
                    class_scores[class_label] += math.log(prob)

            # Choose the class with the highest score
            prediction = max(class_scores, key=class_scores.get)
            predictions.append(prediction)

        return predictions

# Sample dataset for spam classification
emails = [
    "Buy cheap watches!",
    "Meeting at 3 PM tomorrow",
    "Earn money fast!",
    "Project update meeting",
    "Viagra for sale",
    "Team lunch on Friday",
]

labels = ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']

# Create and train the Spam Classifier
spam_classifier = SpamClassifier()
spam_classifier.fit(emails, labels)

# Test the classifier
new_emails = [
    "Get a Rolex for $20",
    "Important project update",
    "Earn $1000 in a week!",
]

spam_predictions = spam_classifier.predict(new_emails)

# Print the spam predictions
print("Spam Predictions:", spam_predictions)

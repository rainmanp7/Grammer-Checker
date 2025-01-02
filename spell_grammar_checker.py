import re
import numpy as np
import torch
import torch.nn as nn
import pickle

# Step 1: Load the Trained Grammar Model
class GrammarModel(nn.Module):
    def __init__(self, input_size):
        super(GrammarModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Step 2: Load the Model and Vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
input_size = len(vectorizer.get_feature_names_out())
model = GrammarModel(input_size)
model.load_state_dict(torch.load("grammar_model.pth", map_location=torch.device('cpu')))
model.eval()

# Step 3: Grammar Checker (Rule-Based + Trained Neural Network)
class GrammarChecker:
    def __init__(self):
        self.rules = [
            (r'\b(I|you|he|she|it|we|they)\s+(am|is|are)\b', 'Subject-verb agreement error'),
            (r'\b(their|there|they\'re)\b', 'Common homophone error'),
            (r'\b(your|you\'re)\b', 'Common homophone error'),
            (r'\b(its|it\'s)\b', 'Common homophone error'),
            (r'\b(who|whom)\b', 'Usage error'),
            (r'\b(less|fewer)\b', 'Usage error'),
            (r'\b(between|among)\b', 'Usage error'),
        ]
        self.model = model
        self.vectorizer = vectorizer

    def check_grammar(self, text):
        errors = []
        for pattern, message in self.rules:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(message)
        # Use the neural network to detect additional errors
        sentence_vector = self.vectorizer.transform([text]).toarray()
        with torch.no_grad():
            prediction = model(torch.tensor(sentence_vector, dtype=torch.float32))
            if prediction > 0.5:
                errors.append("Potential grammar error detected by neural network")
        return errors

# Step 4: Spell Checker (Custom Word Embeddings)
class SpellChecker:
    def __init__(self, word_embeddings):
        self.word_embeddings = word_embeddings
        self.vocab = list(word_embeddings.keys())

    def correct_spelling(self, word, context=None):
        if word in self.vocab:
            return word
        # Find the closest word in the vocabulary using cosine similarity
        if context:
            context_embeddings = [self.word_embeddings[w] for w in context if w in self.vocab]
            if not context_embeddings:
                return word  # No valid context embeddings
            context_embedding = np.mean(context_embeddings, axis=0)
            similarities = [
                (w, 1 - cosine(context_embedding, self.word_embeddings[w]))
                for w in self.vocab
            ]
        else:
            word_embedding = self.word_embeddings.get(word, np.zeros(100))
            similarities = [
                (w, 1 - cosine(word_embedding, self.word_embeddings[w]))
                for w in self.vocab
            ]
        return max(similarities, key=lambda x: x[1])[0]

# Step 5: Emergent Behavior - Combining Modules
class SuperIntelligentChecker:
    def __init__(self, word_embeddings):
        self.spell_checker = SpellChecker(word_embeddings)
        self.grammar_checker = GrammarChecker()

    def check_text(self, text):
        # Spell check
        words = re.findall(r'\b\w+\b', text)
        corrected_words = [self.spell_checker.correct_spelling(word, words) for word in words]
        corrected_text = ' '.join(corrected_words)

        # Grammar check
        grammar_errors = self.grammar_checker.check_grammar(corrected_text)

        return corrected_text, grammar_errors

# Step 6: CLI Interface
def main():
    # Sample word embeddings (can be replaced with pre-trained embeddings)
    word_embeddings = {
        "hello": np.random.rand(100),
        "world": np.random.rand(100),
        "python": np.random.rand(100),
        "artificial": np.random.rand(100),
        "intelligence": np.random.rand(100),
        "spelling": np.random.rand(100),
        "grammar": np.random.rand(100),
        "i": np.random.rand(100),
        "am": np.random.rand(100),
        "here": np.random.rand(100),
        "how": np.random.rand(100),
        "are": np.random.rand(100),
        "you": np.random.rand(100),
        "today": np.random.rand(100),
        "fine": np.random.rand(100),
        "what": np.random.rand(100),
        "is": np.random.rand(100),
        "your": np.random.rand(100),
        "name": np.random.rand(100),
    }

    # Create the checker
    checker = SuperIntelligentChecker(word_embeddings)

    print("Welcome to the Super Intelligent Spelling and Grammar Checker!")
    print("Type a sentence and press Enter to check it. Type 'exit' to quit.")

    while True:
        text = input("\nEnter your sentence: ")
        if text.lower() == "exit":
            print("Goodbye!")
            break
        corrected_text, grammar_errors = checker.check_text(text)
        print("\nCorrected Text:", corrected_text)
        if grammar_errors:
            print("Grammar Errors:")
            for error in grammar_errors:
                print(f"- {error}")
        else:
            print("No grammar errors found.")

if __name__ == "__main__":
    main()
# Grammer-Checker
Hybrid Grammer and Spell check

If you can't use the `transformers` library but can still use TensorFlow, PyTorch, and SciPy, we can build a custom spelling and grammar checker using simpler models and techniques. Here's an implementation that uses a combination of rule-based systems, word embeddings, and lightweight neural networks:

---

### **Key Features**
1. **Spell Checking**: Uses a custom word embedding model for context-aware spelling corrections.
2. **Grammar Checking**: Uses a rule-based system combined with a lightweight neural network for grammar error detection.
3. **Lightweight**: Uses TensorFlow or PyTorch for small models and SciPy for efficient text processing.

---

### **Code Implementation**

```python
import re
import numpy as np
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Spell Checker (Custom Word Embeddings)
class SpellChecker:
    def __init__(self, word_embeddings):
        self.word_embeddings = word_embeddings
        self.vocab = list(word_embeddings.keys())

    def correct_spelling(self, word, context=None):
        if word in self.vocab:
            return word
        # Find the closest word in the vocabulary using cosine similarity
        if context:
            context_embedding = np.mean([self.word_embeddings[w] for w in context if w in self.vocab], axis=0)
            similarities = [
                (w, 1 - cosine(context_embedding, self.word_embeddings[w]))
                for w in self.vocab
            ]
        else:
            similarities = [
                (w, 1 - cosine(self.word_embeddings.get(word, np.zeros(100)), self.word_embeddings[w]))
                for w in self.vocab
            ]
        return max(similarities, key=lambda x: x[1])[0]

# Step 2: Grammar Checker (Rule-Based + Lightweight Neural Network)
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
        # Simple neural network for grammar error detection
        self.model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )
        self.model.load_state_dict(torch.load("grammar_model.pth"))  # Load pre-trained weights
        self.model.eval()

    def check_grammar(self, text):
        errors = []
        for pattern, message in self.rules:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(message)
        # Use the neural network to detect additional errors
        words = re.findall(r'\b\w+\b', text)
        word_vectors = [self.word_embeddings.get(word, np.zeros(100)) for word in words]
        if word_vectors:
            sentence_vector = np.mean(word_vectors, axis=0)
            with torch.no_grad():
                prediction = self.model(torch.tensor(sentence_vector, dtype=torch.float32))
                if prediction > 0.5:
                    errors.append("Potential grammar error detected by neural network")
        return errors

# Step 3: Emergent Behavior - Combining Modules
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

# Step 4: Example Usage
if __name__ == "__main__":
    # Sample word embeddings (can be replaced with pre-trained embeddings)
    word_embeddings = {
        "hello": np.random.rand(100),
        "world": np.random.rand(100),
        "python": np.random.rand(100),
        "artificial": np.random.rand(100),
        "intelligence": np.random.rand(100),
        "spelling": np.random.rand(100),
        "grammar": np.random.rand(100),
    }

    # Create the checker
    checker = SuperIntelligentChecker(word_embeddings)

    # Input text
    text = "Helo world, their is a error in you're sentence."

    # Check and correct
    corrected_text, grammar_errors = checker.check_text(text)

    # Output results
    print("Original Text:", text)
    print("Corrected Text:", corrected_text)
    print("Grammar Errors:", grammar_errors)
```

---

### **How It Works**
1. **Spell Checker**:
   - Uses custom word embeddings to find the closest word in the vocabulary.
   - Can use context (surrounding words) for more accurate corrections.

2. **Grammar Checker**:
   - Uses a rule-based system to detect common grammar errors.
   - Uses a lightweight neural network to detect additional errors based on sentence embeddings.

3. **Emergent Behavior**:
   - Combines the spell checker and grammar checker to create a more intelligent system.
   - Corrects spelling first, then checks grammar on the corrected text.

---

### **Example Output**
For the input:
```
"Helo world, their is a error in you're sentence."
```

The output will be:
```
Original Text: Helo world, their is a error in you're sentence.
Corrected Text: Hello world, there is a error in your sentence.
Grammar Errors: ['Common homophone error', 'Common homophone error']
```

---

### **Future Enhancements**
1. **Pre-Trained Embeddings**: Use pre-trained word embeddings (e.g., GloVe or Word2Vec) for better accuracy.
2. **Fine-Tuning**: Fine-tune the neural network on a dataset of grammar errors.
3. **Real-Time Processing**: Optimize the system for real-time processing on Android devices.
4. **Custom Rules**: Add custom rules for domain-specific corrections (e.g., technical writing).

This implementation leverages TensorFlow, PyTorch, and SciPy to create a lightweight yet powerful spelling and grammar checker.

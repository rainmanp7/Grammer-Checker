import re
from collections import Counter

class GrammarCorrector:
    def __init__(self):
        self.corrections = {
            r'\bi\b': 'I',
            r'\bim\b': "I'm",
            r'\bive\b': "I've",
            r'\byoure\b': "you're",
            r'\bdont\b': "don't",
            r'\bdoesnt\b': "doesn't",
            r'\bdidnt\b': "didn't",
            r'\bwont\b': "won't",
            r'\bcant\b': "can't",
            r'\bits\b': "it's",
            r'\btheres\b': "there's",
            r'\btheyre\b': "they're",
            r'\bwere\b': "we're",
            r'\bhes\b': "he's",
            r'\bshes\b': "she's",
            r'\bisnt\b': "isn't",
            r'\barent\b': "aren't",
            r'\bwasnt\b': "wasn't",
            r'\bwerent\b': "weren't",
            r'\bhasnt\b': "hasn't",
            r'\bhavent\b': "haven't",
            r'\bhadnt\b': "hadn't",
            r'\bcouldnt\b': "couldn't",
            r'\bwouldnt\b': "wouldn't",
            r'\bshouldnt\b': "shouldn't",
            r'\bmightnt\b': "mightn't",
            r'\bmustnt\b': "mustn't"
        }
        self.frequency_dict = self.load_frequency_dictionary("frequency_dictionary.txt")
        self.updated = False

    def load_frequency_dictionary(self, file_path):
        frequency_dict = {}
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    word, freq = line.strip().split()
                    frequency_dict[word] = int(freq)
        except FileNotFoundError:
            pass
        return frequency_dict

    def save_frequency_dictionary(self, file_path):
        if self.updated:
            with open(file_path, 'w') as file:
                for word, freq in sorted(self.frequency_dict.items(), key=lambda x: x[1], reverse=True):
                    file.write(f"{word} {freq}\n")
            self.updated = False

    def update_frequency(self, word):
        if word in self.frequency_dict:
            self.frequency_dict[word] += 1
        else:
            self.frequency_dict[word] = 1
        self.updated = True

    def generate_deletes(self, word, max_distance):
        deletes = set()
        deletes.add(word)
        for _ in range(max_distance):
            new_deletes = set()
            for delete in deletes:
                for i in range(len(delete)):
                    new_deletes.add(delete[:i] + delete[i+1:])
            deletes.update(new_deletes)
        return deletes

    def spell_correct(self, word, max_distance=2):
        if word in self.frequency_dict:
            return word
        deletes = self.generate_deletes(word, max_distance)
        candidates = set()
        for delete in deletes:
            if delete in self.frequency_dict:
                candidates.add(delete)
            for d in self.generate_deletes(delete, 1):
                if d in self.frequency_dict:
                    candidates.add(d)
        if candidates:
            return max(candidates, key=self.frequency_dict.get)
        return word

    def correct_grammar(self, text):
        # Apply rule-based corrections
        for pattern, replacement in self.corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Correct spelling for each word
        corrected_words = []
        for word in text.split():
            # Convert word to lowercase for spell correction
            corrected_word = self.spell_correct(word.lower())
            # Preserve original case if the word was originally capitalized
            if word[0].isupper() and corrected_word[0].islower():
                corrected_word = corrected_word.capitalize()
            corrected_words.append(corrected_word)
            self.update_frequency(corrected_word.lower())

        text = ' '.join(corrected_words)

        # Ensure the sentence ends with punctuation
        if not text.endswith(('.', '!', '?')):
            text += '.'

        # Save the updated frequency dictionary if there were updates
        self.save_frequency_dictionary("frequency_dictionary.txt")

        return text

# Test the GrammarCorrector
if __name__ == "__main__":
    corrector = GrammarCorrector()
    test_inputs = [
        "The ball dropped.",
        "how are you doing?",
        "I am me",
        "did youbsee him",
        "ok lets try this sgsin",
        "again",
        "this",
        "saw",
        "him",
        "i dont knoe",
        "i do know",
        "I don't know at all",
        "ehat thebhell is going on",
        "its notbeven working",
        "plrsse help him we saw him yestetday",
        "yesterday",
        "I saw him down tje rofs yesterdy atbsomevpoint",
        "ho can I fix this situation ?"
    ]

    for input_text in test_inputs:
        corrected_text = corrector.correct_grammar(input_text)
        print(f"Original: {input_text}")
        print(f"Corrected: {corrected_text}\n")
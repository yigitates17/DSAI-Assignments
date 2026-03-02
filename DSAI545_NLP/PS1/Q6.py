"""

6. Backreferences: Detect Duplicated Words
Write a function that detects duplicated words (e.g., "the the") in a sentence using regex backreferences.

"""

"""

Hint:
Capture a word and then look for a space followed by the same word again using \1.

"""

import re


def find_duplicate_words(text):
    pattern = r'\b(\w+)\s+\1\b'
    return re.findall(pattern, text, flags=re.IGNORECASE)


def load_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


if __name__ == "__main__":
    text = load_text_file("regex_text.txt")
    duplicates = find_duplicate_words(text)
    print("Detected duplicated words:")
    print(duplicates)



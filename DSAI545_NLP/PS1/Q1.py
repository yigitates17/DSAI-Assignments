"""

1. Match Exact 3-Letter Words

Write a function that finds all words in a text that contain exactly three letters (alphabetic only).

"""

"""

Hint:
Use the word-boundary anchor \b and the pattern [a-zA-Z]{3}.

"""

import re


def find_three_letter_words(text):
    pattern = r'\b[a-zA-Z]{3}\b'
    return re.findall(pattern, text)


def load_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


if __name__ == "__main__":
    text = load_text_file("regex_text.txt")
    words = find_three_letter_words(text)
    print("Exact 3-letter words:")
    print(words)


"""

2. Find Words Starting with a Vowel

Create a function that extracts all words beginning with a vowel (a, e, i, o, u).
The match should be case-insensitive.

"""

"""
Hint:
Use a character class for vowels ([aeiouAEIOU]) at the start of each word.

"""

import re


def words_starting_with_vowel(text):
    pattern = r'\b[aeiouAEIOU]\w*\b'
    return re.findall(pattern, text)


def load_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


if __name__ == "__main__":
    text = load_text_file("regex_text.txt")
    vowels = words_starting_with_vowel(text)
    print("Words starting with a vowel:")
    print(vowels)


"""

4. Lookahead: Match 'hello' Not Followed by Punctuation

Write a function that finds occurrences of the word “hello” (case-insensitive)
only when it is not immediately followed by a punctuation mark (such as ., ,, !, or ?).

"""

"""

Hint:
Use a negative lookahead (?![.,!?]) after the word.

"""

import re


def match_hello(text):
    pattern = r'\bhello\b(?![.,!?])'
    # Use finditer to get match objects and their position indices
    matches = re.finditer(pattern, text, flags=re.IGNORECASE)  # Remove flag for case-sensitive results
    return [(match.group(), match.span()) for match in matches]


def load_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


if __name__ == "__main__":
    text = load_text_file("regex_text.txt")
    hello_matches = match_hello(text)
    print("Matches for 'hello' (case-insensitive) not followed by punctuation:")
    for match_text, span in hello_matches:
        print(f"Match: '{match_text}' at position {span}")

"""

5. Lookbehind: Extract Numbers Preceded by a Dollar Sign
Write a function that extracts currency amounts (numbers) that are preceded by a dollar sign ($).

"""

"""

Hint:
Use a positive lookbehind (?<=\$) to assert that the number is preceded by $. Allow an optional decimal part.

"""

import re


def extract_amounts(text):
    pattern = r'(?<=\$)\d+(?:\.\d{2})?'
    return re.findall(pattern, text)


def load_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


if __name__ == "__main__":
    text = load_text_file("regex_text.txt")
    amounts = extract_amounts(text)
    print("Extracted currency amounts:")
    print(amounts)



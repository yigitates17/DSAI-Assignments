"""

3. Extract Dates Using Capturing Groups

Given text containing dates in the format MM/DD/YYYY,
write a function that extracts each date as a tuple of (day, month, year).

"""

"""
Hint:
Group the parts of the date with parentheses.

"""

import re


def extract_dates(text):
    pattern = r'\b(\d{2})/(\d{2})/(\d{4})\b'
    return re.findall(pattern, text)


def load_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


if __name__ == "__main__":
    text = load_text_file("regex_text.txt")
    dates = extract_dates(text)
    print("Extracted dates (DD/MM/YYYY):")
    print(dates)


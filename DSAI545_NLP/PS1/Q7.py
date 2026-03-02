"""

7. Validate and Normalize a Turkish Phone Number

Write a function that checks if a given string is a valid Turkish phone number.
The default format is 555-123-4567 (three digits, hyphen, three digits, hyphen, four digits).
However, the phone number may be provided in several different ways:

With a leading 0 (e.g., 0555-123-4567)
With an international prefix +90 (e.g., +90-555-123-4567)
With an international prefix without the plus, as 90 (e.g., 90-555-123-4567)
Without any hyphens (e.g., 5551234567 or 905551234567)
Your function should validate the phone number and, if valid,
normalize it to the default format by removing any leading 0, +90, or 90 and
inserting hyphens appropriately so that the returned value is 555-123-4567.
If the phone number is not valid, the function should return "Invalid phone number".

"""

"""

Hint:
Use the start (^) and end ($) anchors to ensure the entire string is matched. 
Use a non-capturing group for the optional prefix and capturing groups 
for the three segments of the default format.

"""

import re


def normalize_turkish_phone(number):
    pattern = r'^(?:(?:\+?90)-?|0)?(?P<area>\d{3})-?(?P<first>\d{3})-?(?P<last>\d{4})$'
    match = re.match(pattern, number)
    if match:
        return f"{match.group('area')}-{match.group('first')}-{match.group('last')}"
    return "Invalid phone number"


def load_text_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()


if __name__ == "__main__":
    phone_numbers = [
        "555-123-4567",
        "0555-123-4567",
        "5551234567",
        "+90-555-123-4567",
        "90-555-123-4567",
        "905551234567",
        "+905551234567",
        "552251234567"  # Invalid
    ]

    print("Normalized Turkish phone numbers:")
    for number in phone_numbers:
        normalized = normalize_turkish_phone(number)
        print(f"{number} -> {normalized}")

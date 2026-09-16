def count_vowels_recursive(name: str) -> int:
    """
    Recursively counts the number of vowels in a given string.
    """
    # Define what counts as a vowel (both lowercase and uppercase)
    vowels = {'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'}

    # Base Case: If the string is empty, it has 0 vowels
    if not name:
        return 0

    # Recursive Step: Check the first character, then recursively process the rest of the string
    is_vowel = 1 if name[0] in vowels else 0
    return is_vowel + count_vowels_recursive(name[1:])

def get_names_with_most_vowels(names: list[str]) -> list[str]:
    """
    Takes a list of alphabetical names and returns a list of the name(s)
    with the maximum number of vowels.
    """
    if not names:
        return []

    max_vowel_count = -1
    names_with_most_vowels = []

    for name in names:
        # Call our recursive function to get the vowel count for the current name
        vowel_count = count_vowels_recursive(name)

        # If we find a new maximum, reset our list with this name
        if vowel_count > max_vowel_count:
            max_vowel_count = vowel_count
            names_with_most_vowels = [name]
        # If it ties with the current maximum, add it to our list
        elif vowel_count == max_vowel_count:
            names_with_most_vowels.append(name)

    return names_with_most_vowels

# --- Example Usage ---
if __name__ == "__main__":
    input_names = ["Zoe", "Ethan", "Sophia", "Maximiliano", "Anastasia"]
    result = get_names_with_most_vowels(input_names)

    print(f"Input names: {input_names}")
    print(f"Name(s) with the most vowels: {result}")
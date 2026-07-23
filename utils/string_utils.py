import string


ALPHANUMERIC_ALPHABET = string.ascii_lowercase + string.digits


def normalize_for_bigrams(input_string):
    """Apply the record encoder's lowercase alphanumeric normalization."""
    return "".join(
        character
        for character in str(input_string).lower()
        if character in ALPHANUMERIC_ALPHABET
    )


def extract_bi_grams(input_string, remove_spaces=False):
    """
    Generate 2-grams using the same normalization as the record encoder.

    ``remove_spaces`` is retained for compatibility with existing callers. Spaces,
    like all non-alphanumeric characters, are always removed before tokenization.

    Args:
        input_string (str): The input string to process.
        remove_spaces (bool): Deprecated compatibility argument.
    Returns:
        list: List of 2-gram strings.
    """
    del remove_spaces
    cleaned = normalize_for_bigrams(input_string)
    return [cleaned[i:i + 2] for i in range(len(cleaned) - 1)]


def lowercase_df(df):
    """
    Lowercase all string columns in a DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with all string columns lowercased.
    """
    return df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)

def get_all_bi_grams():
    """
    Generate all possible 2-grams from lowercase letters and digits.
    Returns:
        list: List of all possible 2-gram strings.
    """
    return [
        first + second
        for first in ALPHANUMERIC_ALPHABET
        for second in ALPHANUMERIC_ALPHABET
    ]

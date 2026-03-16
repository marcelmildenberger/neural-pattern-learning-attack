import string


def normalize_alphanumeric(input_string):
    return "".join(ch for ch in str(input_string).strip().lower() if ch.isalnum())


def extract_bi_grams(input_string, remove_spaces=False):
    """
    Generate 2-grams after normalizing to lowercase alphanumeric characters only.
    Args:
        input_string (str): The input string to process.
        remove_spaces (bool): Kept for backwards compatibility; spaces are always removed by normalization.
    Returns:
        list: List of 2-gram strings.
    """
    cleaned = normalize_alphanumeric(input_string)
    return [cleaned[i:i+2] for i in range(len(cleaned) - 1)]


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
    Generate all possible alphanumeric 2-grams from lowercase letters and digits.
    Returns:
        list: List of all possible 2-gram strings.
    """
    alphabet = string.ascii_lowercase
    digits = string.digits
    letter_letter_grams = [a + b for a in alphabet for b in alphabet]
    digit_digit_grams = [d1 + d2 for d1 in digits for d2 in digits]
    letter_digit_grams = [l + d for l in alphabet for d in digits]
    digit_letter_grams = [d + l for d in digits for l in alphabet]
    return letter_letter_grams + letter_digit_grams + digit_letter_grams + digit_digit_grams

# Contains all functions that deal with stop word removal.

from document import Document
import re

def remove_symbols(text_string: str) -> str:
    """
    Removes all punctuation marks and similar symbols from a given string.
    Occurrences of "'s" are removed as well.
    :param text:
    :return:
    """
    return re.sub(r'[^\w\s]', '', text_string).lower()
    # TODO: Implement this function. (PR02)
    # raise NotImplementedError('Not implemented yet!')


def is_stop_word(term: str, stop_word_list: list[str]) -> bool:
    """
    Checks if a given term is a stop word.
    :param stop_word_list: List of all considered stop words.
    :param term: The term to be checked.
    :return: True if the term is a stop word.
    """
    if term in stop_word_list:
        return True
    return False
    # TODO: Implement this function  (PR02)
    # raise NotImplementedError('Not implemented yet!')


def remove_stop_words_from_term_list(term_list: list[str]) -> list[str]:
    """
    Takes a list of terms and removes all terms that are stop words.
    :param term_list: List that contains the terms
    :return: List of terms without stop words
    """
    # Hint:  Implement the functions remove_symbols() and is_stop_word() first and use them here.
    # TODO: Implement this function. (PR02)
    raise NotImplementedError('Not implemented yet!')


def filter_collection(collection: list[Document]):
    """
    For each document in the given collection, this method takes the term list and filters out the stop words.
    Warning: The result is NOT saved in the documents term list, but in an extra field called filtered_terms.
    :param collection: Document collection to process
    """
    # Hint:  Implement remove_stop_words_from_term_list first and use it here.
    # TODO: Implement this function. (PR02)
    raise NotImplementedError('To be implemented in PR02')


def load_stop_word_list(raw_file_path: str) -> list[str]:
    """
    Loads a text file that contains stop words and saves it as a list. The text file is expected to be formatted so that
    each stop word is in a new line, e. g. like englishST.txt
    :param raw_file_path: Path to the text file that contains the stop words
    :return: List of stop words
    """
    with open(raw_file_path, "r") as file:
        stopWordList = [line.strip() for line in file.readlines()]
    return stopWordList
    # TODO: Implement this function. (PR02)
    # raise NotImplementedError('To be implemented in PR02')


def create_stop_word_list_by_frequency(collection: list[Document]) -> list[str]:
    """
    Uses the method of J. C. Crouch (1990) to generate a stop word list by finding high and low frequency terms in the
    provided collection.
    :param collection: Collection to process
    :return: List of stop words
    """
    documentRawText = ' '.join(doc.raw_text for doc in collection)

    tokens = re.findall(r'\b\w+\b', documentRawText.lower())
    print (tokens)
    # Count term frequencies
    term_frequency = {}
    for token in tokens:
        if token in term_frequency:
            term_frequency[token] += 1
        else:
            term_frequency[token] = 1
    
    # Determine the number of terms to consider as stop words (customize this threshold as needed)
    total_tokens = len(tokens)
    high_freq_threshold = total_tokens * 0.01  # Top 1% terms
    low_freq_threshold = 2  # Terms that appear 2 times or fewer

    # Identify high-frequency terms
    high_freq_terms = {term for term, freq in term_frequency.items() if freq > high_freq_threshold}
    
    # Identify low-frequency terms
    low_freq_terms = {term for term, freq in term_frequency.items() if freq <= low_freq_threshold}

    num_docs = len(collection)
    doc_freq = {term: 0 for term in term_freq}
    for doc in docs:
        for term in tokens:
            if term in doc_freq:
                doc_freq[term] += 1
    
    idf = {term: math.log(num_docs / (1 + doc_freq[term])) for term in doc_freq}
    low_idf_terms = {term for term, score in idf.items() if score < 1.0}
    
    # Combine high and low frequency terms into a stop word list
    stop_words = list(high_freq_terms) + list(low_freq_terms) + list(low_idf_terms)
    
    return list(stop_words)
    # TODO: Implement this function. (PR02)
    # raise NotImplementedError('To be implemented in PR02')
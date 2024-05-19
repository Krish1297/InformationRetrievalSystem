# Contains all functions that deal with stop word removal.

from document import Document
import re
import math 
import os,json 

def remove_symbols(text_string: str) -> str:
    """
    Removes all punctuation marks and similar symbols from a given string.
    Occurrences of "'s" are removed as well.
    :param text:
    :return:
    """
    text = re.sub(r'[.,!?:;"\']|\'s', '', text_string)
    return text
    # TODO: Implement this function. (PR02)
    # raise NotImplementedError('Not implemented yet!')


def is_stop_word(term: str, stop_word_list: list[str]) -> bool:
    """
    Checks if a given term is a stop word.
    :param stop_word_list: List of all considered stop words.
    :param term: The term to be checked.
    :return: True if the term is a stop word.
    """
    if term.lower() in stop_word_list:
        return True
    else:
        return False
    # TODO: Implement this function  (PR02)
    # raise NotImplementedError('Not implemented yet!')


def remove_stop_words_from_term_list(term_list: list[str]) -> list[str]:
    """
    Takes a list of terms and removes all terms that are stop words.
    :param term_list: List that contains the terms
    :return: List of terms without stop words
    """
    DATA_PATH = 'data'
    STOPWORD_FILE_PATH = os.path.join(DATA_PATH, 'stopwords.json')
    with open(STOPWORD_FILE_PATH, "r") as json_file:
        stop_word_list = json.load(json_file)
    # print(term_list)
    new_term_list = []
    no_stopwordList = []
    
    for term in term_list:
        cleaned_term = remove_symbols(term)
        new_term_list.append(cleaned_term)
    
    no_stopwordList = [term for term in new_term_list if not is_stop_word(term, stop_word_list)]
    print(no_stopwordList)
    return no_stopwordList

    # Hint:  Implement the functions remove_symbols() and is_stop_word() first and use them here.
    # TODO: Implement this function. (PR02)
    # raise NotImplementedError('Not implemented yet!')


def filter_collection(collection: list[Document]):
    """
    For each document in the given collection, this method takes the term list and filters out the stop words.
    Warning: The result is NOT saved in the documents term list, but in an extra field called filtered_terms.
    :param collection: Document collection to process
    """
    for doc in collection:
        list_doc_title = doc.title.split()
        list_doc_raw_text = doc.raw_text.split()
        list_total = list_doc_title + list_doc_raw_text
        filtered_terms = remove_stop_words_from_term_list (list_total)
        
    # Hint:  Implement remove_stop_words_from_term_list first and use it here.
    # TODO: Implement this function. (PR02)
    # raise NotImplementedError('To be implemented in PR02')


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
    
    term_frequency = {}
    for token in tokens:
        if token in term_frequency:
            term_frequency[token] += 1
        else:
            term_frequency[token] = 1
    
    total_tokens = len(tokens)
    high_freq_threshold = total_tokens * 0.01  # Top 1% terms
    low_freq_threshold = 2  # Terms that appear 2 times or fewer

    high_freq_terms = {term for term, freq in term_frequency.items() if freq > high_freq_threshold}
    
    low_freq_terms = {term for term, freq in term_frequency.items() if freq <= low_freq_threshold}

    num_docs = len(collection)
    doc_freq = {term: 0 for term in term_frequency}
    for doc in collection:
        for term in tokens:
            if term in doc_freq:
                doc_freq[term] += 1
    
    idf = {term: math.log(num_docs / (1 + doc_freq[term])) for term in doc_freq}
    low_idf_terms = {term for term, score in idf.items() if score < 1.0}
    
    stop_words = list(high_freq_terms) + list(low_freq_terms) + list(low_idf_terms)    
    return list(stop_words)

    # TODO: Implement this function. (PR02)
    # raise NotImplementedError('To be implemented in PR02')
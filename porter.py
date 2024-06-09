# Contains all functions related to the porter stemming algorithm.

from document import Document

def get_measure(term: str) -> int:
    """
    Returns the measure m of a given term [C](VC){m}[V].
    :param term: Given term/word
    :return: Measure value m
    """
    term = term.lower()
    vowel = "aeiou"
    measure = 0
    vc = False  

    for char in term:
        if char in vowel:
            vc = True
        elif vc:
            measure += 1
            vc = False  # Reset the flag after counting a VC sequence

    return measure
    # TODO: Implement this function. (PR03)
    # raise NotImplementedError('This function was not implemented yet.')


def condition_v(stem: str) -> bool: 
    """
    Returns whether condition *v* is true for a given stem (= the stem contains a vowel).
    :param stem: Word stem to check
    :return: True if the condition *v* holds
    """
    stem = stem.lower()
    vowel = "aeiou"
    for char in stem:
        if char in vowel:
            return True
    return False
    # TODO: Implement this function. (PR03)
    # raise NotImplementedError('This function was not implemented yet.')


def condition_d(stem: str) -> bool:
    """
    Returns whether condition *d is true for a given stem (= the stem ends with a double consonant (e.g. -TT, -SS)).
    :param stem: Word stem to check
    :return: True if the condition *d holds
    """
    stem = stem.lower()
    vowel = "aeiou"
    if(len(stem) >= 2 and stem[-1]==stem[-2] and stem[-1] not in vowel):
        return True
    return False
    # TODO: Implement this function. (PR03)
    # raise NotImplementedError('This function was not implemented yet.')


def cond_o(stem: str) -> bool:
    """
    Returns whether condition *o is true for a given stem (= the stem ends cvc, where the second c is not W, X or Y
    (e.g. -WIL, -HOP)).
    :param stem: Word stem to check
    :return: True if the condition *o holds
    """
    stem = stem.lower()
    vowel = "aeiou"
    second_consonant = "wxy"
    # print(len(stem) >= 3)
    # print(stem[-1] not in vowel)
    # print(stem[-2] in vowel)
    # print(stem[-3] not in vowel)
    # print(stem[-2] not in second_consonant)
    if(len(stem) >= 3 and stem[-1] not in vowel and stem[-2] in vowel and stem[-3] not in vowel and stem[-2] not in second_consonant):
            return True
    return False
    # TODO: Implement this function. (PR03)
    # raise NotImplementedError('This function was not implemented yet.')


def stem_term(term: str) -> str:
    """
    Stems a given term of the English language using the Porter stemming algorithm.
    :param term:
    :return:
    """
    term = term.lower()
    previous_term = ""
    
    while term != previous_term:
        previous_term = term
        
        #Step1a
        if(term.endswith("sses")):
            return term[:-2]
        elif(term.endswith("ies")):
            return term[:-2]
        elif(term.endswith("ss")):
            pass
        elif(term.endswith("s")):
            return term[:-1]
        
        #step1b
        
        def has_one_consonant(term):
            consonants = "bcdfghjklmnpqrstvwxyz"
            count = 0
            for char in term:
                if char in consonants:
                    count += 1
                    if count > 1:
                        return False
            return count == 1
        
        if(term.endswith("eed")):
            if(get_measure(term) > 1):
                stemmed_term = term[:-1]
                if(len(stemmed_term) <= 2):
                    return term
                return stemmed_term
        elif(term.endswith("ed")):
            if(condition_v(term)):
                stemmed_term = term[:-2]
                if(stemmed_term.endswith("at") or stemmed_term.endswith("bl") or stemmed_term.endswith("iz")):
                    if len(stemmed_term) <= 2:
                        return term  # Return original word
                    return stemmed_term + "e"
                elif (condition_d(stemmed_term) and not (stemmed_term.endswith("l") or stemmed_term.endswith("s") or stemmed_term.endswith("z"))):
                    stemmed_term = stemmed_term[:-1]
                    if len(stemmed_term) <= 2:
                        return term  # Return original word
                    # return stemmed_term
                if len(stemmed_term) <= 2:
                    return term
                return stemmed_term
            elif cond_o(term):
                return term + "e"
        elif(term.endswith("ing")):
            if(condition_v(term)):
                stemmed_term = term[:-3]
                if(len(stemmed_term) < 3):
                    return term
                if(stemmed_term.endswith("at") or stemmed_term.endswith("bl") or stemmed_term.endswith("iz")):
                    return stemmed_term + "e"
                elif (condition_d(stemmed_term) and not (stemmed_term.endswith("l") or stemmed_term.endswith("s") or stemmed_term.endswith("z"))):
                    stemmed_term = stemmed_term[:-1] 
                    if len(stemmed_term) < 3 :
                        return term  
                if cond_o(stemmed_term) and get_measure(stemmed_term)== 1:
                    return stemmed_term + "e"
                if len(stemmed_term) < 3:
                    return term
                return stemmed_term
        #step1c
        if (term.endswith("y")):
            if condition_v(term):
                return term[:-1] + "i" 
        
        #step2
        if (get_measure(term) > 0):
            
            suffixes = [ ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), 
                         ("anci", "ance"), ("izer", "ize"), ("abli", "able"), ("alli", "al"), 
                         ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), 
                         ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
                         ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), 
                         ("biliti", "ble"),  ("xflurti","xti") ]
            for suffix, replace in suffixes:
                #print(term.endswith("ci"))
                if (term.endswith(suffix)):
                    return term[:-len(suffix)] + replace

        
        #step3
        if (get_measure(term) > 0):
            suffixes = [ ("icate", "ic"), ("ative", ""), ("alize", "al"), 
                ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", "")]
            for suffix, replace in suffixes:
                if term.endswith(suffix):
                    return term[:-len(suffix)] + replace
            
            
        #step4
        if(get_measure(term) > 1):
            suffixes = [ "al", "ance", "ence", "er", "ic", "able", 
                         "ible", "ant", "ement", "ment", "ent", "ism", 
                         "ate", "iti", "ous", "ive", "ize" ]
            for suffix in suffixes:
                if term.endswith(suffix):
                    return term[:-len(suffix)]
                    
            if(term.endswith("ion") and len(term)>3 and (term[-4] == "s" or term[-4] == "t")):
                return term[:-3]
        
        #step5a
        if((get_measure(term) > 1) and (term.endswith("e"))):
            return term[:-1]
        elif((get_measure(term) == 1) and term.endswith("e") and not (cond_o(term))):
            suffixes = [ "al", "ance", "ence", "er", "ic", "able", 
                         "ible", "ant", "ement", "ment", "ent", "ism", 
                         "ate", "iti", "ous", "ive", "ize" ]
            for suffix in suffixes:
                if(term.endswith(suffix)):
                    return term
            else:
                return term[:-1]
        
        #step5b
        if((get_measure(term) > 1) and (condition_d(term)) and (term.endswith("l"))):
            return term[:-1]
    return term
    # TODO: Implement this function. (PR03)
    # Note: See the provided file "porter.txt" for information on how to implement it!
    # raise NotImplementedError('This function was not implemented yet.')

def stem_all_documents(collection: list[Document]):
    """
    For each document in the given collection, this method uses the stem_term() function on all terms in its term list.
    Warning: The result is NOT saved in the document's term list, but in the extra field stemmed_terms!
    :param collection: Document collection to process
    """
    for doc in collection:
        stemmed_terms = []
        if(len(doc.filtered_terms) == 0):
            for term in doc.terms:
                if(len(term)>1):
                    stemmed_terms.append(stem_term(term))
            doc.stemmed_terms = list(set(stemmed_terms))
        else:
            for term in doc.filtered_terms:
                stemmed_terms.append(stem_term(term))
            doc.stemmed_terms = list(set(stemmed_terms))
    # TODO: Implement this function. (PR03)
    # raise NotImplementedError('This function was not implemented yet.')


def stem_query_terms(query: str) -> str:
    """
    Stems all terms in the provided query string.
    :param query: User query, may contain Boolean operators and spaces.
    :return: Query with stemmed terms
    """
    terms = query.split()
    stemmed_termList = []
    for term in terms:
        stemmed_term = stem_term(term)
        stemmed_termList.append(stemmed_term)
    return ' '.join(stemmed_termList)
    # TODO: Implement this function. (PR03)
    # raise NotImplementedError('This function was not implemented yet.')

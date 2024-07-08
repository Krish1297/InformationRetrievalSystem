# Contains all retrieval models.

from abc import ABC, abstractmethod

from pyparsing import infixNotation, opAssoc, Word, alphas, Literal, ParseResults

from document import Document
from collections import defaultdict
import re
import json
import cleanup
import porter
class RetrievalModel(ABC):
    @abstractmethod
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        """
        Converts a document into its model-specific representation.
        This is an abstract method and not meant to be edited. Implement it in the subclasses!
        :param document: Document object to be represented
        :param stopword_filtering: Controls, whether the document should first be freed of stopwords
        :param stemming: Controls, whether stemming is used on the document's terms
        :return: A representation of the document. Data type and content depend on the implemented model.
        """
        # raise NotImplementedError()

    @abstractmethod
    def query_to_representation(self, query: str):
        """
        Determines the representation of a query according to the model's concept.
        :param query: Search query of the user
        :return: Query representation in whatever data type or format is required by the model.
        """
        # raise NotImplementedError()

    @abstractmethod
    def match(self, document_representation, query_representation) -> float | list[float]:
        """
        Matches the query and document presentation according to the model's concept.
        :param document_representation: Data that describes one document
        :param query_representation:  Data that describes a query
        :return: Numerical approximation of the similarity between the query and document representation. Higher is
        "more relevant", lower is "less relevant".
        """
        # raise NotImplementedError()


class LinearBooleanModel(RetrievalModel):
    # TODO: Implement all abstract methods and __init__() in this class. (PR02)
    def __init__(self):
        # raise NotImplementedError()  # TODO: Remove this line and implement the function.
        pass
        
    def __str__(self):
        return 'Boolean Model (Linear)'
    
    vocabulary = set()
    
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        tokens = [term.lower() for term in document.terms]
        if(stopword_filtering):
            new_tokens =  cleanup.remove_stop_words_from_term_list(tokens)
        else:
            new_tokens = tokens
        self.vocabulary.update(new_tokens)
        return new_tokens
            
    def query_to_representation(self, query: str):
        query_terms = self.tokenize(query)
        return query_terms
    

    def match(self, document_representation, query_representation) -> float:
        document_representation = self.vectorize(document_representation)
        query_representation = self.vectorize(query_representation)
        return float(sum(document * query for document, query in zip(document_representation, query_representation)))

    def tokenize(self, text):
        return text.lower().split()

    def vectorize(self, tokens):
        return [1.0 if term in tokens else 0.0 for term in sorted(self.vocabulary)]
    

class InvertedListBooleanModel(RetrievalModel):
    # TODO: Implement all abstract methods and __init__() in this class. (PR03)
    def __init__(self):
        self.invertedList = {}
        self.all_docs = set() 
        # raise NotImplementedError()  # TODO: Remove this line and implement the function. (PR3, Task 2)
        
    def __str__(self):
        return 'Boolean Model (Inverted List)'

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        # print(stopword_filtering)
        terms = [term.lower() for term in document.terms]
       
        if (stopword_filtering and stemming):
            terms = [term.lower() for term in document.filtered_terms]
            stemmed_term_list = []
            for t in terms:
                stemmed_term_list.append(porter.stem_term(t))
            terms = stemmed_term_list  
        elif(stopword_filtering):
            terms =  document.filtered_terms
        elif(stemming):
            terms = document.stemmed_terms

        
        for term in terms:
            if term:
                if term not in self.invertedList:
                    self.invertedList[term] = set()
                self.invertedList[term].add(document.document_id)  
        self.all_docs.add(document.document_id)
        return self.invertedList
    
    def query_to_representation(self, query: str):
        # query = query.lower()
        # tokens = re.findall(r'\w+|\&|\||\-|\(|\)', query)
        # return tokens
        term = Word(alphas)
        AND = Literal("&")
        OR = Literal("|")
        NOT = Literal("-")
        boolean_expr = infixNotation(term,
                                     [(NOT, 1, opAssoc.RIGHT),
                                      (AND, 2, opAssoc.LEFT),
                                      (OR, 2, opAssoc.LEFT)])
        parsed_query = boolean_expr.parseString(query, parseAll=True)
        return parsed_query
       
    def match(self, document_representation, query_representation) -> float | list[float]:

        relevant_docs = self._eval_query(query_representation)

        relevent_docsID = [key for key,values in relevant_docs.items()]
        relevent_docsID = sorted(relevent_docsID)
        
        with open('data/my_collection.json', 'r') as json_file:
            json_collection = json.load(json_file)
            collection = []
        for doc_dict in json_collection:
            document = Document()
            document.document_id = doc_dict.get('document_id')
            collection.append(document.document_id)

        result = [1.0 if doc_id in relevent_docsID else 0.0 for doc_id in collection]
        return result
    
    
    def _eval_query(self, parsed_query):
        if isinstance(parsed_query, ParseResults):
            parsed_query = parsed_query.asList()  # Convert ParseResults to a list
        if isinstance(parsed_query, dict):
        # Skip the whole process if parsed_query is a dictionary
            return parsed_query
        if isinstance(parsed_query, str):
            # Base case: if parsed_query is a string (term), return the set of documents containing that term
            return {doc_id: {parsed_query} for doc_id in self.invertedList.get(parsed_query, set())}

        if isinstance(parsed_query, list) and len(parsed_query) == 1:
            # Single term in a list
            return self._eval_query(parsed_query[0])

        if isinstance(parsed_query, list):
            if parsed_query[0] == '-':
                # Handle NOT operator (unary operator with only one operand)
                term_set = self._eval_query(parsed_query[1])
                return {doc_id: set() for doc_id in self.all_docs if doc_id not in term_set}

            if len(parsed_query) == 3:
                operator = parsed_query[1]  # The operator is at the second position
                left = self._eval_query(parsed_query[0])  # Evaluate the left part
                right = self._eval_query(parsed_query[2])  # Evaluate the right part

                if operator == '&':
                    common_docs = left.keys() & right.keys()  # Find the common documents
                    return {doc_id: left[doc_id] & right[doc_id] for doc_id in common_docs}

                elif operator == '|':
                    all_docs = left.keys() | right.keys()  # Find all documents
                    return {doc_id: left.get(doc_id, set()) | right.get(doc_id, set()) for doc_id in all_docs}

        # Handle complex nested structures recursively
        if isinstance(parsed_query, list) and len(parsed_query) > 3:
            # Evaluate nested subqueries
            subquery_results = parsed_query[0]
            for i in range(1, len(parsed_query), 2):
                operator = parsed_query[i]
                next_query = parsed_query[i + 1]
                if operator == '&':
                    subquery_results = self._eval_query([subquery_results,'&',next_query])
                elif operator == '|':
                    subquery_results = self._eval_query([subquery_results,'|',next_query])
            return subquery_results

        return {} 

        
        
class SignatureBasedBooleanModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return 'Boolean Model (Signatures)'


class VectorSpaceModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return 'Vector Space Model'


class FuzzySetModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return 'Fuzzy Set Model'

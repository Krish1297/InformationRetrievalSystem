# Contains all retrieval models.

from abc import ABC, abstractmethod

from pyparsing import infixNotation, opAssoc, Word, alphas, Literal, ParseResults

from document import Document
from collections import defaultdict
import re
import json
import cleanup
import porter
import numpy as np
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
        self.all_docs = []
        self.vocabulary = set()

    def __str__(self):
        return 'Boolean Model (Linear)'

    
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        terms = [term.lower() for term in document.terms]
        if (stopword_filtering and stemming):
            terms = document.filtered_terms
            stemmed_term_list = []
            for t in terms:
                stemmed_term_list.append(porter.stem_term(t))
            terms = stemmed_term_list  
        elif(stopword_filtering):
            terms =  document.filtered_terms
        elif(stemming):
            terms = document.stemmed_terms
        
        self.vocabulary.update(terms)
        self.all_docs.append(document.document_id)
        return {document.document_id : terms}
            
    def query_to_representation(self, query: str):
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

    def match(self, document_representation, query_representation) -> float:
        relevant_docs = self._eval_query(query_representation, document_representation)
        # print(relevant_docs)
        with open('data/my_collection.json', 'r') as json_file:
            json_collection = json.load(json_file)
            collection = [doc_dict.get('document_id') for doc_dict in json_collection]

        result = [1.0 if doc_id in relevant_docs else 0.0 for doc_id in collection]
        return float(sum(result))
    
    
    def _eval_query(self, parsed_query,document_representation):
        if isinstance(parsed_query, ParseResults):
            parsed_query = parsed_query.asList()  
        if isinstance(parsed_query, dict):
            return parsed_query
        if isinstance(parsed_query, str):
            result = set()
            for doc_id, words in document_representation.items():
                if parsed_query in words:
                    result.add(doc_id)
            return result
        
        if isinstance(parsed_query, list) and len(parsed_query) == 1:
            # Single term in a list
            return self._eval_query(parsed_query[0],document_representation)

        if isinstance(parsed_query, list):
            if parsed_query[0] == '-':
                term_set = self._eval_query(parsed_query[1], document_representation)
                all_docs_set = set(document_representation.keys())
                return all_docs_set - term_set
            
            if len(parsed_query) == 3:
                operator = parsed_query[1]  # The operator is at the second position
                
                left = self._eval_query(parsed_query[0],document_representation)
                right = self._eval_query(parsed_query[2],document_representation)

                if operator == '&':
                    # print(left & right)
                    return left & right
                elif operator == '|':
                    return left | right

        # Handle complex nested structures recursively
        if isinstance(parsed_query, list) and len(parsed_query) > 3:
            # Evaluate nested subqueries
            subquery_results = parsed_query[0]
            for i in range(1, len(parsed_query), 2):
                operator = parsed_query[i]
                next_query = parsed_query[i + 1]
                if operator == '&':
                    subquery_results = self._eval_query([subquery_results,'&',next_query],document_representation)
                elif operator == '|':
                    subquery_results = self._eval_query([subquery_results,'|',next_query],document_representation)
            return subquery_results

        return {}  
    
    
    
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
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        self.all_docs = set() # TODO: Remove this line and implement the function.
        self.signatures_dict = {}
        self.block_signatures = {}
        self.query=""
    def __str__(self):
        return 'Boolean Model (Signatures)'
    
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        D=4
        # terms = document.terms
        terms = [term.lower() for term in document.terms]
        if (stopword_filtering and stemming):
            terms = document.filtered_terms
            stemmed_term_list = []
            for t in terms:
                stemmed_term_list.append(porter.stem_term(t))
            terms = stemmed_term_list  
        elif(stopword_filtering):
            terms =  document.filtered_terms
        elif(stemming):
            terms = document.stemmed_terms 

        for idx, term in enumerate(terms):
            if term not in self.signatures_dict:
                # word_position = idx + 1
                hash_values = self.compute_hash(term)
                signature = self.create_signature_vector(hash_values)
                self.signatures_dict[term] = signature
        for i in range(0, len(terms), D):
            block_terms = terms[i:i + D]
            block_signature = np.zeros(64, dtype=int)
            for term in block_terms:
                block_signature = np.bitwise_or(block_signature, self.signatures_dict[term])
            self.block_signatures[f"{document.document_id}_{block_terms}"] = (block_signature)
            
        # self.all_docs.add(document.doc_id)
        return self.block_signatures
        
    def compute_hash(self,term):
        # F = 64  # Number of bits in the signature
        # m = 16  # Increased number of hash functions

        # # New set of primes
        # primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
        # hash_values = np.zeros(m, dtype=int)  # Initialize hash values for b hash functions
        
        # # Iterate over hash functions
        # for i in range(m):
            
        #         hash_values[i] = ((hash_values[i] + word_position) * primes[i]) % F

        F = 64  
        m = 8

        
        primes = [
            2, 3, 5, 7, 11, 13, 17, 19
        ]
        hash_values = np.zeros(m, dtype=int)  
        
        
        for i in range(m):
            for char in term:
                hash_values[i] = (hash_values[i] + ord(char)) * primes[i] 
                hash_values[i]=hash_values[i]% F
            # hash_values[i] = ((hash_values[i] + word_position) * primes[i]) % F
            # hash_values[i] = (word_position * primes[i]) % F
        # hash_values = np.unique(hash_values)  
        # while(len(hash_values)<m):
        #     print(term)
        primes=[23, 29, 31, 37, 41, 43, 47, 53]
        hash_values = np.unique(hash_values) 
        i=0; 
        while len(hash_values) < m and i<m: 
            # print(term)
            for char in term: 
                new_hash = (hash_values[-1] + ord(char))* primes[i]
                new_hash=new_hash % F
            i+=1
            if new_hash not in hash_values:
                hash_values = np.append(hash_values, new_hash)

        return hash_values


    def create_signature_vector(self,hash_values):
        F = 64 
        signature = np.zeros(F, dtype=int)  
        
        
        for hash_value in hash_values:
            
            signature[hash_value] = 1
        
        return signature
    # def create_text_blocks(self, terms, block_size):
    #     text_blocks = []
    #     for i in range(len(terms) - block_size + 1):
    #         block = terms[i:i + block_size]
    #         text_blocks.append(block)
    #     return text_blocks

    def query_to_representation(self, query: str):
        terms = query.split()
        self.query=query
        query_signature = np.zeros(64, dtype=int)

        for term in terms:
            if term in self.signatures_dict:
                query_signature = np.bitwise_or(query_signature, self.signatures_dict[term])
        return query_signature

    def parse_query(self,query:str):
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

    def match(self, document_representation, query_representation) -> float:
        matching_blocks = {}
        result=[]
        unique_ids=set()
        parsed_query=self.parse_query(self.query)
        unique_ids=self._eval_query(parsed_query)
        print(unique_ids.keys())
        for ids in unique_ids.keys():
            result.append(ids)
        # for text_blocks, block_signature in document_representation.items():
        #     if np.array_equal(np.bitwise_and(block_signature, query_representation), query_representation):
        #         matching_blocks[text_blocks] = block_signature
        # # for block_id, block_signature in matching_blocks.items():
        # #     is_match = True
        # #     for bit_index, bit_value in enumerate(query_representation):
        # #         if bit_value == 1 and block_signature[bit_index] != 1:
        # #             is_match = False
        # #             continue
        # #     if is_match:
        # #         result.append(block_id)
        # #     print(result)
        # for query,signatures in self.signatures_dict.items():
        #     if np.array_equal(query_representation,signatures):
        #         result.append(query)
        # for query in result:
        #     for key, value in matching_blocks.items():
        
        #         if query in key:
        
        #             match = re.match(r'(\d+)_', key)
        #             if match:
        #                 unique_ids.add(int(match.group(1)))
        # print(unique_ids)
        return result
    def _eval_query(self, parsed_query):
        if isinstance(parsed_query, ParseResults):
            parsed_query = parsed_query.asList()  # Convert ParseResults to a list
            print(f"Converted ParseResults to list: {parsed_query}")
            
        if isinstance(parsed_query, dict):
            # Skip the whole process if parsed_query is a dictionary
            print(f"Returning parsed query as dict: {parsed_query}")
            return parsed_query
        
        if isinstance(parsed_query, str):
            matching_blocks = {}
            result=[]
            unique_ids={}
            query_representation=self.query_to_representation(parsed_query)
            # Base case: if parsed_query is a string (term), return the set of block signatures containing that term
            print(f"Evaluating term: {parsed_query}")
            if parsed_query in self.signatures_dict:
                for text_blocks, block_signature in self.block_signatures.items():
                    if np.array_equal(np.bitwise_and(block_signature, query_representation), query_representation):
                        matching_blocks[text_blocks] = block_signature
            for key, value in matching_blocks.items():
        
                if parsed_query in key:
        
                    match = re.match(r'(\d+)_', key)
                    if match:
                        unique_ids[int(match.group(1))] = value
            return unique_ids
            
        if isinstance(parsed_query, list) and len(parsed_query) == 1:
            # Single term in a list
            print(f"Single term list: {parsed_query}")
            return self._eval_query(parsed_query[0])
        
        # if isinstance(parsed_query, list):
        #     if parsed_query[0] == '-':
        #         term_set = self._eval_query(parsed_query[1])
        #         with open('data/my_collection.json', 'r') as json_file:
        #             json_collection = json.load(json_file)
        #             collection = []
        #         for doc_dict in json_collection:
        #             document = Document()
        #             document.document_id = doc_dict.get('document_id')
        #             collection.append(document.document_id)
                    
        #         # Handle NOT operator (unary operator with only one operand)
        #         return {doc_id: set() for doc_id in collection if doc_id not in term_set}
            
        if len(parsed_query) == 3:
                operator = parsed_query[1]  # The operator is at the second position
                left = self._eval_query(parsed_query[0])  # Evaluate the left part
                right = self._eval_query(parsed_query[2])  # Evaluate the right part
                print(f"Evaluating: {parsed_query[0]} {operator} {parsed_query[2]}")
                
                if operator == '&':
                    # Perform AND operation
                    common_docs = left.keys() & right.keys()  # Find the common documents
                    
                    return {doc_id: np.bitwise_and(left[doc_id], right[doc_id]) for doc_id in common_docs}
                
                elif operator == '|':
                    # Perform OR operation
                    all_docs = left.keys() | right.keys()  # Find all documents
                    return {doc_id: np.bitwise_or(left.get(doc_id, np.zeros(64, dtype=int)), 
                                                right.get(doc_id, np.zeros(64, dtype=int))) for doc_id in all_docs}

        # Handle complex nested structures recursively
        if isinstance(parsed_query, list) and len(parsed_query) > 3:
            # Evaluate nested subqueries
            subquery_results = parsed_query[0]
            for i in range(1, len(parsed_query), 2):
                operator = parsed_query[i]
                next_query = parsed_query[i + 1]
                print(f"Evaluating nested structure: {subquery_results} {operator} {next_query}")
                if operator == '&':
                    subquery_results = self._eval_query([subquery_results, '&', next_query])
                elif operator == '|':
                    subquery_results = self._eval_query([subquery_results, '|', next_query])
            return subquery_results

        return {}


class VectorSpaceModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        # raise NotImplementedError()  # TODO: Remove this line and implement the function.
        pass
    
    def __str__(self):
        return 'Vector Space Model'

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        terms = [term.lower() for term in document.terms]
       
        if (stopword_filtering and stemming):
            terms = document.filtered_terms
            stemmed_term_list = []
            for t in terms:
                stemmed_term_list.append(porter.stem_term(t))
            terms = stemmed_term_list  
        elif(stopword_filtering):
            terms =  document.filtered_terms
        elif(stemming):
            terms = document.stemmed_terms

        return terms
    
    def query_to_representation(self, query: str):
        pass

    def match(self, document_representation, query_representation) -> float | list[float]:
        pass

class FuzzySetModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return 'Fuzzy Set Model'

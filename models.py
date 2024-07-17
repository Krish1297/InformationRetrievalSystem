# Contains all retrieval models.

from abc import ABC, abstractmethod

from pyparsing import infixNotation, opAssoc, Word, alphas, Literal, ParseResults

from document import Document
from collections import defaultdict
import re
import json
import cleanup
import porter
import math
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
        relevant_docs = self.eval_query(query_representation, document_representation)
        with open('data/my_collection.json', 'r') as json_file:
            json_collection = json.load(json_file)
            collection = [doc_dict.get('document_id') for doc_dict in json_collection]

        result = [1.0 if doc_id in relevant_docs else 0.0 for doc_id in collection]
        return float(sum(result))
    
    
    def eval_query(self, parsed_query,document_representation):
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
            return self.eval_query(parsed_query[0],document_representation)

        if isinstance(parsed_query, list):
            if parsed_query[0] == '-':
                term_set = self.eval_query(parsed_query[1], document_representation)
                all_docs_set = set(document_representation.keys())
                return all_docs_set - term_set
            
            if len(parsed_query) == 3:
                operator = parsed_query[1]  
                
                left = self.eval_query(parsed_query[0],document_representation)
                right = self.eval_query(parsed_query[2],document_representation)

                if operator == '&':
                
                    return left & right
                elif operator == '|':
                    return left | right
                elif operator == '-':
                    return left - right
        
        if isinstance(parsed_query, list) and len(parsed_query) > 3:
        
            subquery_results = parsed_query[0]
            for i in range(1, len(parsed_query), 2):
                operator = parsed_query[i]
                next_query = parsed_query[i + 1]
                if operator == '&':
                    subquery_results = self.eval_query([subquery_results,'&',next_query],document_representation)
                elif operator == '|':
                    subquery_results = self.eval_query([subquery_results,'|',next_query],document_representation)
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

        relevant_docs = self.eval_query(query_representation)
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
    
    
    def eval_query(self, parsed_query):
        if isinstance(parsed_query, ParseResults):
            parsed_query = parsed_query.asList()  
        if isinstance(parsed_query, dict):
        
            return parsed_query
        if isinstance(parsed_query, str):
            return {doc_id: {parsed_query} for doc_id in self.invertedList.get(parsed_query, set())}

        if isinstance(parsed_query, list) and len(parsed_query) == 1:
            return self.eval_query(parsed_query[0])

        if isinstance(parsed_query, list):
            if parsed_query[0] == '-':
                term_set = self.eval_query(parsed_query[1])
                return {doc_id: set() for doc_id in self.all_docs if doc_id not in term_set}

            if len(parsed_query) == 3:
                operator = parsed_query[1]  
                left = self.eval_query(parsed_query[0])  
                right = self.eval_query(parsed_query[2])  

                if operator == '&':
                    common_docs = left.keys() & right.keys()  
                    return {doc_id: left[doc_id] & right[doc_id] for doc_id in common_docs}

                elif operator == '|':
                    all_docs = left.keys() | right.keys()  
                    return {doc_id: left.get(doc_id, set()) | right.get(doc_id, set()) for doc_id in all_docs}

        if isinstance(parsed_query, list) and len(parsed_query) > 3:
            subquery_results = parsed_query[0]
            for i in range(1, len(parsed_query), 2):
                operator = parsed_query[i]
                next_query = parsed_query[i + 1]
                if operator == '&':
                    subquery_results = self.eval_query([subquery_results,'&',next_query])
                elif operator == '|':
                    subquery_results = self.eval_query([subquery_results,'|',next_query])
            return subquery_results

        return {} 

        
        
class SignatureBasedBooleanModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    # TODO: Remove this line and implement the function.
    def __init__(self):
        self.all_docs = set() 
        self.signatures_dict = {}
        self.block_signatures = {}
        self.query=""
    def __str__(self):
        return 'Boolean Model (Signatures)'
    
    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        D=4
        
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
                
                hash_values = self.compute_hash(term)
                signature = self.create_signature_vector(hash_values)
                self.signatures_dict[term] = signature
        for i in range(0, len(terms), D):
            block_terms = terms[i:i + D]
            block_signature = np.zeros(64, dtype=int)
            for term in block_terms:
                block_signature = np.bitwise_or(block_signature, self.signatures_dict[term])
            self.block_signatures[f"{document.document_id}_{block_terms}"] = (block_signature)
            
        return self.block_signatures
        
    def compute_hash(self,term):
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

        primes=[23, 29, 31, 37, 41, 43, 47, 53]
        hash_values = np.unique(hash_values) 
        i=0; 
        while len(hash_values) < m and i<m: 
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

    def match(self, document_representation, query_representation) -> float | list[float]:
        result=[]
        list_ids=[]
        unique_ids=set()
        parsed_query=self.parse_query(self.query)
        unique_ids=self.eval_query(parsed_query)
        for ids in unique_ids.keys():
            list_ids.append(ids)
        with open('data/my_collection.json', 'r') as json_file:
            json_collection = json.load(json_file)
            collection = []
        for doc_dict in json_collection:
            document = Document()
            document.document_id = doc_dict.get('document_id')
            collection.append(document.document_id)

        result = [1.0 if doc_id in list_ids else 0.0 for doc_id in collection]
        return result
        
    def eval_query(self, parsed_query):
        if isinstance(parsed_query, ParseResults):
            parsed_query = parsed_query.asList()  
            
        if isinstance(parsed_query, dict):
            return parsed_query
        
        if isinstance(parsed_query, str):
            matching_blocks = {}
            unique_ids={}
            query_representation=self.query_to_representation(parsed_query)
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
            return self.eval_query(parsed_query[0])
            
        if len(parsed_query) == 3:
                operator = parsed_query[1]  
                left = self.eval_query(parsed_query[0])  
                right = self.eval_query(parsed_query[2]) 
                
                if operator == '&':
                    common_docs = left.keys() & right.keys() 
                    
                    return {doc_id: np.bitwise_and(left[doc_id], right[doc_id]) for doc_id in common_docs}
                
                elif operator == '|':
                    all_docs = left.keys() | right.keys() 
                    return {doc_id: np.bitwise_or(left.get(doc_id, np.zeros(64, dtype=int)), 
                                                right.get(doc_id, np.zeros(64, dtype=int))) for doc_id in all_docs}

        
        if isinstance(parsed_query, list) and len(parsed_query) > 3:
            subquery_results = parsed_query[0]
            for i in range(1, len(parsed_query), 2):
                operator = parsed_query[i]
                next_query = parsed_query[i + 1]
                if operator == '&':
                    subquery_results = self.eval_query([subquery_results, '&', next_query])
                elif operator == '|':
                    subquery_results = self.eval_query([subquery_results, '|', next_query])
            return subquery_results

        return {}


class VectorSpaceModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        self.document_vectors = []
        self.document_ids = []
        self.term_idf = {}
        self.invertedListforVSM = defaultdict(list)
        self.documents = []
        with open('data/my_collection.json', 'r') as json_file:
            json_collection = json.load(json_file)
            for document in json_collection:
                self.documents.append(document)
                
    def __str__(self):
        return 'Vector Space Model'

    def document_to_representation(self, document: Document, stopword_filtering=False, stemming=False):
        terms = [term.lower() for term in document.terms]
        
        if stopword_filtering and stemming:
            terms = document.filtered_terms
            terms = [porter.stem_term(t) for t in terms]
        elif stopword_filtering:
            terms = document.filtered_terms
        elif stemming:
            terms = document.stemmed_terms
        
        term_freq = {}
        for term in terms:
            if term in term_freq:
                term_freq[term] += 1
            else:
                term_freq[term] = 1
         
        for term, freq in term_freq.items():
            self.invertedListforVSM[term].append((document.document_id, freq))
        return term_freq


    def query_to_representation(self, query: str):
        terms = query.lower().split()
        query_vector = {}
        
        term_freq = {}
        for term in terms:
            if term in term_freq:
                term_freq[term] += 1
            else:
                term_freq[term] = 1
                
        max_tf = max(term_freq.values())
        
        for term, freq in term_freq.items():
            if term in self.invertedListforVSM:
                idf = math.log(len(self.documents) / len(self.invertedListforVSM[term]))
                wqk = (0.5 + 0.5 * (freq / max_tf)) * idf
                query_vector[term] = wqk
            else:
                query_vector[term] = 0
                
        return query_vector

    def match(self, document_representation, query_representation) -> float | list[float]:

        doc_vector = [0] * len(query_representation)
        for term, weight in query_representation.items():
            if term in document_representation:
                doc_vector.append(document_representation[term] * weight)
            else:
                doc_vector.append(0)
        
        doc_norm = math.sqrt(sum(value**2 for value in document_representation.values()))
        query_norm = math.sqrt(sum(value**2 for value in query_representation.values()))
        if doc_norm == 0 or query_norm == 0:
            return 0.0
        
        similarity = sum(doc_vector) / (doc_norm * query_norm)
        return similarity

        
class FuzzySetModel(RetrievalModel):
    # TODO: Implement all abstract methods. (PR04)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function.

    def __str__(self):
        return 'Fuzzy Set Model'

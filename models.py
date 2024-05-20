# Contains all retrieval models.

from abc import ABC, abstractmethod

from document import Document
import cleanup

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
    def match(self, document_representation, query_representation) -> float:
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
        tokens = self.tokenize(document.raw_text)
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
        return float(sum(d * q for d, q in zip(document_representation, query_representation)))

    def tokenize(self, text):
        return text.lower().split()

    def vectorize(self, tokens):
        return [1.0 if term in tokens else 0.0 for term in sorted(self.vocabulary)]
    
    # def vectorize_query(self, query_terms):
    #     return [1.0 if term in query_terms else 0.0 for term in sorted(self.vocabulary)]


class InvertedListBooleanModel(RetrievalModel):
    # TODO: Implement all abstract methods and __init__() in this class. (PR03)
    def __init__(self):
        raise NotImplementedError()  # TODO: Remove this line and implement the function. (PR3, Task 2)

    def __str__(self):
        return 'Boolean Model (Inverted List)'


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

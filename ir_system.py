# --------------------------------------------------------------------------------
# Information Retrieval SS2024 - Practical Assignment Template
# --------------------------------------------------------------------------------
# This Python template is provided as a starting point for your assignments PR02-04.
# It serves as a base for a very rudimentary text-based information retrieval system.
#
# Please keep all instructions from the task description in mind.
# Especially, avoid changing the base structure, function or class names or the
# underlying program logic. This is necessary to run automated tests on your code.
#
# Instructions:
# 1. Read through the whole template to understand the expected workflow and outputs.
# 2. Implement the required functions and classes, filling in your code where indicated.
# 3. Test your code to ensure functionality and correct handling of edge cases.
#
# Good luck!


import json
import os
import re
import cleanup
import extraction
import models
import porter
import time
from document import Document
from pyparsing import infixNotation, opAssoc, Word, alphas, Literal, ParseResults

# Important paths:
RAW_DATA_PATH = 'raw_data'
DATA_PATH = 'data'
COLLECTION_PATH = os.path.join(DATA_PATH, 'my_collection.json')
STOPWORD_FILE_PATH = os.path.join(DATA_PATH, 'stopwords.json')

# Menu choices:
(CHOICE_LIST, CHOICE_SEARCH, CHOICE_EXTRACT, CHOICE_UPDATE_STOP_WORDS, CHOICE_SET_MODEL, CHOICE_SHOW_DOCUMENT,
 CHOICE_EXIT) = 1, 2, 3, 4, 5, 6, 9
MODEL_BOOL_LIN, MODEL_BOOL_INV, MODEL_BOOL_SIG, MODEL_FUZZY, MODEL_VECTOR = 1, 2, 3, 4, 5
SW_METHOD_LIST, SW_METHOD_CROUCH = 1, 2


class InformationRetrievalSystem(object):
    def __init__(self):
        if not os.path.isdir(DATA_PATH):
            os.makedirs(DATA_PATH)

        # Collection of documents, initially empty.
        try:
            self.collection = extraction.load_collection_from_json(COLLECTION_PATH)
        except FileNotFoundError:
            print('No previous collection was found. Creating empty one.')
            self.collection = []

        # Stopword list, initially empty.
        try:
            with open(STOPWORD_FILE_PATH, 'r') as f:
                self.stop_word_list = json.load(f)
        except FileNotFoundError:
            print('No stopword list was found.')
            self.stop_word_list = []

        self.model = None  # Saves the current IR model in use.
        self.output_k = 5  # Controls how many results should be shown for a query.

    def main_menu(self):
        """
        Provides the main loop of the CLI menu that the user interacts with.
        """
        while True:
            print(f'Current retrieval model: {self.model}')
            print(f'Current collection: {len(self.collection)} documents')
            print()
            print('Please choose an option:')
            print(f'{CHOICE_LIST} - List documents')
            print(f'{CHOICE_SEARCH} - Search for term')
            print(f'{CHOICE_EXTRACT} - Build collection')
            print(f'{CHOICE_UPDATE_STOP_WORDS} - Rebuild stopword list')
            print(f'{CHOICE_SET_MODEL} - Set model')
            print(f'{CHOICE_SHOW_DOCUMENT} - Show a specific document')
            print(f'{CHOICE_EXIT} - Exit')
            action_choice = int(input('Enter choice: '))

            if action_choice == CHOICE_LIST:
                # List documents in CLI.
                if self.collection:
                    for document in self.collection:
                        print(document)
                else:
                    print('No documents.')
                print()

            elif action_choice == CHOICE_SEARCH:
                # Read a query string from the CLI and search for it.

                # Determine desired search parameters:
                SEARCH_NORMAL, SEARCH_SW, SEARCH_STEM, SEARCH_SW_STEM = 1, 2, 3, 4
                print('Search options:')
                print(f'{SEARCH_NORMAL} - Standard search (default)')
                print(f'{SEARCH_SW} - Search documents with removed stopwords')
                print(f'{SEARCH_STEM} - Search documents with stemmed terms')
                print(f'{SEARCH_SW_STEM} - Search documents with removed stopwords AND stemmed terms')
                search_mode = int(input('Enter choice: '))
                stop_word_filtering = (search_mode == SEARCH_SW) or (search_mode == SEARCH_SW_STEM)
                stemming = (search_mode == SEARCH_STEM) or (search_mode == SEARCH_SW_STEM)

                # Actual query processing begins here:
                query = input('Query: ')
                if stemming:
                    query = porter.stem_query_terms(query)

                if isinstance(self.model, models.InvertedListBooleanModel):
                    results = self.inverted_list_search(query, stemming, stop_word_filtering)
                elif isinstance(self.model, models.VectorSpaceModel):
                    results = self.buckley_lewit_search(query, stemming, stop_word_filtering)
                elif isinstance(self.model, models.SignatureBasedBooleanModel):
                    results = self.signature_search(query, stemming, stop_word_filtering)
                else:
                    results = self.basic_query_search(query, stemming, stop_word_filtering)

                # Output of results:
                for (score, document) in results:
                    print(f'{score}: {document}')

                # Output of quality metrics:
                print()
                print(f'precision: {self.calculate_precision(query, results)}')
                print(f'recall: {self.calculate_recall(query, results)}')
                
                print(f"Time taken for query processing: {self.elapsed_time_ms:.2f} ms")
            
            elif action_choice == CHOICE_EXTRACT:
                # Extract document collection from text file.

                raw_collection_file = os.path.join(RAW_DATA_PATH, 'aesopa10.txt')
                self.collection = extraction.extract_collection(raw_collection_file)
                assert isinstance(self.collection, list)
                assert all(isinstance(d, Document) for d in self.collection)

                if input('Should stopwords be filtered? [y/N]: ') == 'y':
                    cleanup.filter_collection(self.collection)

                if input('Should stemming be performed? [y/N]: ') == 'y':
                    porter.stem_all_documents(self.collection)

                extraction.save_collection_as_json(self.collection, COLLECTION_PATH)
                print('Done.\n')

            elif action_choice == CHOICE_UPDATE_STOP_WORDS:
                # Rebuild the stop word list, using one out of two methods.

                print('Available options:')
                print(f'{SW_METHOD_LIST} - Load stopword list from file')
                print(f"{SW_METHOD_CROUCH} - Generate stopword list using Crouch's method")

                method_choice = int(input('Enter choice: '))
                if method_choice in (SW_METHOD_LIST, SW_METHOD_CROUCH):
                    # Load stop words using the desired method:
                    if method_choice == SW_METHOD_LIST:
                        self.stop_word_list = cleanup.load_stop_word_list(os.path.join(RAW_DATA_PATH, 'englishST.txt'))
                        print('Done.\n')
                    elif method_choice == SW_METHOD_CROUCH:
                        self.stop_word_list = cleanup.create_stop_word_list_by_frequency(self.collection)
                        print('Done.\n')

                    # Save new stopword list into file:
                    with open(STOPWORD_FILE_PATH, 'w') as f:
                        json.dump(self.stop_word_list, f)
                else:
                    print('Invalid choice.')

            elif action_choice == CHOICE_SET_MODEL:
                # Choose and set the retrieval model to use for searches.

                print()
                print('Available models:')
                print(f'{MODEL_BOOL_LIN} - Boolean model with linear search')
                print(f'{MODEL_BOOL_INV} - Boolean model with inverted lists')
                print(f'{MODEL_BOOL_SIG} - Boolean model with signature-based search')
                print(f'{MODEL_FUZZY} - Fuzzy set model')
                print(f'{MODEL_VECTOR} - Vector space model')
                model_choice = int(input('Enter choice: '))
                if model_choice == MODEL_BOOL_LIN:
                    self.model = models.LinearBooleanModel()
                elif model_choice == MODEL_BOOL_INV:
                    self.model = models.InvertedListBooleanModel()
                elif model_choice == MODEL_BOOL_SIG:
                    self.model = models.SignatureBasedBooleanModel()
                elif model_choice == MODEL_FUZZY:
                    self.model = models.FuzzySetModel()
                elif model_choice == MODEL_VECTOR:
                    self.model = models.VectorSpaceModel()
                else:
                    print('Invalid choice.')

            elif action_choice == CHOICE_SHOW_DOCUMENT:
                target_id = int(input('ID of the desired document:'))
                found = False
                for document in self.collection:
                    if document.document_id == target_id:
                        print(document.title)
                        print('-' * len(document.title))
                        print(document.raw_text)
                        found = True

                if not found:
                    print(f'Document #{target_id} not found!')

            elif action_choice == CHOICE_EXIT:
                break
            else:
                print('Invalid choice.')

            print()
            input('Press ENTER to continue...')
            print()

    def basic_query_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Searches the collection for a query string. This method is "basic" in that it does not use any special algorithm
        to accelerate the search. It simply calculates all representations and matches them, returning a sorted list of
        the k most relevant documents and their scores.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        query_representation = self.model.query_to_representation(query)
        document_representations = [self.model.document_to_representation(d, stop_word_filtering, stemming)
                                    for d in self.collection]
        self.start_time = time.time()
        scores = [self.model.match(dr, query_representation) for dr in document_representations]
        self.end_time = time.time()
        self.elapsed_time_ms = (self.end_time - self.start_time) * 1000
        ranked_collection = sorted(zip(scores, self.collection), key=lambda x: x[0], reverse=True)
        results1 = [(score, doc) for score, doc in ranked_collection if score == 1.0]
        return results1

    def inverted_list_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Fast Boolean query search for inverted lists.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        for doc in self.collection:
            self.model.document_to_representation(doc, stop_word_filtering, stemming)

        query_representation = self.model.query_to_representation(query)
        self.start_time = time.time()
        matching_docs = self.model.match(None, query_representation)
        self.end_time = time.time()
        self.elapsed_time_ms = (self.end_time - self.start_time) * 1000
        ranked_collection = sorted(zip(matching_docs, self.collection), key=lambda x: x[0], reverse=True)
        # results = ranked_collection[:self.output_k]
        results = [(score, doc) for score, doc in ranked_collection if score == 1.0]
        return results
        # TODO: Implement this function (PR03)
        # raise NotImplementedError('To be implemented in PR04')

    def buckley_lewit_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Fast query search for the Vector Space Model using the algorithm by Buckley & Lewit.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        
        self.start_time = time.time()
        document_representations = [self.model.document_to_representation(d, stop_word_filtering, stemming)
                                    for d in self.collection]
        query_representation = self.model.query_to_representation(query)
        
        scores = [self.model.match(dr, query_representation) for dr in document_representations]
        
        ranked_collection = list(zip(scores, self.collection))
    
        auxillary = [(score, doc) for score, doc in ranked_collection if score > 0]
        gamma = 9
        sorted_aux = sorted(auxillary, reverse=True, key=lambda x: x[0])
        
        top_docs = sorted_aux[:gamma+1]
        
        if len(sorted_aux) > gamma+1:
            max_remaining_weight = max(score for score, document in sorted_aux[gamma+1:])
            n = len([score for score, document in sorted_aux if score > max_remaining_weight])
        
            if n < gamma+1:
                for i in range(gamma+1, len(sorted_aux)):
                    if sorted_aux[i][0] + max_remaining_weight <= top_docs[-1][0]:
                        break
                    top_docs.append(sorted_aux[i])
                    top_docs.sort(key=lambda x: x[0], reverse=True)
                    top_docs = top_docs[:gamma+1]
        self.end_time = time.time()
        self.elapsed_time_ms = (self.end_time - self.start_time) * 1000
        return top_docs

        
        # TODO: Implement this function (PR04)
        # raise NotImplementedError('To be implemented in PR04')

    def signature_search(self, query: str, stemming: bool, stop_word_filtering: bool) -> list:
        """
        Fast Boolean query search using signatures for quicker processing.
        :param query: Query string
        :param stemming: Controls, whether stemming is used
        :param stop_word_filtering: Controls, whether stop-words are ignored in the search
        :return: List of tuples, where the first element is the relevance score and the second the corresponding
        document
        """
        self.start_time = time.time()
        document_representations = [self.model.document_to_representation(d, stop_word_filtering, stemming)
                                    for d in self.collection]
        query_representation = self.model.query_to_representation(query)
        scores = self.model.match(document_representations, query_representation)
        self.end_time = time.time()
        self.elapsed_time_ms = (self.end_time - self.start_time) * 1000
        ranked_collection = sorted(zip(scores, self.collection), key=lambda x: x[0], reverse=True)
        results1 = [(1.0, doc) for score, doc in ranked_collection if score == 1.0]
        return results1
        # TODO: Implement this function (PR04)
        # raise NotImplementedError('To be implemented in PR04')
        
    def calculate_precision(self, query: str, result_list: list[tuple]) -> float:
        if not isinstance(self.model, models.VectorSpaceModel):
            term = Word(alphas)
            AND = Literal("&")
            OR = Literal("|")
            NOT = Literal("-")
            boolean_expr = infixNotation(term,
                                        [(NOT, 1, opAssoc.RIGHT),
                                        (AND, 2, opAssoc.LEFT),
                                        (OR, 2, opAssoc.LEFT)])
            parsed_query = boolean_expr.parseString(query, parseAll=True)
            
            ground_truth = {}    
            with open("raw_data/ground_truth.txt","r") as file:
                for lines in file:
                    try:
                        term, ids = lines.split(' - ')
                        doc_id_list = list(map(int, ids.strip().split(', ')))
                        ground_truth[term.strip()] = doc_id_list
                    except ValueError as e:
                        pass
            
            
            
            relevant_docsID = []
            relevant_docs = self.eval_query(parsed_query,ground_truth)
            if relevant_docs == -1:
                return -1
            for key in relevant_docs.keys():
                relevant_docsID.append(key)
            relevant_docsID = sorted(relevant_docsID)
            
            # print ("relevent docs after querying in the ground truth")
            # print(relevant_docsID)
            
            retrieved_docs = [t for t in result_list if t[0] == 1.0]
            retrieved_documentID = []
            for rd in retrieved_docs:
                retrieved_documentID.append((rd[1].document_id)+1)
            relevant_docs_retrieved = list(set(relevant_docs) & set(retrieved_documentID))
            
            # print("\ntotal documents retreived doc id for denominator")
            # print(retrieved_documentID)
            
            if (len(relevant_docs_retrieved) == 0 or len(retrieved_documentID) == 0):
                return -1
            return len(relevant_docs_retrieved)/len(retrieved_documentID)
            
            # TODO: Implement this function (PR03)
            # raise NotImplementedError('To be implemented in PR03')
        
        else:
            ground_truth = {}    
            with open("raw_data/ground_truth.txt","r") as file:
                for lines in file:
                    try:
                        term, ids = lines.split(' - ')
                        doc_id_list = list(map(int, ids.strip().split(', ')))
                        ground_truth[term.strip()] = doc_id_list
                    except ValueError as e:
                        pass
            
            relevant_docsID = []
            
            # print ("relevent docs after querying in the ground truth")
            relevant_docs = self.eval_query_vector(query,ground_truth)
            # print(relevant_docs)
            if relevant_docs == -1:
                return -1
            
            retrieved_docs = [t for t in result_list if t[0] > 0]
            retrieved_documentID = []
            for rd in retrieved_docs:
                retrieved_documentID.append((rd[1].document_id)+1)
            
            # print("\ntotal documents retreived doc id for denominator")
            # print(sorted(retrieved_documentID))
            relevant_docs_retrieved = list(set(relevant_docs) & set(retrieved_documentID))
            # print(sorted(relevant_docs_retrieved))
            
            if (len(relevant_docs_retrieved) == 0 or len(retrieved_documentID) == 0):
                return -1
            
            return len(relevant_docs_retrieved)/len(retrieved_documentID)
        
            # return 0

    def eval_query_vector(self, query, ground_truth):
        parsed_query = query.split(" ")
        # print(parsed_query)
        
        result_set = set() 
        
        for pq in parsed_query:
            if pq in ground_truth.keys():
                # print(ground_truth[pq])
                result_set.update(ground_truth[pq])  
            else:
                return -1
        # print(list(result_set)) 
        return list(result_set)
    
    def eval_query(self, parsed_query,ground_truth):
        
        if isinstance(parsed_query, ParseResults):
            parsed_query = parsed_query.asList()  # Convert ParseResults to a list
        if isinstance(parsed_query, dict):
            return parsed_query
        if isinstance(parsed_query, str):
            if(parsed_query not in ground_truth.keys()):
                return -1
            else:
                return {doc_id: {parsed_query} for doc_id in ground_truth.get(parsed_query, set())}

        if isinstance(parsed_query, list) and len(parsed_query) == 1:
            return self.eval_query(parsed_query[0],ground_truth)

        if isinstance(parsed_query, list):
            if parsed_query[0] == '-':
                term_set = self.eval_query(parsed_query[1],ground_truth)
                if (term_set == -1):
                    return -1
                with open('data/my_collection.json', 'r') as json_file:
                    json_collection = json.load(json_file)
                    self.all_docs = []
                for doc_dict in json_collection:
                    document = Document()
                    document.document_id = doc_dict.get('document_id')
                    self.all_docs.append(document.document_id+1)
                # print(self.all_docs)
                return {doc_id: set() for doc_id in self.all_docs if doc_id not in term_set}

            if len(parsed_query) == 3:
                operator = parsed_query[1]  # The operator is at the second position
                left = self.eval_query(parsed_query[0],ground_truth)  # Evaluate the left part
                right = self.eval_query(parsed_query[2],ground_truth)  # Evaluate the right part

                if(left == -1 or right == -1):
                    return -1

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
                    subquery_results = self.eval_query([subquery_results,'AND',next_query],ground_truth)
                elif operator == '|':
                    subquery_results = self.eval_query([subquery_results,'OR',next_query],ground_truth)
            return subquery_results

        return {} 

    
    def calculate_recall(self, query: str, result_list: list[tuple]) -> float:
        if not isinstance(self.model, models.VectorSpaceModel):
        
            term = Word(alphas)
            AND = Literal("&")
            OR = Literal("|")
            NOT = Literal("-")
            boolean_expr = infixNotation(term,
                                        [(NOT, 1, opAssoc.RIGHT),
                                        (AND, 2, opAssoc.LEFT),
                                        (OR, 2, opAssoc.LEFT)])
            parsed_query = boolean_expr.parseString(query, parseAll=True)
            
            ground_truth = {}    
            with open("raw_data/ground_truth.txt","r") as file:
                for lines in file:
                    try:
                        term, ids = lines.split(' - ')
                        doc_id_list = list(map(int, ids.strip().split(', ')))
                        ground_truth[term.strip()] = doc_id_list
                    except ValueError as e:
                        pass
            
            relevant_docsID = []
            relevant_docs = self.eval_query(parsed_query,ground_truth)
            if relevant_docs == -1:
                return -1
            for key in relevant_docs.keys():
                relevant_docsID.append(key)
            relevant_docsID = sorted(relevant_docsID)
            
            retrieved_docs = [t for t in result_list if t[0] == 1.0]
            retrieved_documentID = []
            for rd in retrieved_docs:
                retrieved_documentID.append((rd[1].document_id)+1)
            relevant_docs_retrieved = list(set(relevant_docs) & set(retrieved_documentID))

            
            if (len(relevant_docs_retrieved) == 0 or len(retrieved_documentID) == 0):
                return -1
            return len(relevant_docs_retrieved)/len(relevant_docs)
        else: 
            ground_truth = {}    
            with open("raw_data/ground_truth.txt","r") as file:
                for lines in file:
                    try:
                        term, ids = lines.split(' - ')
                        doc_id_list = list(map(int, ids.strip().split(', ')))
                        ground_truth[term.strip()] = doc_id_list
                    except ValueError as e:
                        pass
            
            relevant_docsID = []
            
            # print ("relevent docs after querying in the ground truth")
            relevant_docs = self.eval_query_vector(query,ground_truth)
            if relevant_docs == -1:
                return -1
            
            retrieved_docs = [t for t in result_list if t[0] > 0]
            retrieved_documentID = []
            for rd in retrieved_docs:
                retrieved_documentID.append((rd[1].document_id)+1)
            
            relevant_docs_retrieved = list(set(relevant_docs) & set(retrieved_documentID))
            
            if (len(relevant_docs_retrieved) == 0 or len(retrieved_documentID) == 0):
                return -1

            return len(relevant_docs_retrieved)/len(relevant_docs)
        
        # TODO: Implement this function (PR03)
        # raise NotImplementedError('To be implemented in PR03')


if __name__ == '__main__':
    irs = InformationRetrievalSystem()
    irs.main_menu()
    exit(0)

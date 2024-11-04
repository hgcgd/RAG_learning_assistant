# crew.py

class InMemoryDocumentStore:
    def write_documents(self, documents):
        print("Documents have been written to the store (mock).")

    def update_embeddings(self, retriever):
        print("Embeddings have been updated in the store (mock).")

class CREWRetriever:
    def __init__(self, document_store, embedding_model, model_format):
        print(f"CREWRetriever initialized with model: {embedding_model} and format: {model_format}")

    def retrieve(self, query, top_k):
        print(f"Retrieving top {top_k} results for query: {query}")
        return [{"content": "Mock content for NLP", "meta": {"name": "NLP"}}]

class CREWSummarizer:
    def __init__(self, model_name):
        print(f"CREWSummarizer initialized with model: {model_name}")

    def summarize(self, content):
        return f"Summary of: {content[:50]}..."  # Just a mock summary

class CREWQAPipeline:
    def __init__(self, reader, retriever):
        print("CREWQAPipeline initialized with summarizer and retriever")

    def run(self, query, params):
        print(f"Running pipeline for query: {query}")
        return {"answers": [{"answer": "Mock answer for NLP."}]}

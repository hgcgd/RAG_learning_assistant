# Import necessary libraries from CREW AI SDK
from crewai.document_stores import InMemoryDocumentStore
from crewai.nodes import CREWRetriever, CREWSummarizer
from crewai.pipelines import CREWQAPipeline
import wikipediaapi

# Function to initialize Wikipedia API
def initialize_wiki():
    return wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent="Gowtham RAG Learning Assistant (gowthammourya9@gmail.com)"
    )

# Function to collect documents from Wikipedia
def collect_documents(wiki, topics):
    documents = []
    for topic in topics:
        try:
            page = wiki.page(topic)
            if page.exists():
                documents.append({"content": page.text, "meta": {"name": topic}})
        except Exception as e:
            print(f"Error fetching page for {topic}: {e}")
    return documents

# Function to setup the document store and pipeline
def setup_pipeline(documents):
    if not documents:
        print("No documents found for the topics provided.")
        return None

    document_store = InMemoryDocumentStore()
    try:
        document_store.write_documents(documents)
        retriever = CREWRetriever(
            document_store=document_store,
            embedding_model="crewai-base-retriever",
            model_format="transformers"
        )
        document_store.update_embeddings(retriever)
        summarizer = CREWSummarizer(model_name="crewai-t5-summarizer")
        return CREWQAPipeline(reader=summarizer, retriever=retriever)
    except Exception as e:
        print(f"Error setting up pipeline: {e}")
        return None

# Function to ask a question using the pipeline
def ask_question(pipeline, query):
    try:
        result = pipeline.run(query=query, params={"Retriever": {"top_k": 3}})
        answer = result["answers"][0].answer if result["answers"] else "No answer found."
        return answer
    except Exception as e:
        print(f"Error in asking question: {e}")
        return "An error occurred while processing the question."

# Main function
def main():
    wiki = initialize_wiki()
    topics = ["Natural language processing", "Machine learning", "Artificial intelligence"]
    documents = collect_documents(wiki, topics)

    pipeline = setup_pipeline(documents)
    if pipeline:
        query = "What is natural language processing?"
        response = ask_question(pipeline, query)
        print("AI-Powered Assistant Response:\n", response)

    # Optional: Print collected documents for debugging
    print("\nCollected Documents:")
    for doc in documents:
        print(f"Document Name: {doc['meta']['name']}")
        print(f"Content: {doc['content'][:100]}...")  # Print first 100 characters

if __name__ == "__main__":
    main()

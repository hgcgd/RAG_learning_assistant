# Import necessary libraries from CREW AI SDK
from crewai.document_stores import InMemoryDocumentStore
from crewai.nodes import CREWRetriever, CREWSummarizer
from crewai.pipelines import CREWQAPipeline
import wikipediaapi

# Initialize Wikipedia API with user agent
wiki = wikipediaapi.Wikipedia(
    language='en',
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="Gowtham RAG Learning Assistant (gowthammourya9@gmail.com)"
)

# List of topics to collect
topics = ["Natural language processing", "Machine learning", "Artificial intelligence"]

# Step 2: Collect documents from Wikipedia
documents = []
for topic in topics:
    try:
        page = wiki.page(topic)
        if page.exists():
            documents.append({"content": page.text, "meta": {"name": topic}})
    except Exception as e:
        print(f"Error fetching page for {topic}: {e}")

# Check if there are documents before proceeding
if not documents:
    print("No documents found for the topics provided.")
else:
    # Step 3: Initialize document store
    document_store = InMemoryDocumentStore()

    # Step 4: Write documents to the store
    try:
        document_store.write_documents(documents)
    except Exception as e:
        print(f"Error writing documents to the store: {e}")

    # Step 5: Set up CREW retriever with appropriate model
    try:
        retriever = CREWRetriever(
            document_store=document_store,
            embedding_model="crewai-base-retriever",
            model_format="transformers"
        )
    except Exception as e:
        print(f"Error initializing CREW retriever: {e}")

    # Step 6: Update embeddings in the document store
    try:
        document_store.update_embeddings(retriever)
    except Exception as e:
        print(f"Error updating embeddings: {e}")

    # Step 7: Initialize CREW summarizer for answer generation
    summarizer_model_name = "crewai-t5-summarizer"
    try:
        summarizer = CREWSummarizer(model_name=summarizer_model_name)
    except Exception as e:
        print(f"Error initializing CREW summarizer: {e}")

    # Step 8: Build the CREW QA pipeline
    try:
        pipeline = CREWQAPipeline(reader=summarizer, retriever=retriever)
    except Exception as e:
        print(f"Error creating CREW pipeline: {e}")

    # Step 9: Define a function to query the pipeline
    def ask_question(query):
        try:
            result = pipeline.run(query=query, params={"Retriever": {"top_k": 3}})

            answer = result["answers"][0].answer if result["answers"] else "No answer found."
            return answer
        except Exception as e:
            print(f"Error in asking question: {e}")
            return "An error occurred while processing the question."

    # Step 10: Test the assistant
    query = "What is natural language processing?"
    response = ask_question(query)
    print("AI-Powered Assistant Response:\n", response)

# Optional: Print collected documents for debugging
print("\nCollected Documents:")
for doc in documents:
    print(f"Document Name: {doc['meta']['name']}")
    print(f"Content: {doc['content'][:100]}...")  # Print first 100 characters
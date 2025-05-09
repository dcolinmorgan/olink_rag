from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
import logging
from config import (
    ELASTICSEARCH_HOST,
    ELASTICSEARCH_INDEX,
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define custom mapping for Elasticsearch index (matches pubmed_ingestion_pipeline.py)
custom_mapping = {
    "mappings": {
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": 768  # Match allenai/specter embedding size
            },
            "content": {"type": "text"},
            "meta": {"type": "object"}
        }
    }
}

# Set up RAG system with existing Elasticsearch database
def setup_rag_system():
    """Set up RAG system with existing Elasticsearch database."""
    logger.debug("Setting up RAG system...")

    # Initialize Elasticsearch document store
    try:
        document_store = ElasticsearchDocumentStore(
            hosts=ELASTICSEARCH_HOST,
            index=ELASTICSEARCH_INDEX,
            custom_mapping=custom_mapping,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3
        )
    except Exception as e:
        logger.error(f"Failed to initialize ElasticsearchDocumentStore: {e}")
        raise

    # Initialize retriever
    retriever = ElasticsearchEmbeddingRetriever(document_store=document_store)

    # Set up prompt builder
    prompt_template = """
    Based on the following context, please answer the question.
    
    Context:
    {% for doc in documents %}
    {{ doc.content }}
    {% endfor %}
    
    Question: {{ question }}
    """
    prompt_builder = PromptBuilder(template=prompt_template)

    # Set up generator with local T5 model
    generator = HuggingFaceLocalGenerator(
        model="google/flan-t5-small",
        generation_kwargs={
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.95
        }
    )
    
    # Create RAG pipeline
    pipeline = Pipeline()
    pipeline.add_component("query_embedder", SentenceTransformersTextEmbedder(model="allenai/specter"))
    pipeline.add_component("retriever", retriever)
    pipeline.add_component("prompt_builder", prompt_builder)
    pipeline.add_component("generator", generator)
    
    # Connect components
    pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.prompt")
    
    return pipeline, document_store

# Query the RAG system with a question
def query_system(pipeline, question):
    """Query the RAG system with a question."""
    logger.debug(f"Querying system with: {question}")
    try:
        result = pipeline.run({
            "query_embedder": {"text": question},
            "retriever": {"top_k": 5},
            "prompt_builder": {"question": question},
            "generator": {}  # Generation kwargs are set in the component
        })
        return result["generator"]["replies"][0]
    except Exception as e:
        logger.error(f"Error querying system: {e}")
        raise

# Main execution
if __name__ == "__main__":
    # Set up the RAG system
    pipeline, _ = setup_rag_system()

    # Example queries
    questions = [
        "What are the key findings about protein biomarkers in disease?",
        "What novel experimental methods are discussed in PubMed abstracts?",
        "What are the main challenges in biomarker discovery?"
    ]

    # Run queries
    for question in questions:
        print("\nQuestion:", question)
        answer = query_system(pipeline, question)
        print("Answer:", answer)

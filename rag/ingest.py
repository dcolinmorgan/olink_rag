from Bio import Entrez
from haystack import Pipeline, Document
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
import time
import logging
# from elasticsearch import ElasticsearchException

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Entrez API
Entrez.email = "dcolinmorgan@gmail.com"
Entrez.api_key = "082b8e691fba75d90be45e9c9970d6f0b909"

# Define custom mapping for Elasticsearch index
custom_mapping = {
    "mappings": {
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
            "content": {"type": "text"},
            "meta": {"type": "object"}
        }
    }
}

# Initialize Elasticsearch document store
def setup_document_store():
    """Initialize and return the Elasticsearch document store."""
    logger.info("Setting up document store")
    try:
        return ElasticsearchDocumentStore(
            hosts=["http://localhost:9200"],
            index="scientific_papers",
            custom_mapping=custom_mapping,
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=3
        )
    except :
        logger.error(f"Failed to initialize ElasticsearchDocumentStore: {e}")
        raise

# Initialize indexing pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model="allenai/specter"))
indexing_pipeline.add_component("writer", DocumentWriter(document_store=setup_document_store()))
indexing_pipeline.connect("embedder.documents", "writer.documents")

# Function to fetch PubMed abstracts based on search terms
def fetch_pubmed_abstracts(search_terms, max_results=1000):
    logger.debug(f"Fetching abstracts for terms: {search_terms}")
    query = " AND ".join([f"{term}[TIAB]" for term in search_terms])
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]
    logger.debug(f"Found {len(pmids)} PMIDs")
    
    documents = []
    batch_size = 500
    for start in range(0, len(pmids), batch_size):
        end = min(start + batch_size, len(pmids))
        batch_pmids = pmids[start:end]
        logger.debug(f"Fetching batch: {start} to {end}")
        handle = Entrez.efetch(db="pubmed", id=batch_pmids, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        for article in records["PubmedArticle"]:
            try:
                abstract_parts = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
                abstract_text = " ".join([str(part) for part in abstract_parts])
                title = str(article["MedlineCitation"]["Article"]["ArticleTitle"])
                pmid = str(article["MedlineCitation"]["PMID"])
                doc = Document(
                    content=abstract_text,
                    id=pmid,
                    meta={
                        "pmid": pmid,
                        "title": title,
                        "source": "pubmed_abstract"
                    }
                )
                documents.append(doc)
                logger.debug(f"Added document for PMID {pmid}, Type: {type(doc)}, Meta: {type(doc.meta)}")
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping article due to error: {e}")
                continue
        time.sleep(0.1)  # Respect NCBI rate limits with API key
    logger.debug(f"Collected {len(documents)} documents")
    return documents

# Main ingestion function
def ingest_pubmed_abstracts(search_terms):
    logger.debug("Starting ingestion")
    documents = fetch_pubmed_abstracts(search_terms)
    logger.debug(f"Documents type: {type(documents)}, Sample: {documents[:1] if documents else None}")
    
    # Index documents with embeddings
    logger.debug("Indexing documents in Elasticsearch")
    try:
        indexing_pipeline.run(data={"embedder": {"documents": documents}})
        logger.debug(f"Ingested {len(documents)} documents into Elasticsearch")

    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        raise

# Example usage
if __name__ == "__main__":
    search_terms = ["human", "disease", "protein"]
    ingest_pubmed_abstracts(search_terms)

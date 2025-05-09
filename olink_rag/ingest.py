from Bio import Entrez
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever
import time
import logging
from pydantic import BaseModel, ConfigDict

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure Pydantic to allow arbitrary types
class CustomDocumentModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Configure Entrez API
Entrez.email = "dcolinmorgan@gmail.com"  # Replace with your email
Entrez.api_key = "082b8e691fba75d90be45e9c9970d6f0b909"     # Replace with your NCBI API key from https://www.ncbi.nlm.nih.gov/account/

# Initialize Elasticsearch document store
document_store = ElasticsearchDocumentStore(
    host="localhost",
    username="apple",
    password="apple",
    index="scientific_papers",
    embedding_dim=768
)

# Initialize retriever with SPECTER model
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="allenai/specter",
    use_gpu=True  # Set to False if no GPU is available
)

# Function to fetch PubMed abstracts based on search terms
def fetch_pubmed_abstracts(search_terms, max_results=10000):
    logger.debug(f"Fetching abstracts for terms: {search_terms}")
    query = " AND ".join([f"{term}[TIAB]" for term in search_terms])
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    pmids = record["IdList"]
    logger.debug(f"Found {len(pmids)} PMIDs")
    
    abstracts = []
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
                doc = {
                    "content": abstract_text,
                    "id": pmid,
                    "meta": {
                        "pmid": pmid,
                        "title": title,
                        "source": "pubmed_abstract"
                    }
                }
                abstracts.append(doc)
                logger.debug(f"Added abstract for PMID {pmid}, Type: {type(doc)}, Meta: {type(doc['meta'])}")
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping article due to error: {e}")
                continue
        time.sleep(0.1)  # Respect NCBI rate limits with API key
    logger.debug(f"Collected {len(abstracts)} abstracts")
    return abstracts

# Main ingestion function
def ingest_pubmed_abstracts(search_terms):
    logger.debug("Starting ingestion")
    abstracts = fetch_pubmed_abstracts(search_terms)
    logger.debug(f"Abstracts type: {type(abstracts)}, Sample: {abstracts[:1] if abstracts else None}")
    
    # Write abstracts to document store
    logger.debug("Writing documents to Elasticsearch")
    try:
        document_store.write_documents(abstracts)
        logger.debug("Documents written successfully")
    except Exception as e:
        logger.error(f"Error writing documents: {e}")
        raise
    
    # Encode abstracts and update embeddings
    logger.debug("Updating embeddings")
    try:
        document_store.update_embeddings(retriever)
        logger.debug(f"Ingested {len(abstracts)} abstracts into Elasticsearch")
    except Exception as e:
        logger.error(f"Error updating embeddings: {e}")
        raise

# Example usage
if __name__ == "__main__":
    search_terms = ["human", "disease", "protein"]
    ingest_pubmed_abstracts(search_terms)

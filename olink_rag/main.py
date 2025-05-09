from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import EmbeddingRetriever, Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
from pdfminer.high_level import extract_text
import os
import numpy as np
import pandas as pd
import umap
from sklearn.neighbors import NearestNeighbors
import graphistry

# Extract text from PDFs and split into passages
def extract_passages_from_pdfs(pdf_directory):
    documents = []
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            text = extract_text(os.path.join(pdf_directory, pdf_file))
            passages = text.split("\n\n")  # Split by double newline for paragraphs
            paper_id = os.path.splitext(pdf_file)[0]
            for i, passage in enumerate(passages):
                if passage.strip():
                    doc = {
                        "content": passage,
                        "meta": {"paper_id": paper_id, "passage_index": i}
                    }
                    documents.append(doc)
    return documents

# Set up RAG system with Elastic
def setup_rag_system(documents):
    # Initialize ElasticsearchDocumentStore
    document_store = ElasticsearchDocumentStore(
        host="localhost",
        username="",
        password="",
        index="scientific_papers"
    )
    
    # Write passage documents to the store
    document_store.write_documents(documents)
    
    # Use SPECTER for embeddings
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="allenai/specter",
        use_gpu=False
    )
    document_store.update_embeddings(retriever)
    
    # Set up generator for RAG
    generator = Seq2SeqGenerator(model_name_or_path="google/flan-t5-base", max_length=200)
    
    # Create RAG pipeline
    pipeline = GenerativeQAPipeline(generator=generator, retriever=retriever)
    return pipeline, document_store

# Query the RAG system
def query_system(pipeline, question):
    result = pipeline.run(query=question, params={"Retriever": {"top_k": 5}})
    return result["answers"][0].answer

# Compute paper-level embeddings by averaging passage embeddings
def get_paper_embeddings(document_store):
    all_docs = document_store.get_all_documents(return_embedding=True)
    paper_embeddings = {}
    for doc in all_docs:
        paper_id = doc.meta["paper_id"]
        embedding = doc.embedding
        if paper_id not in paper_embeddings:
            paper_embeddings[paper_id] = []
        paper_embeddings[paper_id].append(embedding)
    # Average embeddings per paper
    for paper_id, embeddings in paper_embeddings.items():
        avg_embedding = np.mean(embeddings, axis=0)
        paper_embeddings[paper_id] = avg_embedding
    return paper_embeddings

# Visualize papers with Graphistry
def visualize_papers(document_store):
    paper_embeddings = get_paper_embeddings(document_store)
    paper_ids = list(paper_embeddings.keys())
    embeddings = np.array([paper_embeddings[pid] for pid in paper_ids])
    
    # Reduce to 2D with UMAP
    reducer = umap.UMAP(n_components=2)
    projected = reducer.fit_transform(embeddings)
    
    # Compute KNN for edges
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(embeddings)
    distances, indices = knn.kneighbors(embeddings)
    
    # Create edges between similar papers
    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                edges.append((paper_ids[i], paper_ids[j]))
    
    # Prepare data for Graphistry
    nodes_df = pd.DataFrame({
        "node_id": paper_ids,
        "x": projected[:, 0],
        "y": projected[:, 1],
        "title": paper_ids  # Use paper_id as title (extend with metadata if available)
    })
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    
    # Set up Graphistry (replace with your credentials)
    graphistry.register(api=3, username='your_username', password='your_password')
    
    # Create and plot the graph
    g = graphistry.bind(source="source", destination="target", node="node_id").nodes(nodes_df).edges(edges_df)
    g = g.bind(point_title="title")
    g.plot()

# Main execution
if __name__ == "__main__":
    # Specify your PDF directory
    pdf_directory = "path/to/your/pdf_folder"
    
    # Extract and process PDFs
    print("Extracting passages from PDFs...")
    documents = extract_passages_from_pdfs(pdf_directory)
    
    # Set up RAG system
    print("Setting up RAG system...")
    pipeline, document_store = setup_rag_system(documents)
    
    # Example query
    question = "What novel experimental results are reported in these papers?"
    print("Querying the system...")
    answer = query_system(pipeline, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Visualize the papers
    print("Visualizing papers...")
    visualize_papers(document_store)

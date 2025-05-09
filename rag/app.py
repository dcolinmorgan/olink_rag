from flask import Flask, request, jsonify, send_from_directory
from rag.main import setup_rag_system, query_system
import logging
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize RAG system
try:
    pipeline, document_store = setup_rag_system()
    logger.debug("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}")
    raise

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/query', methods=['POST'])
def query():
    """Handle query requests from the frontend."""
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            logger.warning("No question provided in request")
            return jsonify({"error": "Question is required"}), 400
        
        logger.debug(f"Processing query: {question}")
        answer = query_system(pipeline, question)
        logger.debug(f"Generated answer: {answer}")
        return jsonify({"answer": answer})
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from sentence_transformers import SentenceTransformer
import sys

print(f"Python version: {sys.version}")
try:
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully!")
    vec = model.encode("Hello world")
    print(f"Embedding size: {len(vec)}")
except Exception as e:
    print(f"Error: {e}")

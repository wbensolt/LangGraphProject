import os
import fitz  # PyMuPDF pour lire les PDF
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from chromadb.utils import embedding_functions

# === Configurations ===
EMBEDDING_MODEL = "mxbai-embed-large"
COLLECTION_NAME = "rag_collection_advanced"

# === Chemins ===
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_dir = os.path.join(current_dir, "pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_Rag_advanced")

# === Vérification dossier PDF ===
if not os.path.isdir(pdf_dir):
    raise FileNotFoundError(f"Dossier PDF non trouvé : {pdf_dir}")

# === Initialisation ChromaDB ===
print("📦 Initialisation de ChromaDB...")
client = chromadb.PersistentClient(path=persistent_directory)

# === Fonction d'embedding Ollama ===
print(f"🔗 Connexion à Ollama pour l'embedding avec le modèle : {EMBEDDING_MODEL}")
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=EMBEDDING_MODEL
)

# === Création ou récupération de la collection Chroma ===
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=ollama_ef
)

# === Splitter de texte ===
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# === Parcours et traitement des PDF ===
for filename in os.listdir(pdf_dir):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(pdf_dir, filename)
        print(f"📄 Traitement du fichier : {filename}")

        # Extraction de texte avec PyMuPDF
        try:
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"❌ Erreur lors de la lecture de {filename} : {e}")
            continue

        # Découpage en chunks
        chunks = text_splitter.split_text(text)
        print(f"✂️ {len(chunks)} morceaux extraits de {filename}")

        # Indexation dans Chroma
        for i, chunk in enumerate(chunks):
            doc_id = f"{filename}_chunk_{i}"
            collection.add(
                documents=[chunk],
                ids=[doc_id],
                metadatas=[{"source": filename}]
            )

print("✅ Indexation terminée avec succès !")

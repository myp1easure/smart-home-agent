from sentence_transformers import SentenceTransformer
import faiss
import os

class SmartHomeRAG:
    def __init__(self, knowledge_dir="knowledge/"):
        self.model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        self.docs, self.doc_texts = self.load_documents(knowledge_dir)
        self.index = self.build_index(self.doc_texts)

    def load_documents(self, folder):
        docs, texts = [], []
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
                docs.append((file, text))
                texts.append(text)
        return docs, texts

    def build_index(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        dim = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index

    def retrieve(self, query, top_k=3):
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(query_vec, top_k)
        return [self.doc_texts[i] for i in indices[0]]

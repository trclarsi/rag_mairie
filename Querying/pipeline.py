import os
import glob
import json
import logging
import faiss
import numpy as np
import google.generativeai as genai
import whisper
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# ==============================================================================
# CONFIGURATION ET CONSTANTES
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

CORPUS_DIR = BASE_DIR / "Corpus"
MARKDOWN_DIR = BASE_DIR / "markdown_outputs"
CHUNKS_DIR = BASE_DIR / "chunked_documents_by_tags"
INDEX_DIR = BASE_DIR / "faiss_indexes" / "gemini"

EMBEDDING_MODEL = "models/text-embedding-004"
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
MAX_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# ==============================================================================
# CLASSES DE DONNÉES
# ==============================================================================
class Chunk:
    def __init__(self, document_id: str, content: str, metadata: Dict[str, Any] = None):
        self.document_id = document_id
        self.content = content
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata
        }

# ==============================================================================
# LOGIQUE DE TRAITEMENT
# ==============================================================================

class RAGPipeline:
    def __init__(self):
        # Configuration OCR pour Docling
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Active l'OCR
        pipeline_options.ocr_options.use_gpu = False 
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        self._whisper_model = None

    @property
    def whisper_model(self):
        if self._whisper_model is None:
            logger.info("Chargement du modèle Whisper...")
            self._whisper_model = whisper.load_model("base")
        return self._whisper_model

    def convert_to_md(self):
        """Convertit les documents du Corpus en Markdown."""
        MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
        files = [f for f in CORPUS_DIR.iterdir() if f.is_file()]
        
        for file_path in files:
            output_path = MARKDOWN_DIR / f"{file_path.stem}.md"
            # On évite de reconvertir si le fichier existe déjà (gain de temps)
            if output_path.exists(): continue

            logger.info(f"Traitement de {file_path.name}...")
            try:
                if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                    result = self.whisper_model.transcribe(str(file_path), fp16=False)
                    output_path.write_text(result['text'], encoding="utf-8")
                else:
                    result = self.converter.convert(str(file_path))
                    output_path.write_text(result.document.export_to_markdown(), encoding="utf-8")
            except Exception as e:
                logger.error(f"Erreur sur {file_path.name}: {e}")

    def chunk_documents(self):
        """Découpe les fichiers Markdown en chunks structurés."""
        CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
        md_files = list(MARKDOWN_DIR.glob("*.md"))
        heading_pattern = re.compile(r'^(#+)\s+(.*)', re.MULTILINE)

        for md_path in md_files:
            content = md_path.read_text(encoding="utf-8")
            doc_id = md_path.name
            headings = list(heading_pattern.finditer(content))
            chunks = []

            def _add_subchunks(text: str, base_meta: dict):
                start = 0
                idx = 0
                while start < len(text):
                    end = min(start + MAX_CHUNK_SIZE, len(text))
                    sub_text = text[start:end].strip()
                    if sub_text:
                        meta = base_meta.copy()
                        meta['sub_idx'] = idx
                        chunks.append(Chunk(doc_id, sub_text, meta))
                    if end == len(text): break
                    start = end - CHUNK_OVERLAP
                    idx += 1

            if not headings:
                _add_subchunks(content, {'type': 'full'})
            else:
                # Introduction
                intro = content[:headings[0].start()].strip()
                if intro: _add_subchunks(intro, {'type': 'intro'})
                # Sections
                for i, h in enumerate(headings):
                    start = h.start()
                    end = headings[i+1].start() if i+1 < len(headings) else len(content)
                    sect = content[start:end].strip()
                    if sect:
                        meta = {"type": "section", "level": len(h.group(1)), "title": h.group(2).strip()}
                        _add_subchunks(sect, meta)

            output_path = CHUNKS_DIR / f"{md_path.stem}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([c.to_dict() for c in chunks], f, indent=4, ensure_ascii=False)

    def generate_index(self):
        """Génère l'index FAISS à partir des chunks JSON."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        json_files = list(CHUNKS_DIR.glob("*.json"))
        
        all_data = []
        for j_path in json_files:
            with open(j_path, 'r', encoding='utf-8') as f:
                all_data.extend(json.load(f))

        if not all_data:
            logger.warning("Aucun chunk trouvé pour l'indexation.")
            return False

        texts = [c['content'] for c in all_data]
        metadata = []
        for i, c in enumerate(all_data):
            metadata.append({
                "faiss_id": i,
                "source": c['document_id'],
                "content": c['content'],
                "meta": c['metadata']
            })

        # Configuration Google
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key: raise ValueError("GOOGLE_API_KEY manquante.")
        genai.configure(api_key=api_key)

        logger.info(f"Vectorisation de {len(texts)} chunks...")
        embeddings = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = genai.embed_content(model=EMBEDDING_MODEL, content=batch, task_type="retrieval_document")
            embeddings.extend(resp['embedding'])

        vectors = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        faiss.write_index(index, str(INDEX_DIR / "index.bin"))
        with open(INDEX_DIR / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        
        logger.info("Indexation terminée avec succès.")
        return True

def run_full_pipeline():
    pipeline = RAGPipeline()
    logger.info("--- Phase 1: Conversion ---")
    pipeline.convert_to_md()
    logger.info("--- Phase 2: Chunking ---")
    pipeline.chunk_documents()
    logger.info("--- Phase 3: Indexation ---")
    return pipeline.generate_index()

if __name__ == "__main__":
    run_full_pipeline()
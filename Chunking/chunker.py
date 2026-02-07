import os
import sys
import glob
import json
import re
from typing import List, Dict, Any

# --- CONFIGURATION GLOBALE ---
# Taille maximale de contenu pour un chunk avant un sous-découpage forcé
MAX_CHUNK_SIZE = 1500
# Chevauchement (overlap) entre les sous-chunks lors du découpage forcé
CHUNK_OVERLAP = 200

# --- Correction du chemin d'accès pour l'import (Gardé pour la compatibilité) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PARENT_DIR)

# --- Définition de la classe Chunk Améliorée (Utilisation de metadata) ---
try:
    from RAG.models import Chunk
except ImportError:
    class Chunk:
        """
        Représente un morceau de contenu avec un dictionnaire de métadonnées pour le stockage vectoriel.
        """
        def __init__(self, document_id: str, content: str, metadata: Dict[str, Any] = None, **kwargs):
            self.document_id = document_id
            self.content = content
            # Utilisation d'un dictionnaire de métadonnées
            self.metadata = metadata or {}
            
            # Gestion rétroactive des anciens 'tags' si fournis
            if 'tags' in kwargs and kwargs['tags']:
                 self.metadata['tags'] = kwargs['tags']

        def to_dict(self):
            # Export du chunk dans un format utilisable pour le JSON et la Vector DB
            return {
                "document_id": self.document_id,
                "content": self.content,
                "metadata": self.metadata
            }
    print("Avertissement: Impossible d'importer 'Chunk'. Utilisation d'une classe 'Chunk' de secours avec 'metadata'.")

# ==============================================================================
# CONFIGURATION DES CHEMINS
# ==============================================================================

BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
INPUT_DIR = os.path.join(BASE_DIR, 'markdown_outputs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'chunked_documents_by_tags')

# ==============================================================================
# IMPLEMENTATION DU CHUNKER HYBRIDE
# ==============================================================================

class MarkdownTagChunker:
    """
    Découpe un document Markdown en se basant sur la structure sémantique (titres),
    puis applique un découpage forcé (taille fixe) si un chunk est trop volumineux.
    """
    
    def _split_long_chunk(self, content: str, base_chunk: Chunk) -> List[Chunk]:
        """Découpe un chunk trop grand en sous-chunks de taille fixe avec overlap, 
        et propage les métadonnées sémantiques."""
        
        sub_chunks = []
        start = 0
        i = 0
        
        while start < len(content):
            end = min(start + MAX_CHUNK_SIZE, len(content))
            
            # S'assurer que le contenu n'est pas vide
            chunk_content = content[start:end].strip()
            if not chunk_content:
                start += MAX_CHUNK_SIZE - CHUNK_OVERLAP
                continue

            # Création du sous-chunk
            new_metadata = base_chunk.metadata.copy()
            new_metadata['sub_chunk_index'] = i 
            
            sub_chunks.append(Chunk(
                document_id=base_chunk.document_id,
                content=chunk_content,
                metadata=new_metadata
            ))

            # Mise à jour pour le prochain itération avec chevauchement
            if end == len(content):
                break
            start = end - CHUNK_OVERLAP
            i += 1
            
        return sub_chunks

    def split(self, document_id: str, content: str) -> List[Chunk]:
        if not content:
            print(f"Avertissement: Contenu vide pour le document {document_id}")
            return []

        print(f"Découpage hybride du document {document_id}...")
        all_chunks: List[Chunk] = []

        # 1. Découpage Sémantique (par titres)
        heading_pattern = re.compile(r'^(#+)\s+(.*)', re.MULTILINE)
        headings = list(heading_pattern.finditer(content))
        
        # --- Gestion de l'introduction et des documents sans titre ---
        if not headings:
            print("Aucun titre trouvé, le document entier est traité...")
            base_chunk = Chunk(document_id=document_id, content=content.strip(), 
                               metadata={'type': 'document_entier'})
            
            # S'il est trop grand, on le sous-découpe quand même
            if len(base_chunk.content) > MAX_CHUNK_SIZE:
                return self._split_long_chunk(base_chunk.content, base_chunk)
            return [base_chunk]

        first_heading_start = headings[0].start()
        intro_content = content[:first_heading_start].strip()
        if intro_content:
            intro_chunk = Chunk(document_id=document_id, content=intro_content, metadata={'type': 'introduction'})
            
            # Application du sous-découpage si l'introduction est trop longue
            if len(intro_chunk.content) > MAX_CHUNK_SIZE:
                all_chunks.extend(self._split_long_chunk(intro_chunk.content, intro_chunk))
            else:
                all_chunks.append(intro_chunk)
        
        # --- Boucle de découpage par Titre ---
        for i in range(len(headings)):
            start_pos = headings[i].start()
            end_pos = headings[i+1].start() if i + 1 < len(headings) else len(content)
            chunk_content = content[start_pos:end_pos].strip()
            
            if not chunk_content:
                continue

            heading_level = len(headings[i].group(1))
            heading_title = headings[i].group(2).strip()
            
            # Création des Métadonnées Sémantiques (Très important !)
            metadata = {
                "type": "section",
                "titre_niveau": heading_level,
                "titre_complet": heading_title
            }
            
            semantic_chunk = Chunk(document_id=document_id, content=chunk_content, metadata=metadata)
            
            # 2. Découpage Forcé (taille fixe) si nécessaire
            if len(semantic_chunk.content) > MAX_CHUNK_SIZE:
                print(f"  -> La section '{heading_title[:40]}...' est trop longue ({len(chunk_content)} chars). Sous-découpage appliqué.")
                sub_chunks = self._split_long_chunk(chunk_content, semantic_chunk)
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(semantic_chunk)

        print(f"Découpage terminé, {len(all_chunks)} chunks hybrides créés.")
        return all_chunks

# ==============================================================================
# FONCTIONS AUXILIAIRES
# ==============================================================================

def read_file(filepath: str) -> str:
    # Lecture simplifiée (sans changement majeur)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {filepath}: {e}")
        return ""

def save_chunks_to_json(chunks: List[Chunk], filepath: str):
    """Sauvegarde les chunks, utilisant maintenant le format avec 'metadata'."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Utilisation de chunk.to_dict() qui retourne le dictionnaire propre
        json.dump([chunk.to_dict() for chunk in chunks], 
                  open(filepath, 'w', encoding='utf-8'), 
                  indent=4, 
                  ensure_ascii=False)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du JSON {filepath}: {e}")

# ==============================================================================
# LOGIQUE PRINCIPALE
# ==============================================================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Création du dossier de sortie: '{OUTPUT_DIR}'")
        os.makedirs(OUTPUT_DIR)

    markdown_files = glob.glob(os.path.join(INPUT_DIR, '*.md'))
    
    if not markdown_files:
        print(f"Aucun fichier .md trouvé dans le dossier '{INPUT_DIR}'.")
        return

    print(f"--> {len(markdown_files)} fichiers Markdown à traiter...")
    
    markdown_chunker = MarkdownTagChunker()

    for md_filepath in markdown_files:
        filename = os.path.basename(md_filepath)
        print(f"\n--- Traitement de: {filename} ---")
        
        content = read_file(md_filepath)
        if not content:
            continue

        # Le cœur du traitement (Découpage Hybride)
        chunks = markdown_chunker.split(document_id=filename, content=content)
        
        if not chunks:
            print("Aucun chunk n'a pu être créé pour ce fichier.")
            continue
            
        output_filename = filename.replace('.md', '.json')
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        
        save_chunks_to_json(chunks, output_filepath)
        print(f"Chunks sauvegardés dans '{output_filepath}'")

    print("\nProcessus de découpage hybride terminé avec succès!")

if __name__ == "__main__":
    main()
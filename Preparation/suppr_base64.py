import re
import os
from pathlib import Path

def clean_base64_images(directory_path):
    # Pattern pour détecter les images Markdown en base64 (indépendant du texte alternatif)
    # Supporte data:image/png, jpeg, webp, etc.
    pattern = r'!\[.*?\]\(data:image\/[^;]+;base64,[^)]+\)'
    
    # Conversion en objet Path pour une manipulation plus facile
    base_dir = Path(directory_path)
    
    if not base_dir.exists():
        print(f"Erreur : Le dossier {directory_path} n'existe pas.")
        return

    print(f"Nettoyage du dossier : {base_dir}")
    
    # Parcours de tous les fichiers .md du dossier
    for md_file in base_dir.glob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Compte le nombre d'occurrences avant suppression
            matches = re.findall(pattern, content)
            if not matches:
                continue
                
            # Nettoyage
            clean_content = re.sub(pattern, '', content)
            
            # Sauvegarde seulement si des modifications ont eu lieu
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(clean_content)
                
            print(f"✅ {md_file.name} : {len(matches)} image(s) supprimée(s)")
            
        except Exception as e:
            print(f"❌ Erreur lors du traitement de {md_file.name} : {e}")

if __name__ == "__main__":
    # Chemin vers le dossier contenant les fichiers markdown
    target_folder = r"D:\Academique\Formation\DL\Projet_RAG\markdown_outputs"
    clean_base64_images(target_folder)
import os
import sys
import whisper
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# Ensemble des extensions audio à traiter avec Whisper
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
# Extensions d'images supportées par Docling
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}

def convert_documents_to_markdown(input_dir, output_dir):
    """
    Convertit les documents d'un répertoire en Markdown.
    - Utilise Whisper pour les fichiers audio.
    - Utilise Docling (avec OCR) pour les PDF, Images et documents Office.
    """
    # Vérifier si le répertoire d'entrée existe
    if not os.path.isdir(input_dir):
        print(f"Erreur : Le répertoire d'entrée '{input_dir}' n'existe pas.")
        return

    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Répertoire de sortie '{output_dir}' créé.")
        except OSError as e:
            print(f"Erreur lors de la création du répertoire de sortie '{output_dir}': {e}")
            return

    print(f"Début de la conversion des documents de '{input_dir}' vers '{output_dir}'...")

    # --- Configuration de Docling avec OCR ---
    print("Initialisation du convertisseur Docling (avec OCR)...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options.use_gpu = False # Mettre à True si disponible

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    # --- Chargement du modèle Whisper ---
    whisper_model = None
    try:
        has_audio_files = any(os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS for f in os.listdir(input_dir))
        if has_audio_files:
            print("Chargement du modèle Whisper 'base'...")
            whisper_model = whisper.load_model("base")
            print("Modèle Whisper chargé.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Whisper : {e}")
        
    # --- Traitement des fichiers ---
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        if os.path.isfile(input_path):
            base_name, extension = os.path.splitext(filename)
            output_filename = f"{base_name}.md"
            output_path = os.path.join(output_dir, output_filename)

            # Optionnel: ignorer si déjà converti
            # if os.path.exists(output_path): continue

            print(f"\nTraitement de '{filename}'...")

            ext_lower = extension.lower()

            # --- Logique Whisper pour les fichiers audio ---
            if ext_lower in AUDIO_EXTENSIONS:
                if whisper_model:
                    try:
                        print(f"Transcription de '{filename}' avec Whisper...")
                        result = whisper_model.transcribe(input_path, fp16=False)
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(result['text'])
                        print(f"Succès (Whisper) : '{filename}' -> '{output_filename}'.")
                    except Exception as e:
                        print(f"Erreur Whisper sur '{filename}' : {e}")
                else:
                    print(f"Whisper non disponible, '{filename}' ignoré.")

            # --- Logique Docling pour les autres fichiers (PDF, Images, Office) ---
            else:
                try:
                    print(f"Conversion de '{filename}' avec Docling (OCR active)...")
                    result = converter.convert(input_path)
                    markdown_content = result.document.export_to_markdown()
                    
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                    print(f"Succès (Docling) : '{filename}' -> '{output_filename}'.")
                except Exception as e:
                    print(f"Erreur Docling sur '{filename}' : {e}")

    print(f"\nConversion terminée. Les fichiers Markdown se trouvent dans '{output_dir}'.")

# --- Configuration ---
INPUT_DIRECTORY = 'Corpus'
OUTPUT_DIRECTORY = 'markdown_outputs'

# --- Exécution ---
if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
    INPUT_DIR = os.path.join(BASE_DIR, INPUT_DIRECTORY)
    OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIRECTORY)

    convert_documents_to_markdown(INPUT_DIR, OUTPUT_DIR)

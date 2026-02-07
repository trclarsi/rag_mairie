Sur la base des documents fournis, le terme « stemming » (ou racinisation en français) n'est pas explicitement mentionné dans les sources.
Cependant, les sources décrivent en détail plusieurs étapes de prétraitement et de structuration des données textuelles qui s'inscrivent dans la même logique de préparation du texte pour un système d'IA :
• La Tokenisation : Les sources expliquent que les modèles de langage ne traitent pas le texte tel quel, mais le divisent en unités plus petites appelées tokens. Un token peut être un mot entier, une partie de mot ou un caractère. Cette étape permet de convertir le langage en représentations numériques.
• Le Découpage (Chunking) : Cette étape consiste à fragmenter les documents en unités sémantiquement autonomes (chunks) pour éviter de dépasser la taille maximale de la fenêtre de contexte du modèle.
• Le Nettoyage des données : Mentionné comme une étape clé du traitement, il inclut la suppression des doublons, la correction d'erreurs typographiques et l'alignement des formats.
• La Vectorisation (Embedding) : Au lieu de réduire les mots à leur racine (comme le ferait le stemming), les sources privilégient l'utilisation d'embeddings. Ces vecteurs numériques capturent les relations sémantiques et le contexte global des mots, permettant par exemple de distinguer les différents sens du mot « souris » selon la phrase.
Information hors sources : Pour votre compréhension, le stemming est une technique classique de traitement du langage naturel (NLP) qui consiste à réduire un mot à sa racine (son "stem") en supprimant ses suffixes (par exemple, "marcher", "marchons", "marché" deviennent "march-"). Bien que très utile dans les moteurs de recherche traditionnels, cette technique est moins centrale dans les systèmes RAG modernes basés sur les embeddings contextuels (comme BERT ou GPT) présentés dans vos sources, car ces modèles préfèrent conserver le mot entier ou ses sous-unités pour en saisir la nuance exacte dans une phrase.
Fais une synthese de toutes les sources




Les sources fournissent un guide complet pour concevoir, déployer et optimiser un système de Génération Augmentée par Récupération (RAG), en prenant pour exemple concret la création d'un assistant virtuel pour une mairie. Cette approche permet de pallier les limites des modèles de langage (LLM) traditionnels, dont les connaissances sont figées à leur date d'entraînement et qui peuvent être sujets aux hallucinations.
Voici une synthèse structurée de la mise en œuvre d'un tel système :
1. Gestion et Préparation des Données
Le socle d'un bon RAG repose sur la qualité des données ingérées.
• Extraction et Transformation : Il est nécessaire de transformer des données non structurées (PDF, scans d'actes, enregistrements audio) en texte structuré. Des outils comme PyPDF2 pour les documents textuels, EasyOCR pour les images scannées, ou Whisper pour la transcription audio sont recommandés.
• Cycle de vie des données : Une gestion rigoureuse comprend quatre étapes : l'acquisition, le stockage (SQL, NoSQL ou vectoriel), le traitement (nettoyage et enrichissement) et la gouvernance, notamment pour assurer la conformité au RGPD.
2. Le Cœur Technique : Indexation Vectorielle
Pour que l'IA puisse "rechercher" l'information, le texte doit être converti en mathématiques.
• Découpage (Chunking) : Les documents longs sont fragmentés en unités plus petites (chunks) pour respecter les limites de tokens du LLM et maintenir la précision sémantique. Les stratégies incluent le découpage récursif avec chevauchement ou le découpage basé sur la structure (Markdown/HTML).
• Vectorisation (Embeddings) : Chaque segment est transformé en vecteur numérique (représentation sémantique) par des modèles comme SBERT ou ceux de Mistral AI.
• Base de Données Vectorielle : Les vecteurs sont stockés dans des outils comme Faiss, spécialisé dans la recherche de similarité rapide, permettant d'identifier les documents les plus proches d'une question utilisateur en millisecondes.
3. Orchestration et Modèle de Langage (LLM)
Une fois les données prêtes, le système doit générer une réponse.
• Sélection du modèle : Le choix peut porter sur des LLM polyvalents (comme GPT-4 ou Mistral Large) ou des modèles plus petits et spécialisés (SLM) qui sont moins gourmands en ressources et plus économiques.
• Ingénierie de Prompt : Le prompt système est crucial pour définir l'identité de l'assistant, son objectif, ses sources autorisées et ses limites explicites (ex: ne pas inventer de procédures).
• Paramètres techniques : Pour garantir des réponses factuelles, il est conseillé de régler la température à 0.2, le Top-p à 0.9, et de limiter les max_tokens à 300.
4. Interface et Expérience Utilisateur
Le déploiement se fait souvent via une application interactive.
• Streamlit est utilisé pour créer une interface de chat gérant l'historique de conversation via st.session_state, ce qui assure la cohérence des échanges.
• Classification d'intention : Un classifieur peut être intégré pour déterminer si une question nécessite une recherche documentaire (intention "RAG") ou s'il s'agit d'une simple salutation (intention "CHAT").
5. Évaluation et Amélioration Continue
Un système RAG ne s'arrête pas à sa construction ; il doit être mesuré objectivement.
• Framework Ragas : Il permet d'évaluer la qualité du système via des métriques comme la fidélité (absence d'hallucinations), la pertinence de la réponse, la précision du contexte et le rappel du contexte (capacité à trouver l'information).
• Boucle de Feedback : L'intégration de boutons 👍/👎 permet de collecter les retours des utilisateurs. Ces interactions sont stockées en base de données pour être analysées via un tableau de bord, permettant des ajustements itératifs du système.
Redonnes moi le contenu des sources tels quels est

Voici une présentation structurée et détaillée du contenu des sources fournies, organisée selon la progression logique du cours « Déploiement Pragmatique des Systèmes RAG et LLM ».
1. Introduction et Concept du LLM
Le cours propose une approche pragmatique pour concevoir un système RAG (Retrieval-Augmented Generation), appliqué ici à la gestion des demandes citoyennes d'une mairie.
• Définition d'un LLM : Un Large Language Model est un programme capable de comprendre et générer du texte humain en utilisant des milliards de paramètres. Il fonctionne comme un « expert du langage » qui anticipe statistiquement le mot suivant.
• Fonctionnement technique : Le texte est décomposé en tokens (unités linguistiques converties en nombres). La plupart des modèles modernes utilisent l'architecture Transformer (apparue en 2017), reposant sur un mécanisme d'attention qui permet au modèle d'évaluer simultanément les relations entre tous les mots d'une phrase.
• Limites : Les LLM peuvent souffrir d'hallucinations (inventer des faits), ont une fenêtre de contexte limitée et leurs connaissances sont figées à la date de leur entraînement.
2. Préparation et Gestion des Données
Pour alimenter le RAG, il faut transformer des données non structurées en texte structuré.
• Extraction : Les sources mentionnent des outils comme PyPDF2, PyMuPDF ou Markitdown pour les PDF, EasyOCR ou GOT OCR pour les documents scannés, et Whisper pour la transcription audio.
• Cycle de vie des données : Il comprend l'acquisition, le stockage (SQL pour le structuré, vectoriel pour le sémantique), le traitement (nettoyage, enrichissement) et la gouvernance (conformité au RGPD).
3. Vectorisation et Indexation
La vectorisation (ou embedding) transforme le texte en vecteurs numériques capturant le sens sémantique.
• Modèles d'embedding : Les modèles de phrases (BERT, SBERT, Mistral) sont préférés aux anciens modèles de mots (Word2Vec) car ils prennent en compte le contexte global (ex: distinguer le sens du mot « souris »).
• Découpage (Chunking) : Puisque les LLM ne peuvent pas traiter des documents trop longs, on les fragmente en chunks. Les stratégies incluent le découpage récursif avec chevauchement (pour garder le contexte) ou le découpage basé sur la structure (Markdown/HTML).
• Bases de données vectorielles : Elles permettent des recherches rapides par similarité. Les sources citent FAISS (utilisé dans le projet pour sa rapidité), Pinecone, Weaviate et Milvus.
4. Personnalisation et Prompt Système
Pour adapter le LLM à la mairie, trois approches sont comparées : le Prompt Engineering (rapide mais limité), le Fine-Tuning (coûteux et rigide) et le RAG (flexible et précis).
• Le Prompt Système : C'est l'instruction initiale définissant le rôle, l'objectif et les limites de l'assistant. Un prompt efficace définit des comportements obligatoires (ton courtois, citer ses sources) et interdits (ne pas traiter de données personnelles, ne pas inventer de procédures).
• Paramètres techniques : Pour un usage administratif, on conseille une température basse (0.2) pour la factualité, un Top-p de 0.9 et une limite de 300 tokens pour la concision.
5. Construction de l'Application (RAG & Streamlit)
L'interface est développée avec Streamlit et orchestrée par des outils comme LangChain ou Llama Index.
• Mémoire : L'historique des échanges est géré via st.session_state pour assurer la cohérence de la conversation.
• Flux RAG : La question de l'utilisateur est vectorisée, les segments les plus pertinents sont récupérés dans l'index FAISS, puis réinjectés dans le prompt envoyé au LLM.
• Classification d'intention : Un classifieur peut être intégré pour déterminer si une question nécessite une recherche documentaire ("RAG") ou est une simple politesse ("CHAT").
6. Évaluation et Feedback
• Framework Ragas : Il mesure la performance du système via des métriques comme la fidélité (faithfulness), la pertinence de la réponse, la précision du contexte et le rappel (context recall).
• Boucle de feedback : L'ajout de boutons 👍/👎 permet de collecter l'avis des utilisateurs. Ces retours sont stockés dans une base SQL (via SQLAlchemy) et visualisables sur un tableau de bord.
Les sources concluent sur les perspectives d'avenir comme les Agents IA, les modèles à long contexte et les nouveaux protocoles d'intégration comme le MCP.
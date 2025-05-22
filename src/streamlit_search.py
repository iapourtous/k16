#!/usr/bin/env python3
"""
Interface Streamlit pour la recherche K16
Permet de rechercher des questions similaires dans la base Natural Questions
"""

import streamlit as st
import numpy as np
import time
import sys
import os
from sentence_transformers import SentenceTransformer

# Add the parent directory to sys.path to make the lib modules importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importation des modules de la bibliothèque K16
from lib.config import ConfigManager
from lib.io import VectorReader, TreeIO
from lib.search import Searcher

@st.cache_resource
def load_resources():
    """Charge les ressources nécessaires (modèle, vecteurs, arbre)"""
    # Configuration
    config_manager = ConfigManager()
    
    # Chemins des fichiers
    files_config = config_manager.get_section("files")
    vectors_path = os.path.join(files_config["vectors_dir"], files_config["default_vectors"])
    tree_path = os.path.join(files_config["trees_dir"], files_config["default_tree"])
    qa_path = os.path.join(files_config["vectors_dir"], files_config["default_qa"])
    
    # Charger le modèle d'embeddings
    prepare_config = config_manager.get_section("prepare_data")
    model_name = prepare_config.get("model", "intfloat/multilingual-e5-large")
    with st.spinner(f"Chargement du modèle {model_name}..."):
        model = SentenceTransformer(model_name)
    
    # Charger les vecteurs
    with st.spinner("Chargement des vecteurs..."):
        vectors_reader = VectorReader(vectors_path, mode="ram")
    
    # Charger l'arbre avec structure plate optimisée
    with st.spinner("Chargement de l'arbre de recherche..."):
        tree_io = TreeIO()
        # Utiliser la structure plate optimisée
        k16tree = tree_io.load_as_k16tree(tree_path, use_flat_structure=True)

        # Vérifier si la conversion a réussi
        if hasattr(k16tree, 'flat_tree') and k16tree.flat_tree is not None:
            st.success("✓ Structure plate optimisée chargée avec succès!")
        else:
            st.warning("⚠️ Utilisation de l'arbre standard (structure plate non disponible)")
    
    # Charger les textes
    with st.spinner("Chargement des questions et réponses..."):
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_lines = f.readlines()
    
    # Créer le chercheur avec les paramètres de configuration
    search_config = config_manager.get_section("search")
    searcher = Searcher(
        k16tree,  # Utiliser l'arbre avec structure plate
        vectors_reader,
        use_faiss=True,
        search_type=search_config.get("search_type", "beam"),  # Utiliser beam search par défaut
        beam_width=search_config.get("beam_width", 2)  # Beam width optimisé par défaut
    )
    
    return model, searcher, qa_lines, config_manager

def search_similar_questions(query, model, searcher, qa_lines, k=10):
    """Recherche les questions similaires avec métriques détaillées"""
    # Mesures de temps
    timings = {}

    # Encoder la requête
    encode_start = time.time()
    query_vector = model.encode(f"query: {query}", normalize_embeddings=True)
    timings["encode"] = time.time() - encode_start

    # Rechercher les k plus proches voisins
    search_start = time.time()

    # Si on a une structure plate, on peut mesurer séparément le temps de tree_search et filter
    has_flat = hasattr(searcher, 'k16tree') and hasattr(searcher.k16tree, 'flat_tree') and searcher.k16tree.flat_tree is not None

    # Effectuer la recherche
    if has_flat and searcher.search_type == "beam":
        # Utiliser directement la structure plate pour optimiser
        tree_search_start = time.time()
        flat_tree = searcher.k16tree.flat_tree

        # Utiliser la nouvelle capacité de tri intégré de TreeFlat.
        candidates = flat_tree.search_tree(
            query_vector,
            searcher.beam_width,
            vectors_reader=searcher.vectors_reader,
            k=k,
        )

        timings["tree_search"] = time.time() - tree_search_start
        indices = candidates  # Déjà triés et limités à k
        timings["filter"] = 0  # Plus de filtrage additionnel
    else:
        # Recherche standard
        indices = searcher.search_k_nearest(query_vector, k=k)
        timings["tree_search"] = time.time() - search_start
        timings["filter"] = 0  # Inclus dans tree_search

    search_time = time.time() - search_start

    # Récupérer les résultats
    results = []
    for idx in indices:
        if idx < len(qa_lines):
            parts = qa_lines[idx].strip().split(" ||| ")
            if len(parts) == 2:
                question, answer = parts
                results.append({
                    "question": question,
                    "answer": answer,
                    "index": idx
                })

    # Calculer le temps total
    total_time = timings["encode"] + search_time

    # Retourner les résultats et les métriques
    return results, total_time, timings

# Configuration de la page
st.set_page_config(
    page_title="K16 - Recherche de Questions Similaires",
    page_icon="🔍",
    layout="wide"
)

# Titre principal
st.title("🔍 K16 - Recherche de Questions Similaires")
st.markdown("### Interface de recherche rapide basée sur l'arbre K16")

# Chargement des ressources
try:
    model, searcher, qa_lines, config_manager = load_resources()
    search_config = config_manager.get_section("search")
except Exception as e:
    st.error(f"Erreur lors du chargement des ressources : {str(e)}")
    st.stop()

# Barre latérale avec les paramètres
with st.sidebar:
    st.header("⚙️ Paramètres")
    
    # Nombre de résultats
    k = st.slider(
        "Nombre de résultats",
        min_value=1,
        max_value=50,
        value=10,
        step=1
    )
    
    # Informations sur les données
    st.header("📊 Informations")
    st.info(f"""
    **Base de données**
    - {len(qa_lines):,} questions-réponses
    - Dimension : {model.get_sentence_embedding_dimension()}
    - Modèle : {config_manager.get_section('prepare_data')['model']}
    """)
    
    # Mode de recherche
    st.header("🔎 Mode")

    # Vérifier si la structure plate est utilisée
    if hasattr(searcher, 'k16tree') and hasattr(searcher.k16tree, 'flat_tree') and searcher.k16tree.flat_tree is not None:
        flat_tree = searcher.k16tree.flat_tree
        st.success(f"🚀 Recherche optimisée avec structure plate\n- Mode: {searcher.search_type}\n- Beam width: {searcher.beam_width}")

        # Afficher les statistiques de l'arbre plat
        stats = flat_tree.get_statistics()
        st.info(f"""
        **Structure plate**
        - Noeuds: {stats.get('n_nodes', '?')}
        - Feuilles: {stats.get('n_leaves', '?')}
        - Profondeur: {stats.get('max_depth', '?')}
        - Taille moy. feuilles: {stats.get('avg_leaf_size', '?'):.1f}
        """)
    else:
        st.warning(f"Recherche standard avec arbre K16\n- Mode: {searcher.search_type}")

# Zone de recherche principale
with st.form(key="search_form"):
    query = st.text_input(
        "Entrez votre question",
        placeholder="Par exemple : Quand a été construit la tour Eiffel ?",
        help="Tapez une question pour trouver des questions similaires dans la base",
        key="search_query"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.form_submit_button("🔍 Rechercher", type="primary", use_container_width=True)

# Effectuer la recherche
if search_button and query:
    with st.spinner("Recherche en cours..."):
        results, total_time, timings = search_similar_questions(query, model, searcher, qa_lines, k=k)

    # Afficher le temps de recherche avec un résumé
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Recherche terminée en **{total_time*1000:.2f} ms**")
    with col2:
        has_flat = hasattr(searcher, 'k16tree') and hasattr(searcher.k16tree, 'flat_tree') and searcher.k16tree.flat_tree is not None
        structure_label = "Structure plate optimisée" if has_flat else "Structure standard"
        st.info(f"{structure_label} • {len(results)} résultats")

    # Afficher les métriques détaillées
    with st.expander("📊 Métriques détaillées"):
        metrics_cols = st.columns(3)
        with metrics_cols[0]:
            st.metric("Encodage", f"{timings['encode']*1000:.2f} ms")
        with metrics_cols[1]:
            st.metric("Recherche arbre", f"{timings['tree_search']*1000:.2f} ms")
        with metrics_cols[2]:
            search_only = total_time - timings['encode']
            st.metric("Recherche totale", f"{search_only*1000:.2f} ms")

    # Afficher les résultats
    if results:
        st.markdown("### 📋 Résultats")

        for i, result in enumerate(results, 1):
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{i}. {result['question']}**")
                    st.markdown(f"**Réponse :** {result['answer']}")
                if i < len(results):
                    st.divider()
    else:
        st.warning("Aucun résultat trouvé")

# Section d'aide
with st.expander("❓ Aide"):
    st.markdown("""
    ### Comment utiliser cette interface ?
    
    1. **Entrez une question** dans le champ de recherche
    2. **Cliquez sur Rechercher** ou appuyez sur Entrée
    3. **Consultez les résultats** classés par similarité
    
    ### À propos de K16
    
    K16 est un système de recherche rapide basé sur un arbre de clustering hiérarchique.
    Il permet de trouver efficacement les questions les plus similaires dans une grande base de données.
    
    ### Paramètres
    
    - **Nombre de résultats** : Ajustez le nombre de questions similaires à afficher
    - **Mode K16** : Utilise l'arbre optimisé pour une recherche ultra-rapide
    """)

# Pied de page
st.markdown("---")
st.caption("K16 Search Engine - Recherche rapide de questions similaires")
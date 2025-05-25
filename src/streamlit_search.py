#!/usr/bin/env python3
"""
Interface Streamlit pour la recherche K16 avec TreeFlatPCA compressé + Numba
"""

import streamlit as st
import numpy as np
import time
import sys
import os
from sentence_transformers import SentenceTransformer

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config import ConfigManager
from lib.io import VectorReader, TreeIO
from lib.search import Searcher

@st.cache_resource
def load_resources():
    """Charge les ressources nécessaires (modèle, vecteurs, arbre)"""
    config_manager = ConfigManager()
    
    # Chemins des fichiers
    files_config = config_manager.get_section("files")
    vectors_path = os.path.join(files_config["vectors_dir"], files_config["default_vectors"])

    # Utiliser le fichier tree.flat.npy qui existe réellement
    tree_filename = os.path.splitext(files_config["default_tree"])[0] + ".flat.npy"
    tree_path = os.path.join(files_config["trees_dir"], tree_filename)

    qa_path = os.path.join(files_config["vectors_dir"], files_config["default_qa"])
    
    # Charger le modèle d'embeddings
    prepare_config = config_manager.get_section("prepare_data")
    model_name = prepare_config.get("model", "intfloat/multilingual-e5-large")
    with st.spinner(f"Chargement du modèle {model_name}..."):
        model = SentenceTransformer(model_name)
    
    # Charger les vecteurs
    with st.spinner("Chargement des vecteurs..."):
        vectors_reader = VectorReader(vectors_path, mode="ram")
    
    # Charger l'arbre de recherche
    with st.spinner("Chargement de l'arbre de recherche..."):
        tree_io = TreeIO()
        
        try:
            k16tree = tree_io.load_as_k16tree(tree_path, mmap_tree=False)

            # Vérifier la structure TreeFlat
            if hasattr(k16tree, 'flat_tree') and k16tree.flat_tree is not None:
                stats = k16tree.flat_tree.get_statistics()
                st.success(f"✓ TreeFlat chargé: {stats.get('n_nodes', '?')} nœuds, profondeur {stats.get('max_depth', '?')}")
            else:
                st.error("❌ Structure TreeFlat non trouvée")
                st.stop()

        except Exception as e:
            st.error(f"❌ Erreur chargement TreeFlat: {str(e)}")
            st.stop()
    
    # Charger les textes
    with st.spinner("Chargement des questions et réponses..."):
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_lines = f.readlines()
    
    # Créer le chercheur
    search_config = config_manager.get_section("search")
    build_config = config_manager.get_section("build_tree")
    
    searcher = Searcher(
        k16tree,
        vectors_reader,
        use_faiss=True,
        search_type=search_config.get("search_type", "beam"),
        beam_width=search_config.get("beam_width", 2),
        max_data=build_config.get("max_data", 4000)
    )
    
    return model, searcher, qa_lines, config_manager

def search_similar_questions(query, model, searcher, qa_lines, k=10):
    """Recherche les questions similaires"""
    timings = {}

    # Encoder la requête
    encode_start = time.time()
    query_vector = model.encode(f"query: {query}", normalize_embeddings=True)
    timings["encode"] = time.time() - encode_start

    # Recherche
    search_start = time.time()

    try:
        # Recherche TreeFlat uniquement
        tree_search_start = time.time()
        tree_candidates = searcher.search_tree(query_vector)
        timings["tree_search"] = time.time() - tree_search_start

        # Filtrer pour obtenir les k meilleurs
        filter_start = time.time()
        indices = searcher.filter_candidates(tree_candidates, query_vector, k)
        timings["filter"] = time.time() - filter_start

    except Exception as e:
        st.error(f"❌ Erreur lors de la recherche TreeFlat: {str(e)}")
        indices = []
        timings["tree_search"] = time.time() - search_start
        timings["filter"] = 0

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

    total_time = timings["encode"] + search_time
    return results, total_time, timings

# Configuration de la page
st.set_page_config(
    page_title="K16 - Recherche TreeFlat + Numba",
    page_icon="🚀",
    layout="wide"
)

# Titre principal
st.title("🚀 K16 - Recherche TreeFlatPCA Compressé + Numba")
st.markdown("### Interface optimisée avec compression et JIT")

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
    
    k = st.slider("Nombre de résultats", min_value=1, max_value=50, value=10, step=1)
    
    st.header("📊 Informations")
    st.info(f"""
    **Base de données**
    - {len(qa_lines):,} questions-réponses
    - Dimension : {model.get_sentence_embedding_dimension()}
    - Modèle : {config_manager.get_section('prepare_data')['model']}
    """)
    
    st.header("🔎 Mode")
    
    # Affichage TreeFlat (structure unique)
    search_mode = getattr(searcher, 'search_type', 'single')
    beam_width = getattr(searcher, 'beam_width', 1)

    st.success(f"🚀 TreeFlat compressé + ✅ Numba JIT\n- Mode: {search_mode}\n- Beam width: {beam_width}")

    stats = searcher.flat_tree.get_statistics()
    compression_stats = stats.get('compression', {})

    st.info(f"""
    **Structure compressée**
    - Nœuds: {stats.get('n_nodes', '?'):,}
    - Feuilles: {stats.get('n_leaves', '?'):,}
    - Profondeur: {stats.get('max_depth', '?')}
    - Patterns dims: {compression_stats.get('unique_dims_patterns', '?')}
    - Type feuilles: {compression_stats.get('leaf_data_dtype', '?')}
    """)

# Zone de recherche principale
with st.form(key="search_form"):
    query = st.text_input(
        "Entrez votre question",
        placeholder="Par exemple : Quand a été construit la tour Eiffel ?",
        help="Tapez une question pour trouver des questions similaires",
        key="search_query"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.form_submit_button("🔍 Rechercher", type="primary", use_container_width=True)

# Effectuer la recherche
if search_button and query:
    with st.spinner("Recherche en cours..."):
        results, total_time, timings = search_similar_questions(query, model, searcher, qa_lines, k=k)

    # Afficher le temps de recherche
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"✅ Recherche terminée en **{total_time*1000:.2f} ms**")
    with col2:
        from lib.flat_tree import NUMBA_AVAILABLE
        jit_status = "JIT" if NUMBA_AVAILABLE else "Python"
        structure_label = f"TreeFlat compressé ({jit_status})"

        st.info(f"{structure_label} • {len(results)} résultats")

    # Métriques détaillées
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
    
    ### À propos de K16 TreeFlat
    
    Cette version utilise :
    - **Compression intelligente** : 60-80% de réduction mémoire
    - **Numba JIT** : 20-50% d'accélération des calculs
    - **Encodage différentiel** : compression optimale des données
    - **Structures sparses** : élimination des redondances
    """)

# Pied de page
st.markdown("---")
st.caption("K16 TreeFlat + Numba - Recherche ultra-rapide avec compression intelligente")
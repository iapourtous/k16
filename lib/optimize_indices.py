"""
Module d'optimisation des indices pour K16.
Implémente des algorithmes pour optimiser les indices stockés dans les feuilles
en utilisant des méthodes de recherche approximative rapide comme HNSW.
Version ultra-rapide par lot basée sur HNSW.
"""

import time
import numpy as np
from typing import List, Optional, Tuple
from tqdm.auto import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS n'est pas installé. L'optimisation des indices sera plus lente.")

def optimize_leaf_indices(tree, vectors: np.ndarray, max_data: int, use_gpu: bool = False) -> any:
    """
    Post-traitement ultra-rapide pour optimiser les indices dans chaque feuille en utilisant HNSW.
    Remplace les indices existants par les max_data vecteurs globalement les plus proches
    en utilisant une recherche par lot optimisée.

    Args:
        tree: L'arbre K16 à optimiser
        vectors: Les vecteurs d'entrée
        max_data: Nombre maximum de vecteurs à stocker dans chaque feuille
        use_gpu: Utiliser le GPU pour l'accélération si disponible

    Returns:
        L'arbre avec des indices optimisés
    """
    print("⏳ Optimisation des indices de feuilles avec HNSW (méthode par lot ultra-rapide)...")
    start_time = time.time()
    
    # Collecte des centroïdes et feuilles
    leaf_nodes = []
    centroids = []
    
    # Collecter toutes les feuilles et leurs centroïdes en une seule passe
    def collect_leaves_and_centroids(node) -> None:
        """Collecte les feuilles et leurs centroïdes en une seule passe."""
        # Gestion selon le type du nœud (TreeNode ou tuple)
        if hasattr(node, 'children'):
            if not node.children:  # C'est une feuille (TreeNode)
                leaf_nodes.append(node)
                centroids.append(node.centroid)
            else:
                for child in node.children:
                    collect_leaves_and_centroids(child)
        elif isinstance(node, tuple) and len(node) >= 2:
            if not node[1]:  # C'est une feuille (tuple)
                leaf_nodes.append(node)
                centroids.append(node[0])
            else:
                for child in node[1]:  # Children sont dans node[1] pour le format tuple
                    collect_leaves_and_centroids(child)
    
    collect_leaves_and_centroids(tree)
    n_leaves = len(leaf_nodes)
    print(f"  → {n_leaves:,} feuilles trouvées")
    
    if n_leaves == 0:
        print("⚠️ Aucune feuille trouvée, rien à optimiser.")
        return tree
    
    # Préparation des centroïdes pour la recherche par lot
    if not centroids:
        print("⚠️ Aucun centroïde trouvé, rien à optimiser.")
        return tree
    
    # Créer un tableau numpy des centroïdes normalisés
    centroids_array = np.zeros((len(centroids), vectors.shape[1]), dtype=np.float32)
    for i, centroid in enumerate(centroids):
        if isinstance(centroid, np.ndarray):
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroids_array[i] = centroid / norm
            else:
                # Centroïde nul, utiliser un vecteur aléatoire normalisé
                random_vec = np.random.randn(vectors.shape[1])
                centroids_array[i] = random_vec / np.linalg.norm(random_vec)
        else:
            # Conversion depuis un autre format
            tmp = np.array(centroid, dtype=np.float32)
            norm = np.linalg.norm(tmp)
            if norm > 0:
                centroids_array[i] = tmp / norm
            else:
                random_vec = np.random.randn(vectors.shape[1])
                centroids_array[i] = random_vec / np.linalg.norm(random_vec)
    
    prep_time = time.time() - start_time
    print(f"  → Centroïdes préparés en {prep_time:.2f}s")
    
    # Utilisation optimisée de FAISS pour la recherche par lot
    if FAISS_AVAILABLE:
        # Création d'un index HNSW optimisé pour la recherche rapide
        d = vectors.shape[1]
        n_vectors = vectors.shape[0]
        
        # Paramètres optimisés pour HNSW
        M = 64  # Nombre de connexions par nœud (plus élevé = meilleure qualité)
        ef_construction = 400  # Qualité de construction (plus élevé = meilleure qualité)
        
        # Créer l'index avec la métrique du produit intérieur (pour cosinus)
        if use_gpu and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
            print("  → Utilisation du GPU pour la recherche HNSW")
            # Construire l'index CPU d'abord
            index_cpu = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
            index_cpu.hnsw.efConstruction = ef_construction
            # Passer au GPU
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        else:
            print("  → Utilisation du CPU pour la recherche HNSW")
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
        
        # Ajouter les vecteurs à l'index
        build_start = time.time()
        
        # Par lots pour les très grandes collections
        if n_vectors > 1000000:
            batch_size = 100000
            for i in range(0, n_vectors, batch_size):
                end_idx = min(i + batch_size, n_vectors)
                index.add(vectors[i:end_idx].astype(np.float32))
                if i % 500000 == 0 and i > 0:
                    print(f"    - {i:,} vecteurs ajoutés à l'index...")
        else:
            # Ajouter tous les vecteurs d'un coup
            index.add(vectors.astype(np.float32))
        
        build_time = time.time() - build_start
        print(f"  → Index HNSW construit en {build_time:.2f}s")
        
        # Optimiser efSearch pour la qualité/vitesse
        ef_search = max(max_data * 2, 128)  # Paramètre clé pour la qualité de recherche
        index.hnsw.efSearch = ef_search
        
        # Recherche par lot pour toutes les feuilles en une seule opération
        print(f"  → Recherche des {max_data} plus proches voisins pour {n_leaves} feuilles en une seule passe...")
        print(f"    (efSearch={ef_search}, M={M})")
        batch_search_start = time.time()
        
        # Recherche par lots avec des batches pour les très grands arbres
        batch_size = 5000  # Taille de lot optimale déterminée empiriquement
        all_indices = []
        
        if n_leaves > batch_size:
            # Traitement par lots pour les très grands arbres
            n_batches = (n_leaves + batch_size - 1) // batch_size
            for i in range(0, n_leaves, batch_size):
                end_idx = min(i + batch_size, n_leaves)
                current_batch = centroids_array[i:end_idx]
                _, batch_indices = index.search(current_batch, max_data)
                all_indices.extend(batch_indices.tolist())
                print(f"    - Lot {i//batch_size + 1}/{n_batches} traité ({end_idx-i} feuilles)")
        else:
            # Recherche en une seule fois pour les arbres plus petits
            _, all_indices_array = index.search(centroids_array, max_data)
            all_indices = all_indices_array.tolist()
        
        batch_search_time = time.time() - batch_search_start
        print(f"  → Recherche par lot terminée en {batch_search_time:.2f}s")
        print(f"    Vitesse moyenne: {(batch_search_time*1000/n_leaves):.2f} ms par feuille")
    else:
        # Version fallback CPU (sans FAISS)
        print("⚠️ FAISS non disponible, utilisation d'une recherche naïve (beaucoup plus lente)")
        
        # Recherche naïve par lot
        batch_search_start = time.time()
        
        # Initialiser la liste pour les résultats
        all_indices = []
        
        # Boucle sur les centroïdes
        for i, centroid in tqdm(enumerate(centroids_array), total=n_leaves, desc="Recherche des plus proches voisins"):
            # Calcul des produits scalaires avec tous les vecteurs
            similarities = np.dot(vectors, centroid)
            # Obtenir les indices des plus proches voisins
            top_indices = np.argsort(-similarities)[:max_data]
            # Stocker dans la liste de résultats
            all_indices.append(top_indices.tolist())
        
        batch_search_time = time.time() - batch_search_start
        print(f"  → Recherche naïve terminée en {batch_search_time:.2f}s")
    
    # Mise à jour des indices dans les feuilles
    update_start = time.time()
    
    # Application des nouveaux indices aux feuilles
    for i, leaf in enumerate(leaf_nodes):
        indices_list = all_indices[i]
        
        if hasattr(leaf, 'indices'):
            # Format TreeNode
            leaf.indices = indices_list
        else:
            # Format tuple
            if len(leaf) >= 3:
                # Créer un nouveau tuple avec les indices mis à jour
                leaf_nodes[i] = (leaf[0], leaf[1], indices_list) + leaf[3:]
    
    update_time = time.time() - update_start
    print(f"  → Mise à jour des feuilles terminée en {update_time:.2f}s")
    
    # Temps total et statistiques
    total_time = time.time() - start_time
    print(f"✓ Optimisation des indices terminée en {total_time:.2f}s")
    
    avg_time_per_leaf = (total_time * 1000) / max(n_leaves, 1)
    print(f"  → Temps moyen par feuille: {avg_time_per_leaf:.3f} ms")
    if FAISS_AVAILABLE:
        print(f"  → Dont {(batch_search_time*1000/n_leaves):.3f} ms pour la recherche HNSW")
    
    return tree
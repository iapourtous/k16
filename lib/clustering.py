"""
Module de clustering pour K16.
Implémente les algorithmes de clustering utilisés pour construire l'arbre K16.
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Any, Optional, Union
import multiprocessing
from joblib import Parallel, delayed
from tqdm.auto import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS n'est pas installé. K-means utilisera une implémentation plus lente.")

try:
    from kneed import KneeLocator
    KNEEDLE_AVAILABLE = True
except ImportError:
    KNEEDLE_AVAILABLE = False
    print("⚠️ Module kneed n'est pas installé. La méthode du coude utilisera une approche simplifiée.")

from .tree import TreeNode

def kmeans_faiss(vectors: np.ndarray, k: int, gpu: bool = False, niter: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means sphérique optimisé pour les embeddings normalisés.
    Utilise la similarité cosinus (produit scalaire) au lieu de la distance euclidienne.

    Args:
        vectors: Les vecteurs à clusteriser (shape: [n, d])
        k: Nombre de clusters
        gpu: Utiliser le GPU si disponible
        niter: Nombre d'itérations

    Returns:
        Tuple[np.ndarray, np.ndarray]: (centroids, labels)
    """
    # S'assurer que les vecteurs sont en float32
    vectors = vectors.astype(np.float32)

    # Vérifier si les vecteurs sont normalisés
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        # Normaliser si nécessaire
        vectors = vectors / norms[:, np.newaxis]

    # Dimension des vecteurs
    d = vectors.shape[1]
    n = vectors.shape[0]

    # Ne pas créer plus de clusters que de points
    if k > n:
        k = n

    if FAISS_AVAILABLE:
        # Initialisation K-means++ pour une meilleure convergence
        centroids = spherical_kmeans_plusplus(vectors, k)

        # Créer un index pour la similarité cosinus (produit scalaire)
        gpu_available = gpu and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
        if gpu_available:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatIP(res, d)  # IP = Inner Product
        else:
            index = faiss.IndexFlatIP(d)  # Pour vecteurs normalisés = cosinus

        # Algorithme K-means sphérique
        labels = np.zeros(n, dtype=np.int32)
        prev_inertia = float('inf')

        for i in range(niter):
            # Ajouter les centroïdes à l'index
            index.reset()
            index.add(centroids)

            # Assigner chaque point au centroïde le plus proche (similarité maximale)
            similarities, idx = index.search(vectors, 1)
            labels = idx.reshape(-1)

            # Recalculer les centroïdes
            new_centroids = np.zeros((k, d), dtype=np.float32)
            counts = np.zeros(k, dtype=np.int32)

            for j, label in enumerate(labels):
                new_centroids[label] += vectors[j]
                counts[label] += 1

            # Gérer les clusters et normaliser
            for cluster_idx in range(k):
                if counts[cluster_idx] == 0:
                    # Cluster vide : assigner un point aléatoire
                    random_idx = np.random.randint(0, n)
                    new_centroids[cluster_idx] = vectors[random_idx]
                else:
                    # Normaliser le centroïde (projection sur la sphère unitaire)
                    centroid = new_centroids[cluster_idx]
                    new_centroids[cluster_idx] = centroid / np.linalg.norm(centroid)

            # Calculer l'inertie pour vérifier la convergence
            inertia = -similarities.sum()  # Négatif car on maximise la similarité

            # Vérifier la convergence
            if abs(prev_inertia - inertia) < 1e-4 * abs(prev_inertia):
                break

            prev_inertia = inertia
            centroids = new_centroids
    else:
        # Version fallback sans FAISS
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        # Normaliser les vecteurs
        vectors_norm = normalize(vectors, norm='l2')

        # K-means avec similarité cosinus
        kmeans = KMeans(n_clusters=k, max_iter=niter, n_init=1, random_state=42)
        labels = kmeans.fit_predict(vectors_norm)

        # Normaliser les centroïdes
        centroids = normalize(kmeans.cluster_centers_, norm='l2')

    return centroids, labels


def spherical_kmeans_plusplus(vectors: np.ndarray, k: int) -> np.ndarray:
    """
    Initialisation K-means++ adaptée pour le clustering sphérique.
    Sélectionne intelligemment les centroïdes initiaux.

    Args:
        vectors: Vecteurs normalisés
        k: Nombre de clusters

    Returns:
        np.ndarray: Centroïdes initiaux
    """
    n, d = vectors.shape
    centroids = np.zeros((k, d), dtype=np.float32)

    # Choisir le premier centroïde aléatoirement
    centroids[0] = vectors[np.random.randint(0, n)]

    # Si FAISS est disponible, l'utiliser pour accélérer
    if FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(d)

        for i in range(1, k):
            # Calculer les distances au centroïde le plus proche
            index.reset()
            index.add(centroids[:i])
            similarities, _ = index.search(vectors, 1)

            # Convertir similarités en distances angulaires
            similarities = np.clip(similarities.reshape(-1), -1, 1)
            angular_distances = np.arccos(similarities)

            # Probabilité proportionnelle au carré de la distance
            probabilities = angular_distances ** 2
            probabilities = probabilities / probabilities.sum()

            # Sélectionner le prochain centroïde
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            idx = np.searchsorted(cumulative_probs, r)
            centroids[i] = vectors[idx]
    else:
        # Version sans FAISS
        for i in range(1, k):
            # Calculer les distances au centroïde le plus proche
            min_distances = np.ones(n) * float('inf')
            for j in range(i):
                similarities = np.dot(vectors, centroids[j])
                angular_distances = np.arccos(np.clip(similarities, -1, 1))
                min_distances = np.minimum(min_distances, angular_distances)

            # Probabilité proportionnelle au carré de la distance
            probabilities = min_distances ** 2
            probabilities = probabilities / probabilities.sum()

            # Sélectionner le prochain centroïde
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            idx = np.searchsorted(cumulative_probs, r)
            centroids[i] = vectors[idx]

    return centroids

def find_optimal_k(vectors: np.ndarray, k_min: int = 2, k_max: int = 32, gpu: bool = False) -> int:
    """
    Trouve le nombre optimal de clusters k en utilisant la méthode du coude.
    Adapté pour le k-means sphérique avec distance angulaire.

    Args:
        vectors: Les vecteurs à clusteriser
        k_min: Nombre minimum de clusters à tester
        k_max: Nombre maximum de clusters à tester
        gpu: Utiliser le GPU si disponible

    Returns:
        int: Nombre optimal de clusters k
    """
    # Limiter k_max par rapport au nombre de vecteurs
    k_max = min(k_max, len(vectors) // 2) if len(vectors) > 2 else k_min
    k_range = range(k_min, k_max + 1)

    # Si nous avons trop peu de vecteurs, retourner le minimum
    if len(k_range) <= 1:
        return k_min

    inertias = []

    # Calculer l'inertie pour différentes valeurs de k
    for k in tqdm(k_range, desc="Recherche du k optimal"):
        centroids, labels = kmeans_faiss(vectors, k, gpu=gpu, niter=15)  # Moins d'itérations pour la recherche

        # Calculer l'inertie adaptée au clustering sphérique
        inertia = 0
        for i in range(k):
            cluster_vectors = vectors[labels == i]
            if len(cluster_vectors) > 0:
                # Calculer les similarités avec le centroïde
                similarities = np.dot(cluster_vectors, centroids[i])
                # Convertir en distances angulaires et sommer
                angular_distances = np.arccos(np.clip(similarities, -1, 1))
                inertia += np.sum(angular_distances ** 2)

        inertias.append(inertia)

    # Utiliser KneeLocator si disponible, sinon utiliser une approche simplifiée
    if KNEEDLE_AVAILABLE:
        try:
            kneedle = KneeLocator(
                list(k_range), inertias,
                curve="convex", direction="decreasing", S=1.0
            )

            # Si un coude est trouvé, l'utiliser
            if kneedle.knee is not None:
                optimal_k = kneedle.knee
                return optimal_k
        except:
            pass  # Si KneeLocator échoue, utiliser l'approche simplifiée

    # Approche simplifiée: analyse des différences
    inertia_diffs = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]

    # Normaliser les différences
    if max(inertia_diffs) > 0:
        normalized_diffs = [d / max(inertia_diffs) for d in inertia_diffs]

        # Trouver l'indice où la différence normalisée devient inférieure à 0.2 (heuristique)
        for i, diff in enumerate(normalized_diffs):
            if diff < 0.2:
                return k_range[i]

    # Par défaut, retourner la médiane des k testés
    return (k_min + k_max) // 2

def select_closest_vectors(vectors: np.ndarray, all_indices: List[int], centroid: np.ndarray, max_data: int) -> List[int]:
    """
    Sélectionne les max_data vecteurs les plus proches du centroïde.
    
    Args:
        vectors: Tous les vecteurs disponibles
        all_indices: Indices de tous les vecteurs disponibles
        centroid: Centroïde de référence
        max_data: Nombre maximum de vecteurs à sélectionner
    
    Returns:
        List[int]: Liste des indices des max_data vecteurs les plus proches du centroïde
    """
    # Calculer la similarité entre chaque vecteur et le centroïde
    similarities = np.dot(vectors, centroid)
    
    # Trier les indices par similarité décroissante
    sorted_indices = np.argsort(-similarities)
    
    # Sélectionner les max_data indices les plus proches
    selected_count = min(max_data, len(sorted_indices))
    selected_local_indices = sorted_indices[:selected_count]
    
    # Convertir les indices locaux en indices globaux
    selected_global_indices = [all_indices[i] for i in selected_local_indices]
    
    return selected_global_indices

def build_tree_node(vectors: np.ndarray, global_vectors: np.ndarray, global_indices: List[int], 
                   level: int, max_depth: int, k_adaptive: bool, k: int, k_min: int, k_max: int, 
                   max_leaf_size: int, max_data: int, use_gpu: bool = False) -> TreeNode:
    """
    Construit un nœud de l'arbre optimisé.

    Args:
        vectors: Les vecteurs assignés à ce nœud
        global_vectors: Tous les vecteurs de la collection complète
        global_indices: Tous les indices de la collection complète
        level: Niveau actuel dans l'arbre
        max_depth: Profondeur maximale de l'arbre
        k_adaptive: Utiliser la méthode du coude pour déterminer k
        k: Nombre fixe de clusters (si k_adaptive=False)
        k_min, k_max: Limites pour k adaptatif
        max_leaf_size: Taille maximale d'une feuille pour l'arrêt de la subdivision
        max_data: MAX_DATA - Nombre de vecteurs les plus proches à stocker dans chaque feuille
        use_gpu: Utiliser le GPU pour K-means

    Returns:
        TreeNode: Un nœud de l'arbre
    """
    # Récupérer les indices actuels des vecteurs dans ce nœud
    local_indices = list(range(len(vectors)))

    # Calculer le centroïde de ce nœud
    centroid = np.mean(vectors, axis=0)

    # Créer une feuille si les conditions sont remplies
    if level >= max_depth or len(vectors) <= max_leaf_size:
        leaf = TreeNode(centroid, level)

        # Sélectionner les max_data vecteurs les plus proches du centroïde parmi TOUS les vecteurs
        leaf.indices = select_closest_vectors(global_vectors, global_indices, centroid, max_data)

        return leaf

    # Déterminer le nombre de clusters à utiliser
    if k_adaptive:
        if len(vectors) > 10000:
            sample_size = 10000
            sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
            sample_vectors = vectors[sample_indices]
            used_k = find_optimal_k(sample_vectors, k_min, k_max, gpu=use_gpu)
        else:
            used_k = find_optimal_k(vectors, k_min, k_max, gpu=use_gpu)
    else:
        used_k = k

    # Appliquer K-means pour obtenir k clusters avec FAISS
    if level <= 2:  # Afficher la progression seulement pour les premiers niveaux
        print(f"  → Niveau {level+1}: Clustering K-means avec k={used_k} sur {len(vectors):,} vecteurs...")
    centroids, labels = kmeans_faiss(vectors, used_k, gpu=use_gpu)

    # Créer le noeud actuel avec son centroïde
    node = TreeNode(centroid, level)

    # Créer des groupes pour chaque cluster
    cluster_groups = [[] for _ in range(used_k)]
    cluster_indices = [[] for _ in range(used_k)]

    # Assigner chaque vecteur à son cluster
    for i, cluster_idx in enumerate(labels):
        cluster_groups[cluster_idx].append(vectors[i])
        cluster_indices[cluster_idx].append(local_indices[i])

    # Construire les enfants pour chaque cluster (récursivement)
    for i in range(used_k):
        if len(cluster_groups[i]) > 0:
            cluster_vectors = np.array(cluster_groups[i])
            # Pour les nœuds internes, passer les vecteurs et indices actuels
            child = build_tree_node(
                cluster_vectors,
                global_vectors,
                global_indices,
                level + 1,
                max_depth,
                k_adaptive,
                k,
                k_min,
                k_max,
                max_leaf_size,
                max_data,
                use_gpu
            )
            node.children.append(child)
        else:
            # Si un cluster est vide, créer un nœud vide
            empty_node = TreeNode(centroids[i], level + 1)
            node.children.append(empty_node)

    return node

def process_cluster(i: int, vectors: np.ndarray, cluster_indices: List[int], cluster_vectors: np.ndarray, 
                   global_vectors: np.ndarray, global_indices: List[int], level: int, max_depth: int, 
                   k_adaptive: bool, k: int, k_min: int, k_max: int, max_leaf_size: int, max_data: int, 
                   use_gpu: bool = False) -> Tuple[int, Optional[TreeNode]]:
    """
    Fonction pour traiter un cluster en parallèle.
    
    Args:
        i: Indice du cluster
        vectors: Tous les vecteurs
        cluster_indices: Indices des vecteurs dans ce cluster
        cluster_vectors: Vecteurs dans ce cluster
        global_vectors: Tous les vecteurs de la collection complète
        global_indices: Tous les indices de la collection complète
        level: Niveau actuel dans l'arbre
        max_depth: Profondeur maximale de l'arbre
        k_adaptive: Utiliser la méthode du coude pour déterminer k
        k: Nombre fixe de clusters (si k_adaptive=False)
        k_min, k_max: Limites pour k adaptatif
        max_leaf_size: Taille maximale d'une feuille pour l'arrêt de la subdivision
        max_data: MAX_DATA - Nombre de vecteurs les plus proches à stocker dans chaque feuille
        use_gpu: Utiliser le GPU pour K-means
    
    Returns:
        Tuple[int, Optional[TreeNode]]: (indice du cluster, nœud construit)
    """
    if len(cluster_vectors) > 0:
        return i, build_tree_node(
            cluster_vectors,
            global_vectors,
            global_indices,
            level,
            max_depth,
            k_adaptive,
            k,
            k_min,
            k_max,
            max_leaf_size,
            max_data,
            use_gpu
        )
    else:
        # Si un cluster est vide, retourner None
        return i, None

def build_tree(vectors: np.ndarray, max_depth: int = 6, k: int = 16, k_adaptive: bool = False, 
              k_min: int = 2, k_max: int = 32, max_leaf_size: int = 100, max_data: int = 3000, 
              max_workers: Optional[int] = None, use_gpu: bool = False) -> TreeNode:
    """
    Construit un arbre à profondeur variable avec 'k' branches à chaque niveau,
    et pré-calcule les max_data indices les plus proches pour chaque feuille.

    Args:
        vectors: Les vecteurs à indexer
        max_depth: Profondeur maximale de l'arbre
        k: Nombre de branches par nœud (si k_adaptive=False)
        k_adaptive: Utiliser la méthode du coude pour déterminer k automatiquement
        k_min: Nombre minimum de clusters pour k adaptatif
        k_max: Nombre maximum de clusters pour k adaptatif
        max_leaf_size: Taille maximale d'une feuille pour l'arrêt de la subdivision
        max_data: MAX_DATA - Nombre de vecteurs les plus proches à stocker dans chaque feuille
        max_workers: Nombre maximum de processus parallèles
        use_gpu: Utiliser le GPU pour K-means si disponible

    Returns:
        TreeNode: Racine de l'arbre
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    # Vérifier si le GPU est vraiment disponible
    gpu_available = use_gpu and FAISS_AVAILABLE and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0
    if use_gpu and not gpu_available:
        print("⚠️ GPU demandé mais non disponible. Utilisation du CPU à la place.")
        use_gpu = False

    gpu_str = "GPU" if use_gpu else "CPU"
    print(f"⏳ Construction de l'arbre optimisé avec FAISS sur {gpu_str} (max {max_depth} niveaux) avec {'k adaptatif' if k_adaptive else f'k={k}'},")
    print(f"   max_leaf_size={max_leaf_size}, MAX_DATA={max_data}, max_workers={max_workers}")

    start_time = time.time()

    # Statistiques sur les feuilles
    leaf_sizes = []
    leaf_depths = []

    # Initialiser les indices globaux
    global_indices = list(range(len(vectors)))

    # Déterminer le nombre de clusters pour le premier niveau
    if k_adaptive:
        if len(vectors) > 10000:
            sample_size = 10000
            sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
            sample_vectors = vectors[sample_indices]
            root_k = find_optimal_k(sample_vectors, k_min, k_max, gpu=use_gpu)
        else:
            root_k = find_optimal_k(vectors, k_min, k_max, gpu=use_gpu)
    else:
        root_k = k

    # Appliquer K-means pour le premier niveau avec FAISS
    print(f"  → Niveau 1: Clustering K-means avec k={root_k} sur {len(vectors):,} vecteurs...")
    centroids, labels = kmeans_faiss(vectors, root_k, gpu=use_gpu)

    # Créer le noeud racine
    root = TreeNode(np.mean(centroids, axis=0), 0)

    # Créer des groupes pour chaque cluster
    cluster_groups = [[] for _ in range(root_k)]
    cluster_indices = [[] for _ in range(root_k)]

    # Assigner chaque vecteur à son cluster
    for i, cluster_idx in enumerate(labels):
        cluster_groups[cluster_idx].append(vectors[i])
        cluster_indices[cluster_idx].append(global_indices[i])

    # Convertir les listes en tableaux numpy
    cluster_vectors = []
    for i in range(root_k):
        if len(cluster_groups[i]) > 0:
            cluster_vectors.append(np.array(cluster_groups[i]))
        else:
            cluster_vectors.append(np.array([]))

    # Construction parallèle des sous-arbres de premier niveau
    print(f"  → Construction parallèle des sous-arbres avec {max_workers} workers...")

    # Créer la liste des arguments pour chaque cluster
    tasks = []
    for i in range(root_k):
        if len(cluster_groups[i]) > 0:
            tasks.append((
                i,
                vectors,  # Pas utilisé directement dans le traitement des clusters
                cluster_indices[i],
                cluster_vectors[i],
                vectors,  # Vecteurs globaux pour pré-calculer les indices proches
                global_indices,  # Indices globaux pour pré-calculer les indices proches
                1,  # Niveau des sous-arbres (1)
                max_depth,
                k_adaptive,
                k,
                k_min,
                k_max,
                max_leaf_size,
                max_data,
                use_gpu
            ))
    
    # Exécuter les tâches en parallèle avec joblib
    results = Parallel(n_jobs=max_workers, verbose=10)(
        delayed(process_cluster)(*task) for task in tasks
    )
    
    # Reconstruire l'arbre à partir des résultats
    for i, child in results:
        if child is not None:
            root.children.append(child)
        else:
            # Si un cluster est vide, créer un nœud vide
            empty_node = TreeNode(centroids[i], 1)
            root.children.append(empty_node)
    
    # Collecter des statistiques sur les feuilles
    def collect_leaf_stats(node, level=0):
        if not node.children:  # C'est une feuille
            leaf_sizes.append(len(node.indices))
            leaf_depths.append(level)
        else:
            for child in node.children:
                collect_leaf_stats(child, level + 1)
    
    collect_leaf_stats(root)
    
    elapsed = time.time() - start_time

    # Afficher des statistiques sur les feuilles
    if leaf_sizes:
        avg_size = sum(leaf_sizes) / len(leaf_sizes)
        min_size = min(leaf_sizes)
        max_size = max(leaf_sizes)
        avg_depth = sum(leaf_depths) / len(leaf_depths)
        min_depth = min(leaf_depths)
        max_depth_seen = max(leaf_depths)

        print(f"✓ Construction de l'arbre optimisé terminée en {elapsed:.2f}s")
        print(f"  → Statistiques des feuilles:")
        print(f"     - Nombre de feuilles   : {len(leaf_sizes):,}")
        print(f"     - Taille moyenne       : {avg_size:.1f} vecteurs")
        print(f"     - Taille minimale      : {min_size} vecteurs")
        print(f"     - Taille maximale      : {max_size} vecteurs")
        print(f"     - Profondeur moyenne   : {avg_depth:.1f}")
        print(f"     - Profondeur minimale  : {min_depth}")
        print(f"     - Profondeur maximale  : {max_depth_seen}")
    else:
        print(f"✓ Construction de l'arbre optimisé terminée en {elapsed:.2f}s")

    return root
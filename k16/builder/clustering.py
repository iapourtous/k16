"""
Module de clustering pour K16.
Implémente les algorithmes de clustering utilisés pour construire l'arbre K16.
Optimisé pour les instructions SIMD AVX2/AVX-512.
"""

import numpy as np
import time
import platform
import os
from typing import Tuple, List, Dict, Any, Optional, Union
import multiprocessing
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import faiss

try:
    from kneed import KneeLocator
    KNEEDLE_AVAILABLE = True
except ImportError:
    KNEEDLE_AVAILABLE = False
    print("⚠️ Module kneed n'est pas installé. La méthode du coude utilisera une approche simplifiée.")

from k16.core.tree import TreeNode
from k16.utils.optimization import configure_simd

# Détection des capacités SIMD
def optimize_for_simd():
    """Configure l'environnement pour maximiser l'utilisation des instructions SIMD."""
    # Appliquer les optimisations de base
    configure_simd()
    
    # Détection des capacités SIMD
    cpu_info = platform.processor().lower()
    has_avx2 = "avx2" in cpu_info
    has_avx512 = "avx512" in cpu_info

    # Afficher les informations sur les optimisations SIMD détectées
    if has_avx512:
        print("✓ Optimisation pour AVX-512 (512 bits)")
    elif has_avx2:
        print("✓ Optimisation pour AVX2 (256 bits)")
    else:
        print("✓ Optimisation pour SSE (128 bits)")

    # Vérifier si on a un nombre de clusters optimal pour SIMD
    if platform.machine() in ('x86_64', 'AMD64', 'x86'):
        print("✓ Architecture x86_64 détectée - k=16 est optimal pour SIMD")

    return has_avx2, has_avx512

# Appliquer les optimisations SIMD au démarrage
HAS_AVX2, HAS_AVX512 = optimize_for_simd()

def kmeans_faiss(vectors: np.ndarray, k: int, gpu: bool = False, niter: int = 5, remove_empty: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means sphérique optimisé pour les embeddings normalisés.
    Utilise la similarité cosinus (produit scalaire) au lieu de la distance euclidienne.

    Args:
        vectors: Les vecteurs à clusteriser (shape: [n, d])
        k: Nombre de clusters
        gpu: Utiliser le GPU si disponible
        niter: Nombre d'itérations
        remove_empty: Si True, supprime les clusters vides et ajuste k en conséquence

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

    # Utiliser l'implémentation K-means optimisée de FAISS
    try:
        # Créer l'objet K-means FAISS
        kmeans = faiss.Kmeans(d, k, niter=niter, verbose=False, spherical=True, gpu=gpu)

        # Entraîner le modèle
        kmeans.train(vectors)

        # Obtenir les centroïdes
        centroids = kmeans.centroids.copy()

        # Assigner les points aux clusters
        index = faiss.IndexFlatIP(d)
        if gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.GpuIndexFlatIP(res, d)
            except:
                pass  # Fallback au CPU

        index.add(centroids)
        _, labels = index.search(vectors, 1)
        labels = labels.reshape(-1)

        # Gérer les clusters vides si demandé
        if remove_empty:
            cluster_sizes = np.bincount(labels, minlength=k)
            empty_clusters = np.where(cluster_sizes == 0)[0]

            if len(empty_clusters) > 0 and len(empty_clusters) < k - 1:
                print(f"  ⚠️ Suppression de {len(empty_clusters)} clusters vides...")

                # Garder seulement les centroïdes des clusters non vides
                valid_indices = np.where(cluster_sizes > 0)[0]
                new_k = len(valid_indices)

                if new_k >= 2:
                    # Filtrer les centroïdes
                    final_centroids = centroids[valid_indices]

                    # Ré-étiqueter les points
                    index.reset()
                    index.add(final_centroids)
                    _, final_labels = index.search(vectors, 1)
                    final_labels = final_labels.reshape(-1)

                    print(f"  ✓ K réduit de {k} à {new_k} pour éviter les clusters vides")
                    return final_centroids, final_labels

        return centroids, labels

    except Exception as e:
        # Fallback vers sklearn si FAISS échoue
        print(f"⚠️ FAISS K-means failed ({e}), using sklearn fallback")
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import normalize

        # Normaliser les vecteurs
        vectors_norm = normalize(vectors, norm='l2')

        # K-means avec sklearn
        kmeans = KMeans(n_clusters=k, max_iter=niter, n_init=1, random_state=42)
        labels = kmeans.fit_predict(vectors_norm)

        # Vérifier les clusters vides
        if remove_empty:
            cluster_sizes = np.bincount(labels, minlength=k)
            empty_clusters = np.where(cluster_sizes == 0)[0]

            if len(empty_clusters) > 0 and len(empty_clusters) < k-1:
                print(f"  ⚠️ Suppression de {len(empty_clusters)} clusters vides (version CPU)...")

                # Garder seulement les centroïdes des clusters non vides
                valid_indices = np.where(cluster_sizes > 0)[0]
                new_k = len(valid_indices)

                # Relabel les points
                new_labels = np.zeros_like(labels)
                for new_idx, old_idx in enumerate(valid_indices):
                    new_labels[labels == old_idx] = new_idx

                # Extraire les centroïdes valides
                valid_centroids = kmeans.cluster_centers_[valid_indices]

                # Normaliser les centroïdes
                centroids = normalize(valid_centroids, norm='l2')
                labels = new_labels

                print(f"  ✓ K réduit de {k} à {new_k} pour éviter les clusters vides")
                return centroids, labels

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

    # Si k = n, simplement retourner tous les vecteurs comme centroïdes
    if k >= n:
        # Copier les vecteurs pour éviter les modifications accidentelles
        return vectors.copy()

    centroids = np.zeros((k, d), dtype=np.float32)

    # Choisir le premier centroïde aléatoirement
    centroids[0] = vectors[np.random.randint(0, n)]

    # Utiliser FAISS pour accélérer
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
        # Éviter la division par zéro si toutes les distances sont nulles
        if np.sum(probabilities) > 0:
            probabilities = probabilities / np.sum(probabilities)
        else:
            # Si toutes les distances sont nulles, choisir uniformément
            probabilities = np.ones(n) / n

        # Sélectionner le prochain centroïde
        cumulative_probs = np.cumsum(probabilities)
        r = np.random.rand()

        # S'assurer que l'indice est valide
        if cumulative_probs[-1] > 0:
            idx = np.searchsorted(cumulative_probs, r)
            if idx >= n:  # Pour éviter les erreurs d'indexation
                idx = n - 1
        else:
            # En cas de problème avec les probabilités, choisir un indice aléatoire
            idx = np.random.randint(0, n)

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
        centroids, labels = kmeans_faiss(vectors, k, gpu=gpu, niter=5)  # Moins d'itérations pour la recherche

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

def select_closest_vectors(vectors: np.ndarray, all_indices: List[int], centroid: np.ndarray, max_data: int) -> np.ndarray:
    """
    Sélectionne les max_data vecteurs les plus proches du centroïde.

    Args:
        vectors: Tous les vecteurs disponibles
        all_indices: Indices de tous les vecteurs disponibles
        centroid: Centroïde de référence
        max_data: Nombre maximum de vecteurs à sélectionner

    Returns:
        np.ndarray: Tableau des indices des max_data vecteurs les plus proches du centroïde
    """
    # Calculer la similarité entre chaque vecteur et le centroïde
    similarities = np.dot(vectors, centroid)

    # Trier les indices par similarité décroissante
    sorted_indices = np.argsort(-similarities)

    # Sélectionner les max_data indices les plus proches
    selected_count = min(max_data, len(sorted_indices))
    selected_local_indices = sorted_indices[:selected_count]

    # Convertir les indices locaux en indices globaux
    if isinstance(all_indices, list):
        all_indices = np.array(all_indices, dtype=np.int32)
    
    selected_global_indices = all_indices[selected_local_indices]

    return selected_global_indices

def select_closest_natural_vectors(leaf_vectors: np.ndarray, leaf_indices: Union[List[int], np.ndarray],
                                  global_vectors: np.ndarray, global_indices: Union[List[int], np.ndarray],
                                  centroid: np.ndarray, max_data: int, sibling_indices: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """
    Sélectionne d'abord les vecteurs qui tombent naturellement dans la feuille,
    puis complète avec les frères/sœurs, et enfin avec les plus proches au global si nécessaire.

    Args:
        leaf_vectors: Vecteurs qui tombent naturellement dans cette feuille
        leaf_indices: Indices des vecteurs qui tombent naturellement dans cette feuille
        global_vectors: Tous les vecteurs disponibles
        global_indices: Tous les indices disponibles
        centroid: Centroïde de référence
        max_data: Nombre maximum de vecteurs à sélectionner
        sibling_indices: Liste des indices des frères/sœurs (autres clusters du même niveau)

    Returns:
        np.ndarray: Tableau des indices des max_data vecteurs les plus proches du centroïde
    """
    # Conversion en arrays numpy si nécessaire
    if isinstance(leaf_indices, list):
        leaf_indices = np.array(leaf_indices, dtype=np.int32)
    if isinstance(global_indices, list):
        global_indices = np.array(global_indices, dtype=np.int32)
    
    # D'abord, prioriser les vecteurs qui tombent naturellement dans la feuille
    leaf_indices_set = set(leaf_indices.tolist())

    # Si la feuille contient déjà assez de vecteurs, pas besoin d'en chercher d'autres
    if len(leaf_indices) >= max_data:
        # Calculer la similarité entre les vecteurs de la feuille et le centroïde
        similarities = np.dot(leaf_vectors, centroid)
        # Trier les indices par similarité décroissante
        sorted_local_indices = np.argsort(-similarities)
        # Prendre les max_data plus proches
        selected_count = min(max_data, len(sorted_local_indices))
        return leaf_indices[sorted_local_indices[:selected_count]]

    # Sinon, compléter avec les frères/sœurs d'abord, puis au global
    result = leaf_indices.tolist()  # Commencer avec les vecteurs naturels
    used_indices = set(leaf_indices.tolist())  # Maintenir un set de tous les indices utilisés

    # Étape 1: Ajouter des vecteurs des frères/sœurs si disponibles
    if sibling_indices is not None:
        sibling_candidates = []
        for sibling_idx_array in sibling_indices:
            if len(sibling_idx_array) > 0:
                sibling_candidates.extend(sibling_idx_array.tolist())

        if sibling_candidates:
            # Calculer la similarité des vecteurs frères/sœurs avec le centroïde
            sibling_candidates = np.array(sibling_candidates, dtype=np.int32)
            sibling_vectors = global_vectors[sibling_candidates]
            sibling_similarities = np.dot(sibling_vectors, centroid)
            sibling_sorted_indices = np.argsort(-sibling_similarities)

            # Ajouter les meilleurs frères/sœurs qui ne sont pas déjà utilisés
            for idx in sibling_sorted_indices:
                global_idx = sibling_candidates[idx]
                if global_idx not in used_indices:
                    result.append(global_idx)
                    used_indices.add(global_idx)  # Mettre à jour le set d'exclusion
                    if len(result) >= max_data:
                        return np.array(result, dtype=np.int32)

    # Étape 2: Si encore pas assez, compléter avec les plus proches au global
    # Calculer la similarité entre tous les vecteurs et le centroïde
    similarities = np.dot(global_vectors, centroid)

    # Trier les indices par similarité décroissante
    sorted_indices = np.argsort(-similarities)

    # Ajouter les vecteurs les plus proches qui ne sont pas déjà utilisés
    for i in sorted_indices:
        global_idx = global_indices[i]
        if global_idx not in used_indices:
            result.append(global_idx)
            used_indices.add(global_idx)  # Mettre à jour le set d'exclusion
            if len(result) >= max_data:
                break

    return np.array(result, dtype=np.int32)

def build_tree_node(vectors: np.ndarray, current_indices: Union[List[int], np.ndarray], global_vectors: np.ndarray, global_indices: Union[List[int], np.ndarray],
                   level: int, max_depth: int, k_adaptive: bool, k: int, k_min: int, k_max: int,
                   max_leaf_size: int, max_data: int, use_gpu: bool = False, sibling_indices: Optional[List[np.ndarray]] = None) -> TreeNode:
    """
    Construit un nœud de l'arbre optimisé.

    Args:
        vectors: Les vecteurs assignés à ce nœud
        current_indices: Les indices des vecteurs assignés à ce nœud
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
    # Conversion en arrays numpy si nécessaire
    if isinstance(global_indices, list):
        global_indices = np.array(global_indices, dtype=np.int32)
    if isinstance(current_indices, list):
        current_indices = np.array(current_indices, dtype=np.int32)

    # Vérifier si nous avons assez de vecteurs pour continuer
    if len(vectors) == 0:
        # Créer un nœud vide avec un centroïde aléatoire normalisé
        random_vector = np.random.randn(global_vectors.shape[1])
        random_centroid = random_vector / np.linalg.norm(random_vector)
        empty_node = TreeNode(level=level)
        empty_node.centroid = random_centroid
        return empty_node

    # Calculer le centroïde de ce nœud (normalisé pour le produit scalaire)
    centroid = np.mean(vectors, axis=0)
    # Normaliser le centroïde pour la similarité cosinus
    norm = np.linalg.norm(centroid)
    if norm > 0:  # Éviter la division par zéro
        centroid = centroid / norm
    else:
        # Si le centroïde est un vecteur nul (cas rare), créer un vecteur aléatoire normalisé
        random_vector = np.random.randn(vectors.shape[1])
        centroid = random_vector / np.linalg.norm(random_vector)

    # Créer une feuille si les conditions sont remplies
    if level >= max_depth or len(vectors) <= max_leaf_size:
        leaf = TreeNode(level=level)
        leaf.centroid = centroid

        # Sélectionner les max_data vecteurs les plus proches du centroïde
        # Priorité aux vecteurs qui tombent naturellement dans cette feuille, puis frères/sœurs, puis global
        leaf.set_indices(select_closest_natural_vectors(
            vectors, current_indices,
            global_vectors, global_indices,
            centroid, max_data, sibling_indices
        ))

        return leaf

    # Déterminer le nombre de clusters à utiliser
    if k_adaptive:
        if len(vectors) > 10000:
            sample_size = min(10000, len(vectors))
            sample_indices = np.random.choice(len(vectors), sample_size, replace=False)
            sample_vectors = vectors[sample_indices]
            used_k = find_optimal_k(sample_vectors, k_min, k_max, gpu=use_gpu)
        else:
            used_k = find_optimal_k(vectors, k_min, k_max, gpu=use_gpu)
    else:
        used_k = min(k, len(vectors))  # Ne pas créer plus de clusters que de vecteurs

    # Afficher la progression seulement pour les premiers niveaux
    if level <= 2:
        print(f"  → Niveau {level+1}: Clustering K-means avec k={used_k} sur {len(vectors):,} vecteurs...")

    # Appliquer K-means pour obtenir k clusters avec FAISS
    # En activant la suppression des clusters vides
    centroids, labels = kmeans_faiss(vectors, used_k, gpu=use_gpu, remove_empty=True)

    # Ajuster used_k au nombre réel de clusters après suppression des vides
    used_k = len(centroids)

    # Vérifier que les centroïdes sont normalisés
    for i in range(len(centroids)):
        norm = np.linalg.norm(centroids[i])
        if norm > 0:  # Éviter la division par zéro
            centroids[i] = centroids[i] / norm

    # Créer le noeud actuel avec son centroïde
    node = TreeNode(level=level)
    node.centroid = centroid

    # Créer des groupes pour chaque cluster en utilisant numpy
    cluster_indices = [[] for _ in range(used_k)]
    cluster_vectors = [[] for _ in range(used_k)]

    # Assigner chaque vecteur à son cluster
    for i, cluster_idx in enumerate(labels):
        cluster_vectors[cluster_idx].append(vectors[i])
        cluster_indices[cluster_idx].append(current_indices[i])
    
    # Convertir les listes en tableaux numpy
    cluster_vectors_np = [np.array(vec_list) if vec_list else np.array([]) for vec_list in cluster_vectors]
    cluster_indices_np = [np.array(idx_list, dtype=np.int32) if idx_list else np.array([], dtype=np.int32) for idx_list in cluster_indices]

    # Construire les enfants pour chaque cluster (récursivement)
    for i in range(used_k):
        if len(cluster_vectors_np[i]) > 0:
            # Préparer les indices des frères/sœurs (tous les autres clusters du même niveau)
            sibling_indices_list = [cluster_indices_np[j] for j in range(used_k) if j != i and len(cluster_indices_np[j]) > 0]

            # Pour les nœuds internes, passer les vecteurs et indices actuels
            child = build_tree_node(
                cluster_vectors_np[i],
                cluster_indices_np[i],
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
                use_gpu,
                sibling_indices_list
            )
            node.add_child(child)
        else:
            # Si un cluster est vide, créer une feuille basée sur le centroïde
            empty_centroid = centroids[i]
            empty_node = TreeNode(level=level + 1)
            empty_node.centroid = empty_centroid

            # Sélectionner les vecteurs globaux les plus proches de ce centroïde vide
            similarities = np.dot(global_vectors, empty_centroid)
            sorted_indices = np.argsort(-similarities)

            # Sélectionner jusqu'à max_data indices les plus proches
            selected_count = min(max_data, len(sorted_indices))
            empty_node.set_indices(global_indices[sorted_indices[:selected_count]])

            node.add_child(empty_node)

    # Bien que node.centroids soit déjà mis à jour dans add_child,
    # on s'assure qu'il contient tous les centroïdes alignés avec children
    node.set_children_centroids()
    
    return node

def process_cluster(i: int, vectors: np.ndarray, cluster_indices: np.ndarray, cluster_vectors: np.ndarray,
                   global_vectors: np.ndarray, global_indices: np.ndarray, level: int, max_depth: int,
                   k_adaptive: bool, k: int, k_min: int, k_max: int, max_leaf_size: int, max_data: int,
                   centroids: np.ndarray, use_gpu: bool = False) -> Tuple[int, Optional[TreeNode]]:
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
        centroids: Les centroïdes de tous les clusters
        use_gpu: Utiliser le GPU pour K-means

    Returns:
        Tuple[int, Optional[TreeNode]]: (indice du cluster, nœud construit)
    """
    if len(cluster_vectors) > 0:
        return i, build_tree_node(
            cluster_vectors,
            cluster_indices,
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
        # Si un cluster est vide, créer une feuille basée sur le centroïde
        # Normaliser le centroïde pour la similarité cosinus
        centroid = centroids[i]
        centroid = centroid / np.linalg.norm(centroid)

        # Sélectionner les vecteurs globaux les plus proches de ce centroïde
        similarities = np.dot(global_vectors, centroid)
        sorted_indices = np.argsort(-similarities)

        # Créer une feuille avec le centroïde normalisé
        leaf = TreeNode(level=level)
        leaf.centroid = centroid

        # Sélectionner jusqu'à max_data indices les plus proches
        selected_count = min(max_data, len(sorted_indices))
        leaf.set_indices(global_indices[sorted_indices[:selected_count]])

        return i, leaf

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
    gpu_available = False
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            # Test simple de création d'un index GPU
            test_index = faiss.GpuIndexFlatIP(res, 128)
            gpu_available = True
            print("✓ GPU FAISS détecté et disponible")
        except (AttributeError, RuntimeError) as e:
            print(f"⚠️ GPU demandé mais non disponible: {e}. Utilisation du CPU à la place.")
            use_gpu = False

    gpu_str = "GPU" if use_gpu else "CPU"
    print(f"⏳ Construction de l'arbre optimisé avec FAISS sur {gpu_str} (max {max_depth} niveaux) avec {'k adaptatif' if k_adaptive else f'k={k}'},")
    print(f"   max_leaf_size={max_leaf_size}, MAX_DATA={max_data}, max_workers={max_workers}")

    start_time = time.time()

    # Statistiques sur les feuilles
    leaf_sizes = []
    leaf_depths = []

    # Initialiser les indices globaux comme un tableau numpy
    global_indices = np.arange(len(vectors), dtype=np.int32)

    # Utiliser k=16 par défaut avec ajustement intelligent
    # pour les petits datasets ou les niveaux profonds
    vectors_count = len(vectors)
    min_points_per_cluster = max(1, max_leaf_size // 2)

    # Détermine le nombre optimal de clusters (k)
    if k_adaptive:
        # Mode adaptatif traditionnel (coude)
        if vectors_count > 10000:
            sample_size = 10000
            sample_indices = np.random.choice(vectors_count, sample_size, replace=False)
            sample_vectors = vectors[sample_indices]
            root_k = find_optimal_k(sample_vectors, k_min, k_max, gpu=use_gpu)
        else:
            root_k = find_optimal_k(vectors, k_min, k_max, gpu=use_gpu)
    else:
        # Ajustement automatique - k est réduit si le nombre de points est trop petit
        # k=16 est optimal pour SIMD, mais pas si on a trop peu de points
        if vectors_count < k * min_points_per_cluster:
            # Calculer un k qui garantit au moins min_points_per_cluster points par cluster
            adjusted_k = max(2, min(k, vectors_count // min_points_per_cluster))
            if adjusted_k != k:
                print(f"  → Ajustement automatique: k={k} → k={adjusted_k} (car {vectors_count} points < {k}×{min_points_per_cluster})")
            root_k = adjusted_k
        else:
            # Utiliser k=16 standard (optimal pour SIMD)
            root_k = k

    # Appliquer K-means pour le premier niveau avec FAISS
    print(f"  → Niveau 1: Clustering K-means avec k={root_k} sur {len(vectors):,} vecteurs...")
    centroids, labels = kmeans_faiss(vectors, root_k, gpu=use_gpu, remove_empty=True)

    # Ajuster root_k après suppression des clusters vides
    root_k = len(centroids)

    # Créer le noeud racine (avec centroïde normalisé)
    root_centroid = np.mean(centroids, axis=0)
    root_centroid = root_centroid / np.linalg.norm(root_centroid)
    root = TreeNode(level=0)
    root.centroid = root_centroid

    # Créer des groupes pour chaque cluster en utilisant numpy
    # Utiliser une liste de tableaux pour améliorer l'efficacité
    cluster_vectors = [[] for _ in range(root_k)]
    cluster_indices = [[] for _ in range(root_k)]

    # Assigner chaque vecteur à son cluster
    for i, cluster_idx in enumerate(labels):
        cluster_vectors[cluster_idx].append(vectors[i])
        cluster_indices[cluster_idx].append(global_indices[i])
    
    # Convertir en tableaux numpy
    cluster_vectors_np = [np.array(vec_list) if vec_list else np.array([]) for vec_list in cluster_vectors]
    cluster_indices_np = [np.array(idx_list, dtype=np.int32) if idx_list else np.array([], dtype=np.int32) for idx_list in cluster_indices]

    # Construction parallèle des sous-arbres de premier niveau
    print(f"  → Construction parallèle des sous-arbres avec {max_workers} workers...")

    # Créer la liste des arguments pour chaque cluster
    tasks = []
    for i in range(root_k):
        tasks.append((
            i,
            vectors,  # Pas utilisé directement dans le traitement des clusters
            cluster_indices_np[i] if i < len(cluster_indices_np) else np.array([], dtype=np.int32),
            cluster_vectors_np[i] if i < len(cluster_vectors_np) else np.array([]),
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
            centroids,  # Passer les centroïdes pour traiter les clusters vides
            use_gpu
        ))

    # Exécuter les tâches en parallèle avec joblib
    results = Parallel(n_jobs=max_workers, verbose=10)(
        delayed(process_cluster)(*task) for task in tasks
    )

    # Reconstruire l'arbre à partir des résultats
    for i, child in results:
        if child is not None:
            root.add_child(child)
        else:
            # Ce cas ne devrait pas arriver avec le nouveau process_cluster
            # mais gardons-le par précaution
            empty_node = TreeNode(level=1)
            empty_node.centroid = centroids[i]
            root.add_child(empty_node)
    
    # S'assurer que root.centroids est bien initialisé
    root.set_children_centroids()
    
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
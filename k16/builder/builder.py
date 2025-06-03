"""
Constructeur simplifié d'arbres K16 optimisés.
Une seule fonction qui fait tout: construction, réduction, HNSW, optimisation.
"""

import os
from typing import Optional, Union, Dict, Any
import numpy as np

from k16.utils.config import ConfigManager
from k16.core.tree import K16Tree, TreeNode
from k16.core.flat_tree import TreeFlat
from k16.io.reader import VectorReader
from k16.utils.optimization import configure_simd

def build_optimized_tree(
    vectors: Union[np.ndarray, str],
    output_file: str,
    config: Optional[Dict[str, Any]] = None,
    max_depth: Optional[int] = None,
    max_leaf_size: Optional[int] = None,
    max_data: Optional[int] = None,
    max_dims: Optional[int] = None,
    use_hnsw: Optional[bool] = None,
    k: Optional[int] = None,
    k_adaptive: Optional[bool] = None,
    verbose: bool = True
) -> TreeFlat:
    """
    Construit un arbre K16 optimisé en une seule fonction.

    Cette fonction fait tout:
    1. Chargement des vecteurs si nécessaire
    2. Construction de l'arbre hiérarchique avec k=16 par défaut
    3. Réduction dimensionnelle
    4. Amélioration HNSW si activée
    5. Optimisation des centroïdes (économie ~56% d'espace)
    6. Sauvegarde de l'arbre

    Args:
        vectors: Soit un tableau numpy de vecteurs, soit un chemin vers un fichier de vecteurs
        output_file: Chemin vers le fichier de sortie pour sauvegarder l'arbre
        config: Configuration personnalisée (facultatif, sinon utilise config.yaml)
        max_depth: Profondeur maximale de l'arbre (facultatif)
        max_leaf_size: Taille maximale des feuilles (facultatif)
        max_data: Nombre de vecteurs à stocker par feuille (facultatif)
        max_dims: Nombre de dimensions à conserver (facultatif)
        use_hnsw: Activer l'amélioration HNSW (facultatif)
        k: Nombre de branches par nœud (16 par défaut)
        k_adaptive: Utiliser k adaptatif (désactivé par défaut)
        verbose: Afficher les messages de progression

    Returns:
        L'arbre optimisé construit
    """
    # S'assurer que les optimisations SIMD sont configurées
    configure_simd()
    
    # 1. Charger la configuration
    if config is None:
        config_manager = ConfigManager()
        build_config = config_manager.get_section("build_tree")
        flat_config = config_manager.get_section("flat_tree")
    else:
        build_config = config.get("build_tree", {})
        flat_config = config.get("flat_tree", {})
    
    # 2. Utiliser les paramètres explicites ou les valeurs de configuration
    max_depth = max_depth if max_depth is not None else build_config.get("max_depth", 32)
    max_leaf_size = max_leaf_size if max_leaf_size is not None else build_config.get("max_leaf_size", 50)
    max_data = max_data if max_data is not None else build_config.get("max_data", 256)  # Optimisé pour SIMD (multiple de 16)
    max_dims = max_dims if max_dims is not None else flat_config.get("max_dims", 512)  # Optimisé pour SIMD (multiple de 16)
    k_adaptive = k_adaptive if k_adaptive is not None else build_config.get("k_adaptive", False)
    use_hnsw = use_hnsw if use_hnsw is not None else build_config.get("use_hnsw_improvement", True)

    # Autres paramètres de configuration
    k = k if k is not None else build_config.get("k", 16)  # Valeur fixe k=16 par défaut
    k_min = build_config.get("k_min", 2)
    k_max = build_config.get("k_max", 32)
    max_workers = build_config.get("max_workers", None)
    reduction_method = flat_config.get("reduction_method", "directional")
    
    # 3. Préparer les vecteurs
    if isinstance(vectors, str):
        if verbose:
            print(f"⏳ Chargement des vecteurs depuis {vectors}...")
        vectors_reader = VectorReader(file_path=vectors, mode="ram")
        if verbose:
            print(f"✓ {len(vectors_reader):,} vecteurs chargés (dim {vectors_reader.d})")
        vectors_data = vectors_reader.vectors
    else:
        vectors_data = vectors
        vectors_reader = None
    
    # 4. Créer le répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 5. Construction de l'arbre hiérarchique avec k-means
    if verbose:
        print(f"⏳ Construction de l'arbre K16 avec max_depth={max_depth}, "
              f"max_leaf_size={max_leaf_size}, max_data={max_data}, "
              f"{'k adaptatif' if k_adaptive else f'k={k}'}")
              
    # Importation locale pour éviter les dépendances circulaires
    from k16.builder.clustering import build_tree as clustering_build_tree
    
    tree = clustering_build_tree(
        vectors=vectors_data,
        max_depth=max_depth,
        k=k,
        k_adaptive=k_adaptive,
        k_min=k_min,
        k_max=k_max,
        max_leaf_size=max_leaf_size,
        max_data=max_data,
        max_workers=max_workers
    )
    
    # 6. Calcul de la réduction dimensionnelle et conversion en K16Tree
    k16tree = K16Tree(tree)
    
    if verbose:
        print(f"⏳ Calcul de la réduction dimensionnelle (max_dims={max_dims}, method={reduction_method})...")
    
    k16tree.compute_dimensional_reduction(max_dims=max_dims, method=reduction_method)
    
    # 7. Conversion en structure plate
    if verbose:
        print(f"⏳ Conversion en structure plate...")
    
    flat_tree = TreeFlat.from_tree(k16tree)
    k16tree.flat_tree = flat_tree
    
    # 8. Amélioration optionnelle avec HNSW
    if use_hnsw:
        if verbose:
            print(f"⏳ Amélioration avec HNSW...")
        
        if vectors_reader is None:
            # Créer un lecteur de vecteurs si on n'en a pas déjà un
            vectors_reader = VectorReader(vectors=vectors_data)
            
        k16tree = k16tree.improve_with_hnsw(vectors_reader, max_data)
        flat_tree = k16tree.flat_tree
        
        if verbose:
            print(f"✓ Amélioration HNSW terminée!")
    elif verbose:
        print(f"ℹ️ Amélioration HNSW désactivée")
    
    # 9. Sauvegarde de l'arbre (avec optimisation automatique)
    if output_file:
        # Assurer l'extension correcte
        if not (output_file.endswith(".flat.npy") or output_file.endswith(".flat")):
            output_file = os.path.splitext(output_file)[0] + ".flat.npy"
            
        if verbose:
            print(f"⏳ Sauvegarde de l'arbre vers {output_file}...")
            
        flat_tree.save(output_file)  # L'optimisation est faite automatiquement
        
        if verbose:
            print(f"✓ Arbre optimisé sauvegardé dans {output_file}")
    
    # 10. Nettoyer
    if vectors_reader is not None and isinstance(vectors, str):
        vectors_reader.close()
    
    return flat_tree

def build_tree(
    vectors: np.ndarray,
    max_depth: int = 6, 
    max_leaf_size: int = 100, 
    max_data: int = 256,
    max_dims: int = 512,
    use_hnsw: bool = True,
    k: int = 16,
    k_adaptive: bool = False,
    output_file: Optional[str] = None
) -> K16Tree:
    """
    Fonction simplifiée pour construire un arbre K16.
    
    Args:
        vectors: Vecteurs à indexer
        max_depth: Profondeur maximale de l'arbre
        max_leaf_size: Taille maximale d'une feuille pour l'arrêt de la subdivision
        max_data: Nombre de vecteurs à stocker par feuille
        max_dims: Nombre de dimensions à conserver pour la réduction dimensionnelle
        use_hnsw: Activer l'amélioration HNSW
        k: Nombre de branches par nœud
        k_adaptive: Utiliser k adaptatif
        output_file: Chemin vers le fichier de sortie (facultatif)
        
    Returns:
        K16Tree: Arbre K16 construit
    """
    # Utiliser build_optimized_tree avec des paramètres par défaut
    flat_tree = build_optimized_tree(
        vectors=vectors,
        output_file=output_file or "",  # Chaîne vide pour ne pas sauvegarder si None
        max_depth=max_depth,
        max_leaf_size=max_leaf_size,
        max_data=max_data,
        max_dims=max_dims,
        use_hnsw=use_hnsw,
        k=k,
        k_adaptive=k_adaptive,
        verbose=True
    )
    
    # Créer un K16Tree à partir de l'arbre plat
    tree = K16Tree(None)
    tree.flat_tree = flat_tree
    
    return tree
"""
Module d'écriture de vecteurs et d'arbres pour K16.
Fournit des classes optimisées pour sauvegarder des vecteurs et des arbres.
"""

import os
import struct
import time
import numpy as np
from typing import Optional

from k16.core.tree import K16Tree

class VectorWriter:
    """Classe pour écrire des vecteurs dans un fichier binaire."""
    
    @staticmethod
    def write_bin(vectors: np.ndarray, file_path: str) -> None:
        """
        Écrit des vecteurs dans un fichier binaire.
        Format: header (n, d: uint64) suivi des données en float32.
        
        Args:
            vectors: Tableau numpy contenant les vecteurs (shape: [n, d])
            file_path: Chemin du fichier de sortie
        """
        n, d = vectors.shape
        start_time = time.time()
        print(f"⏳ Écriture de {n:,} vecteurs (dim {d}) vers {file_path}...")
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Convertir en float32 si ce n'est pas déjà le cas
        vectors_float32 = vectors.astype(np.float32)
        
        with open(file_path, "wb") as f:
            # Écrire l'entête: nombre de vecteurs (n) et dimension (d)
            f.write(struct.pack("<QQ", n, d))
            # Écrire les données
            f.write(vectors_float32.tobytes())
        
        elapsed = time.time() - start_time
        print(f"✓ {n:,} vecteurs (dim {d}) écrits dans {file_path} [terminé en {elapsed:.2f}s]")

def write_vectors(vectors: np.ndarray, file_path: str) -> None:
    """
    Fonction utilitaire pour écrire des vecteurs dans un fichier.
    
    Args:
        vectors: Tableau numpy contenant les vecteurs (shape: [n, d])
        file_path: Chemin du fichier de sortie
    """
    VectorWriter.write_bin(vectors, file_path)

def write_tree(tree: K16Tree, file_path: str, mmap_dir: bool = False, minimal: bool = True) -> None:
    """
    Sauvegarde un arbre K16 au format plat optimisé.
    
    Args:
        tree: L'arbre K16 à sauvegarder
        file_path: Chemin du fichier de sortie
        mmap_dir: Si True, crée un répertoire de fichiers numpy pour le memory-mapping
        minimal: Si True (par défaut), sauvegarde uniquement les structures essentielles
    """
    # Vérifier que l'arbre a une structure plate
    if tree.flat_tree is None:
        raise ValueError("L'arbre doit avoir une structure plate (flat_tree) pour être sauvegardé")
    
    # Déterminer le chemin de sauvegarde
    if not file_path.endswith('.flat.npy'):
        flat_path = os.path.splitext(file_path)[0] + '.flat.npy'
    else:
        flat_path = file_path
    
    print(f"⏳ Sauvegarde de la structure plate vers {flat_path}...")
    
    # Créer le répertoire si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(flat_path)), exist_ok=True)
    
    # Sauvegarder la structure plate
    tree.flat_tree.save(flat_path, mmap_dir=mmap_dir, minimal=minimal)
    
    print(f"✓ Structure plate sauvegardée vers {flat_path}")
    
    return flat_path
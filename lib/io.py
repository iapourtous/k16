"""
Module d'entrée/sortie pour K16.
Gère la lecture et l'écriture des vecteurs et des arbres.
"""

import os
import struct
import pickle
import time
import numpy as np
import mmap
import sys
from typing import Tuple, List, Dict, Any, Optional, Union
from collections import OrderedDict

from .tree import TreeNode, K16Tree
from .config import ConfigManager

# Custom unpickler to handle legacy TreeNode objects
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect old TreeNode references to our new module
        if name == 'TreeNode' and module == '__main__':
            from lib.tree import TreeNode
            return TreeNode
        return super().find_class(module, name)

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

class VectorReader:
    """
    Classe pour lire des vecteurs depuis un fichier binaire.
    Supporte deux modes: RAM et mmap.
    """
    
    def __init__(self, file_path: str, mode: str = "ram", cache_size_mb: Optional[int] = None):
        """
        Initialise le lecteur de vecteurs.
        
        Args:
            file_path: Chemin vers le fichier binaire contenant les vecteurs
            mode: Mode de lecture - "ram" (défaut) ou "mmap"
            cache_size_mb: Taille du cache en mégaoctets pour le mode mmap (si None, utilise la valeur de ConfigManager)
        """
        self.file_path = file_path
        self.mode = mode.lower()
        
        if self.mode not in ["ram", "mmap"]:
            raise ValueError("Mode doit être 'ram' ou 'mmap'")
        
        # Paramètres des vecteurs
        self.n = 0  # Nombre de vecteurs
        self.d = 0  # Dimension des vecteurs
        self.header_size = 16  # 2 entiers 64 bits
        self.vector_size = 0  # Taille d'un vecteur en octets
        
        # Stockage des vecteurs
        self.vectors = None  # Pour le mode RAM
        self.mmap_file = None  # Pour le mode mmap
        self.mmap_obj = None  # Pour le mode mmap
        
        # Cache LRU pour le mode mmap
        self.cache_enabled = False
        self.cache_size_mb = cache_size_mb
        if self.cache_size_mb is None:
            # Obtenir la valeur par défaut de la configuration
            config_manager = ConfigManager()
            self.cache_size_mb = config_manager.get("search", "cache_size_mb", 500)
        
        self.cache = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 0
        self.cache_capacity = 0
        
        # Charger les vecteurs
        self._load_vectors()
    
    def _load_vectors(self) -> None:
        """Charge les vecteurs selon le mode choisi."""
        start_time = time.time()
        print(f"⏳ Chargement des vecteurs depuis {self.file_path} en mode {self.mode.upper()}...")
        
        with open(self.file_path, "rb") as f:
            # Lire l'en-tête (nombre et dimension des vecteurs)
            self.n, self.d = struct.unpack("<QQ", f.read(self.header_size))
            print(f"  → Format détecté: {self.n:,} vecteurs de dimension {self.d}")
            
            # Calculer la taille d'un vecteur en octets
            self.vector_size = self.d * 4  # float32 = 4 octets
            
            if self.mode == "ram":
                # Mode RAM: charger tous les vecteurs en mémoire
                buffer = f.read(self.n * self.vector_size)
                self.vectors = np.frombuffer(buffer, dtype=np.float32).reshape(self.n, self.d)
            else:
                # Mode mmap: mapper le fichier en mémoire
                self.mmap_file = open(self.file_path, "rb")
                self.mmap_obj = mmap.mmap(self.mmap_file.fileno(), 0, access=mmap.ACCESS_READ)
                
                # Initialiser le cache LRU
                self.max_cache_size = self.cache_size_mb * 1024 * 1024  # Convertir MB en octets
                self.cache = OrderedDict()  # Cache LRU
                self.cache_enabled = True
                
                # Limiter la taille du cache si trop grande
                max_possible_vectors = self.n
                max_possible_size = max_possible_vectors * self.vector_size
                if self.max_cache_size > max_possible_size:
                    adjusted_size = max_possible_size
                    adjusted_mb = adjusted_size / (1024 * 1024)
                    print(f"  → Cache limité à {adjusted_mb:.1f} MB (taille totale des données)")
                    self.max_cache_size = adjusted_size
                
                # Calculer combien de vecteurs peuvent tenir dans le cache
                self.cache_capacity = self.max_cache_size // self.vector_size
                print(f"  → Cache LRU initialisé: {self.cache_size_mb} MB ({self.cache_capacity:,} vecteurs)")
        
        elapsed = time.time() - start_time
        
        if self.mode == "ram":
            memory_usage = f"{self.vectors.nbytes / (1024**2):.1f} MB"
        else:
            memory_usage = f"Mmap + Cache {self.cache_size_mb} MB"
            
        print(f"✓ {self.n:,} vecteurs (dim {self.d}) prêts en mode {self.mode.upper()} [terminé en {elapsed:.2f}s]")
        print(f"  → Mémoire utilisée: {memory_usage}")
    
    def _update_cache_stats(self, force: bool = False) -> bool:
        """
        Affiche les statistiques du cache si assez d'accès ont été effectués.
        
        Args:
            force: Si True, affiche les statistiques même si le seuil n'est pas atteint
            
        Returns:
            bool: True si les statistiques ont été affichées, False sinon
        """
        if not self.cache_enabled:
            return False
            
        total = self.cache_hits + self.cache_misses
        
        # Afficher les stats tous les 100000 accès ou si forcé
        if force or (total > 0 and total % 100000 == 0):
            hit_rate = self.cache_hits / total * 100 if total > 0 else 0
            cache_usage = len(self.cache) / self.cache_capacity * 100 if self.cache_capacity > 0 else 0
            print(f"  → Cache stats: {hit_rate:.1f}% hits, {cache_usage:.1f}% rempli "
                  f"({len(self.cache):,}/{self.cache_capacity:,} vecteurs)")
            return True
        return False
    
    def __getitem__(self, index):
        """
        Récupère un ou plusieurs vecteurs par leur indice, avec cache LRU en mode mmap.
        
        Args:
            index: Un entier, une liste d'entiers ou un slice
            
        Returns:
            Un vecteur numpy ou un tableau de vecteurs
        """
        if self.mode == "ram":
            # En mode RAM, juste indexer le tableau numpy
            return self.vectors[index]
        else:
            # En mode mmap, utiliser le cache si disponible, sinon lire depuis le fichier
            if isinstance(index, (int, np.integer)):
                # Cas d'un seul indice - vérifier le cache d'abord
                if self.cache_enabled and index in self.cache:
                    # Cache hit
                    self.cache_hits += 1
                    # Déplacer l'élément à la fin (MRU)
                    vector = self.cache.pop(index)
                    self.cache[index] = vector
                    self._update_cache_stats()
                    return vector
                else:
                    # Cache miss - lire depuis le fichier
                    if self.cache_enabled:
                        self.cache_misses += 1
                    
                    # Lire le vecteur depuis mmap
                    offset = self.header_size + index * self.vector_size
                    self.mmap_obj.seek(offset)
                    vector_bytes = self.mmap_obj.read(self.vector_size)
                    vector = np.frombuffer(vector_bytes, dtype=np.float32)
                    
                    # Mettre en cache si activé
                    if self.cache_enabled:
                        # Si le cache est plein, supprimer l'élément le moins récemment utilisé (LRU)
                        if len(self.cache) >= self.cache_capacity and self.cache_capacity > 0:
                            self.cache.popitem(last=False)  # Supprimer le premier élément (LRU)
                        
                        # Ajouter au cache
                        self.cache[index] = vector
                        
                        self._update_cache_stats()
                    
                    return vector
            elif isinstance(index, slice):
                # Cas d'un slice
                start = index.start or 0
                stop = index.stop or self.n
                step = index.step or 1
                
                if step == 1:
                    # Optimisation: lecture en bloc pour les slices consécutifs
                    size = stop - start
                    
                    # Vérifier si tous les éléments sont dans le cache
                    if self.cache_enabled and size <= 1000:  # Pour éviter de trop grandes vérifications
                        cache_indices = [i for i in range(start, stop) if i in self.cache]
                        # Si plus de 80% des indices sont dans le cache, utiliser le cache
                        if len(cache_indices) > 0.8 * size:
                            return np.array([self[i] for i in range(start, stop)])
                    
                    # Lecture directe depuis le fichier
                    offset = self.header_size + start * self.vector_size
                    self.mmap_obj.seek(offset)
                    buffer = self.mmap_obj.read(size * self.vector_size)
                    vectors = np.frombuffer(buffer, dtype=np.float32).reshape(size, self.d)
                    
                    # Mettre en cache les vecteurs fréquemment utilisés
                    if self.cache_enabled and size <= 50:  # Limiter aux petites plages
                        for i, idx in enumerate(range(start, stop)):
                            # Si le cache est plein, supprimer l'élément LRU
                            if len(self.cache) >= self.cache_capacity:
                                self.cache.popitem(last=False)
                            self.cache[idx] = vectors[i]
                    
                    return vectors
                else:
                    # Cas non optimisé pour les slices avec step > 1
                    indices = range(start, stop, step)
                    return np.array([self[i] for i in indices])
            else:
                # Cas d'une liste d'indices
                if len(index) > 200:
                    # Optimisation 1: Vérifier le cache pour tous les indices d'abord
                    if self.cache_enabled:
                        # Séparer les indices présents dans le cache et ceux absents
                        cached_indices = []
                        missing_indices = []
                        
                        for idx in index:
                            if idx in self.cache:
                                cached_indices.append(idx)
                            else:
                                missing_indices.append(idx)
                        
                        # Si plus de 80% sont dans le cache, accéder individuellement
                        if len(cached_indices) > 0.8 * len(index):
                            return np.array([self[i] for i in index])
                    
                    # Optimisation 2: regrouper les indices consécutifs pour les grandes listes
                    sorted_indices = np.sort(index)
                    groups = []
                    current_group = [sorted_indices[0]]
                    
                    # Identifier les groupes d'indices consécutifs
                    for i in range(1, len(sorted_indices)):
                        if sorted_indices[i] == sorted_indices[i-1] + 1:
                            current_group.append(sorted_indices[i])
                        else:
                            groups.append(current_group)
                            current_group = [sorted_indices[i]]
                    groups.append(current_group)
                    
                    # Lire chaque groupe en bloc
                    vectors = []
                    for group in groups:
                        if len(group) == 1:
                            # Un seul indice
                            vectors.append(self[group[0]])
                        else:
                            # Bloc d'indices consécutifs
                            start, end = group[0], group[-1] + 1
                            vectors.append(self[start:end])
                    
                    # Réorganiser les vecteurs selon l'ordre original des indices
                    result = np.vstack([v if len(v.shape) > 1 else v.reshape(1, -1) for v in vectors])
                    index_map = {idx: i for i, idx in enumerate(sorted_indices)}
                    return result[[index_map[idx] for idx in index]]
                else:
                    # Pour les petites listes, utiliser le cache pour chaque vecteur
                    return np.array([self[i] for i in index])
    
    def __len__(self) -> int:
        """Retourne le nombre de vecteurs."""
        return self.n
    
    def dot(self, vectors, query):
        """
        Calcule le produit scalaire entre les vecteurs et une requête.
        Optimisé pour chaque mode avec des techniques d'accélération.
        
        Args:
            vectors: Les vecteurs (indice ou liste d'indices)
            query: Le vecteur requête
            
        Returns:
            Un tableau numpy avec les scores de similarité
        """
        if self.mode == "ram":
            # En mode RAM, utiliser directement numpy
            return np.dot(self[vectors], query)
        else:
            # En mode mmap, stratégie optimisée pour différentes tailles
            if isinstance(vectors, (int, np.integer)):
                # Cas d'un seul vecteur
                return np.dot(self[vectors], query)
            elif len(vectors) > 3000:
                # Pour les grandes listes, traiter par lots de taille optimale
                batch_size = 3000  # Taille optimale pour balance entre I/O et calcul
                n_batches = (len(vectors) + batch_size - 1) // batch_size
                
                results = np.empty(len(vectors), dtype=np.float32)
                
                for i in range(n_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(vectors))
                    batch_indices = vectors[start_idx:end_idx]
                    
                    # Utiliser l'accès optimisé pour la lecture par blocs
                    batch_vectors = self[batch_indices]
                    results[start_idx:end_idx] = np.dot(batch_vectors, query)
                
                return results
            else:
                # Pour les listes de taille modérée, lecture optimisée en un bloc
                vectors_array = self[vectors]  # Utilise déjà l'optimisation par blocs
                return np.dot(vectors_array, query)
    
    def close(self) -> None:
        """Libère les ressources, y compris le cache."""
        if self.mode == "mmap" and self.mmap_obj is not None:
            # Afficher les statistiques finales du cache
            if self.cache_enabled:
                self._update_cache_stats(force=True)
                print(f"  → Cache final: {len(self.cache):,} vecteurs, {self.cache_hits:,} hits, {self.cache_misses:,} misses")
                
                # Vider le cache
                self.cache.clear()
                self.cache = None
            
            # Fermer mmap
            self.mmap_obj.close()
            self.mmap_file.close()
            self.mmap_obj = None
            self.mmap_file = None

class TreeIO:
    """Classe pour la lecture et l'écriture d'arbres K16."""
    
    @staticmethod
    def save_tree(tree: Union[TreeNode, K16Tree], file_path: str) -> None:
        """
        Sauvegarde un arbre dans un fichier binaire avec pickle.
        
        Args:
            tree: Arbre K16 ou nœud racine à sauvegarder
            file_path: Chemin du fichier de sortie
        """
        start_time = time.time()
        print(f"⏳ Sauvegarde de l'arbre vers {file_path}...")
        
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Si on reçoit un K16Tree, extraire la racine
        root = tree.root if isinstance(tree, K16Tree) else tree
        
        with open(file_path, "wb") as f:
            pickle.dump(root, f)
        
        elapsed = time.time() - start_time
        print(f"✓ Arbre sauvegardé vers {file_path} en {elapsed:.2f}s")
    
    @staticmethod
    def load_tree(file_path: str) -> Tuple[TreeNode, float]:
        """
        Charge un arbre depuis un fichier binaire.

        Args:
            file_path: Chemin du fichier contenant l'arbre

        Returns:
            Tuple[TreeNode, float]: L'arbre chargé et le temps d'exécution
        """
        start_time = time.time()
        print(f"⏳ Chargement de l'arbre depuis {file_path}...")

        try:
            with open(file_path, "rb") as f:
                # Use our custom unpickler to handle legacy TreeNode objects
                unpickler = CustomUnpickler(f)
                tree = unpickler.load()

            elapsed = time.time() - start_time
            print(f"✓ Arbre chargé depuis {file_path} en {elapsed:.2f}s")
            return tree, elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Erreur lors du chargement de l'arbre: {str(e)}")
            raise
    
    @staticmethod
    def load_as_k16tree(file_path: str) -> K16Tree:
        """
        Charge un arbre depuis un fichier binaire et l'encapsule dans un objet K16Tree.
        
        Args:
            file_path: Chemin du fichier contenant l'arbre
            
        Returns:
            K16Tree: L'arbre K16 encapsulant le nœud racine chargé
        """
        root, _ = TreeIO.load_tree(file_path)
        return K16Tree(root)
# K16 - Recherche ultra-rapide de vecteurs similaires

K16 est une bibliothèque Python optimisée pour la recherche rapide de vecteurs d'embedding similaires. Elle utilise une structure d'arbre hiérarchique avec des optimisations SIMD et Numba pour offrir des performances exceptionnelles.

## Caractéristiques

- 🚀 **Performances élevées** : 50-100x plus rapide que la recherche naïve
- 🧠 **Mémoire optimisée** : Réduction de 50-80% de la consommation mémoire
- 🔍 **Précision élevée** : Recall > 85% configurable
- 📊 **Scalable** : Supporte des millions de vecteurs
- 🛠️ **Modulaire** : API simple et extensible

## Installation

```bash
# Installation depuis le dépôt
git clone https://github.com/iapourtous/k16
cd k16
pip install -e .

# Installation avec support GPU (nécessite CUDA)
pip install -e ".[gpu]"

# Installation avec outils de développement
pip install -e ".[dev]"

# Vérification de l'installation
python -m k16.cli --version
```

La bibliothèque peut également être utilisée directement en ligne de commande après installation:

```bash
k16 --version
k16 getData
k16 build
k16 test
k16 api
```

## Guide rapide

K16 propose une interface en ligne de commande simple pour toutes les fonctionnalités:

### 1. Télécharger et préparer les données

```bash
# Téléchargement de Natural Questions (open)
python -m k16.cli getData

# Options personnalisées
python -m k16.cli getData --model intfloat/multilingual-e5-large --batch-size 128
```

### 2. Construire un arbre

```bash
# Construction d'un arbre avec les paramètres par défaut
python -m k16.cli build

# Options personnalisées
python -m k16.cli build --max_depth 32 --max_data 256 --k 16 --hnsw
```

### 3. Tester les performances

```bash
# Test avec les paramètres par défaut
python -m k16.cli test

# Options personnalisées
python -m k16.cli test --k 100 --queries 1000 --search_type beam --beam_width 3
```

### 4. API REST avec FastAPI

```bash
# Démarrer l'API REST
python -m k16.cli api

# Options personnalisées
python -m k16.cli api --host 0.0.0.0 --port 5000 --reload
```

Endpoints disponibles:
- `GET /`: Informations sur l'API
- `GET /health`: Vérification de l'état de l'API
- `GET /stats`: Statistiques sur l'arbre et les vecteurs
- `POST /search`: Recherche de vecteurs similaires

Exemple d'utilisation de l'endpoint `/search`:

```bash
curl -X 'POST' \
  'http://localhost:8000/search' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "comment fonctionne un arbre de recherche?",
  "k": 5
}'
```

Réponse:
```json
{
  "results": [
    {
      "question": "Comment fonctionne un arbre binaire de recherche ?",
      "answer": "Un arbre binaire de recherche est une structure de données...",
      "similarity": 0.89,
      "index": 120
    },
    ...
  ],
  "timings": {
    "encode_ms": 15.5,
    "tree_search_ms": 5.2,
    "filter_ms": 3.1,
    "total_ms": 23.8
  },
  "stats": {
    "candidates_count": 256
  }
}

## Configuration

Toutes les options peuvent être configurées dans le fichier `config.yaml`. Voici une documentation détaillée des paramètres:

### Configuration générale

```yaml
general:
  debug: false  # Mode debug pour informations supplémentaires
```

### Paramètres de construction d'arbre

```yaml
build_tree:
  # Paramètres principaux
  max_depth: 32            # Profondeur maximale de l'arbre
  max_leaf_size: 16        # Taille maximale d'une feuille pour l'arrêt de la subdivision
  max_data: 256            # Nombre de vecteurs à stocker dans chaque feuille
  max_workers: 12          # Nombre de processus parallèles (default: CPU count)
  use_gpu: true            # Utilisation du GPU pour le clustering K-means si disponible
  k: 16                    # Nombre de branches par nœud (non présent dans l'exemple)

  # Paramètres d'amélioration HNSW
  use_hnsw_improvement: true   # Amélioration des candidats avec HNSW après construction
  prune_unused_leaves: true    # Suppression des feuilles jamais utilisées pendant la recherche
  hnsw_batch_size: 1000        # Taille du lot pour les recherches HNSW
  grouping_batch_size: 5000    # Taille du lot pour le regroupement de vecteurs
  hnsw_m: 16                   # Paramètre M pour HNSW (connections par nœud)
  hnsw_ef_construction: 200    # Paramètre efConstruction pour HNSW (qualité vs vitesse)
```

### Paramètres de recherche

```yaml
search:
  k: 10                 # Nombre de voisins les plus proches à récupérer
  queries: 100          # Nombre de requêtes aléatoires pour les tests de benchmark
  mode: "ram"           # Mode de chargement: "ram" (complet) ou "mmap" (mappé en mémoire)
  cache_size_mb: 500    # Taille du cache en mégaoctets (pour le mode mmap)
  use_faiss: true       # Utilisation de FAISS pour la recherche naïve et le filtrage final

  # Configuration de l'algorithme de recherche
  search_type: "single" # Algorithme de recherche: "single" (descente simple) ou "beam" (faisceau)
  beam_width: 1         # Nombre de branches à explorer simultanément (uniquement pour beam search)
                        # Des valeurs plus élevées améliorent le recall au détriment de la vitesse
```

### Paramètres de représentation d'arbre

```yaml
use_flat_tree: true     # Utilisation de la structure plate optimisée pour une recherche plus rapide

flat_tree:
  # Paramètres de réduction dimensionnelle
  max_dims: 512                     # Nombre de dimensions à conserver à chaque niveau
  reduction_method: "directional"   # Méthode de sélection des dimensions: "variance" ou "directional"
```

### Paramètres de préparation des données

```yaml
prepare_data:
  model: "intfloat/multilingual-e5-large"  # Modèle d'embedding à utiliser
  batch_size: 128                          # Taille du lot pour l'encodage
  normalize: true                          # Normaliser les embeddings à la longueur unitaire
```

### Chemins de fichiers et valeurs par défaut

```yaml
files:
  vectors_dir: "/path/to/data"  # Répertoire des vecteurs
  trees_dir: "/path/to/models"  # Répertoire des arbres
  default_qa: "qa.txt"          # Fichier texte par défaut
  default_vectors: "data.bin"   # Fichier de vecteurs par défaut
  default_tree: "tree.bsp"      # Fichier d'arbre par défaut
```

### Configuration de l'API

```yaml
api:
  host: "127.0.0.1"    # Adresse d'hôte pour l'API
  port: 8000           # Port pour l'API
  reload: false        # Rechargement automatique pour le développement
```

### Impact des paramètres sur les performances

- **max_depth**: Une profondeur plus élevée permet une meilleure précision mais augmente le temps de construction.
- **max_data**: Une valeur plus élevée améliore le recall mais ralentit la recherche.
- **k**: Le nombre de clusters par nœud. Une valeur plus élevée permet une meilleure précision mais ralentit la construction.
- **search_type** et **beam_width**: "beam" avec une largeur plus grande améliore le recall mais ralentit la recherche.
- **use_hnsw_improvement**: Améliore considérablement le recall avec un léger impact sur le temps de construction.

## Structure du projet

- `/k16/` : Package principal
  - `/core/` : Structures de données principales (Tree, TreeFlat)
  - `/builder/` : Construction d'arbres optimisés
  - `/search/` : Algorithmes de recherche
  - `/io/` : Lecture/écriture de vecteurs et d'arbres
  - `/utils/` : Utilitaires et configuration
- `/tests/` : Tests unitaires et d'intégration
- `/data/` : Données d'exemple
- `/models/` : Arbres pré-construits

## Licence

Ce projet est sous licence [MIT](LICENSE).
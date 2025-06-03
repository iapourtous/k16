"""
Module de gestion de la configuration pour K16.
Centralise le chargement et l'accès à la configuration YAML.
"""

import os
import yaml

# Chemin par défaut du fichier de configuration
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yaml")

# Configuration par défaut
DEFAULT_CONFIG = {
    "general": {
        "debug": False
    },
    "build_tree": {
        "max_depth": 6,
        "k": 16,
        "k_adaptive": False,  # Valeur fixe k=16 par défaut
        "k_min": 2,
        "k_max": 32,
        "max_leaf_size": 100,
        "max_data": 256,  # Multiple de 16 pour optimisation SIMD
        "max_workers": 8,
        "use_gpu": True,
        "prune_unused": False,  # Paramètre obsolète
        # Le pruning des feuilles inutilisées est maintenant automatique et ne peut plus être désactivé
        "hnsw_batch_size": 1000,
        "grouping_batch_size": 5000,
        "hnsw_m": 16,
        "hnsw_ef_construction": 200,
    },
    "flat_tree": {
        "max_dims": 512,  # Multiple de 16 pour optimisation SIMD
        "reduction_method": "variance"  # ou "directional"
    },
    "search": {
        "k": 100,
        "queries": 100,
        "mode": "ram",
        "cache_size_mb": 500,
        "use_faiss": True
    },
    "prepare_data": {
        "model": "intfloat/multilingual-e5-large",
        "batch_size": 128,
        "normalize": True
    },
    "files": {
        "vectors_dir": ".",
        "trees_dir": ".",
        "default_qa": "qa.txt",
        "default_vectors": "vectors.bin",
        "default_tree": "tree.bin"
    }
}

class ConfigManager:
    """Gestionnaire de configuration pour K16."""
    
    def __init__(self, config_path=None):
        """
        Initialise le gestionnaire de configuration.
        
        Paramètres :
            config_path: Chemin vers le fichier de configuration YAML.
                         Si None, utilise le chemin par défaut.
        """
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config = self.load_config()
        
    def load_config(self):
        """
        Charge la configuration depuis le fichier YAML.
        
        Retourne :
            Dict: La configuration chargée, ou la configuration par défaut en cas d'erreur.
        """
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Vérifier et compléter la configuration
            self._ensure_complete_config(config)
            
            return config
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement de la configuration: {str(e)}")
            print(f"⚠️ Utilisation des paramètres par défaut")
            return DEFAULT_CONFIG.copy()
    
    def _ensure_complete_config(self, config):
        """
        S'assure que la configuration contient toutes les sections nécessaires.
        Complète avec les valeurs par défaut si nécessaire.
        
        Paramètres :
            config: Configuration à vérifier et compléter.
        """
        for section, default_values in DEFAULT_CONFIG.items():
            if section not in config:
                config[section] = default_values.copy()
            else:
                for key, value in default_values.items():
                    if key not in config[section]:
                        config[section][key] = value
    
    def get_section(self, section):
        """
        Récupère une section complète de la configuration.
        
        Paramètres :
            section: Nom de la section à récupérer.

        Retourne :
            Dict: La section demandée, ou un dictionnaire vide si la section n'existe pas.
        """
        return self.config.get(section, {})
    
    def get(self, section, key, default=None):
        """
        Récupère une valeur spécifique de la configuration.
        
        Paramètres :
            section: La section contenant la clé.
            key: La clé à récupérer.
            default: Valeur par défaut si la clé n'existe pas.

        Retourne :
            La valeur associée à la clé, ou la valeur par défaut si la clé n'existe pas.
        """
        section_data = self.get_section(section)
        return section_data.get(key, default)
    
    def get_file_path(self, file_key, default=None):
        """
        Construit le chemin complet vers un fichier spécifié dans la configuration.
        
        Paramètres :
            file_key: Clé du fichier dans la section 'files'.
            default: Valeur par défaut si la clé n'existe pas.

        Retourne :
            Le chemin complet vers le fichier.
        """
        files_section = self.get_section("files")
        
        if file_key.startswith("default_"):
            # Pour les fichiers par défaut, construire le chemin complet
            file_name = files_section.get(file_key, default)
            
            # Déterminer le répertoire approprié
            if "vectors" in file_key:
                dir_key = "vectors_dir"
            elif "tree" in file_key:
                dir_key = "trees_dir"
            else:
                dir_key = "vectors_dir"  # Par défaut
            
            dir_path = files_section.get(dir_key, ".")
            return os.path.join(dir_path, file_name)
        else:
            # Pour les autres clés, retourner directement la valeur
            return files_section.get(file_key, default)
    
    def reload(self, config_path=None):
        """
        Recharge la configuration depuis un nouveau fichier.
        
        Paramètres :
            config_path: Nouveau chemin de configuration. Si None, utilise le chemin actuel.
        """
        if config_path:
            self.config_path = config_path
        self.config = self.load_config()
        
    def __str__(self):
        """Représentation de la configuration pour le débogage."""
        return f"Configuration chargée depuis: {self.config_path}"

# Fonction utilitaire pour charger une configuration
def load_config(config_path=None):
    """
    Fonction utilitaire pour charger rapidement une configuration.
    
    Paramètres :
        config_path: Chemin vers le fichier de configuration YAML.
                     Si None, utilise le chemin par défaut.
    
    Retourne :
        ConfigManager: Instance du gestionnaire de configuration.
    """
    return ConfigManager(config_path)
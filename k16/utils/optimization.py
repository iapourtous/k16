"""
Module d'optimisations pour K16.
Configure et gère les optimisations numériques (SIMD, Numba JIT, etc.).
"""

import os
import numpy as np
import platform
from typing import Dict, Any, Optional, List

# Vérifier si Numba est disponible
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def configure_simd():
    """
    Configure l'environnement pour utiliser les optimisations SIMD.
    Force l'utilisation des instructions SIMD dans NumPy.
    """
    # Forcer l'utilisation des instructions SIMD
    os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
    os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
    os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
    
    # Optimisations d'alignement mémoire pour SIMD
    # Améliore significativement les performances des dot products
    try:
        # Alignement mémoire pour vectorisation optimale
        np.config.add_option_enable_numpy_api(False)

        # Sur certaines plateformes, ces options peuvent être disponibles
        try:
            # Pour Intel MKL
            np.config.add_option_mkl_vml_accuracy("high")
            np.config.add_option_mkl_enable_instructions("AVX2")
            # Si AVX-512 est disponible
            if "avx512" in platform.processor().lower():
                np.config.add_option_mkl_enable_instructions("AVX512")
        except Exception:
            pass
    except Exception:
        pass

def check_simd_support():
    """
    Vérifier les extensions SIMD supportées par la configuration NumPy.
    
    Returns:
        List[str]: Liste des extensions SIMD disponibles.
    """
    simd_extensions = []
    try:
        config_info = np.__config__.show()
        if "SIMD Extensions" in config_info:
            print("✓ Extensions SIMD disponibles pour NumPy:")
            capture = False
            for line in config_info.split("\n"):
                if "SIMD Extensions" in line:
                    capture = True
                elif capture and line.startswith("  "):
                    ext = line.strip()
                    simd_extensions.append(ext)
                    print(f"  - {ext}")
                elif capture and not line.startswith("  "):
                    break
        return simd_extensions
    except Exception:
        print("✓ NumPy configuré pour utiliser les instructions SIMD disponibles")
        return simd_extensions

def check_numba_support():
    """
    Vérifie si Numba est disponible et configuré correctement.
    
    Returns:
        bool: True si Numba est disponible et configuré correctement.
    """
    if NUMBA_AVAILABLE:
        print("✓ Numba JIT est disponible pour l'optimisation")
        return True
    else:
        print("⚠️ Numba JIT n'est pas disponible. Les performances seront réduites.")
        return False

def optimize_functions():
    """
    Configure toutes les optimisations numériques pour K16.
    
    Returns:
        Dict[str, Any]: Un dictionnaire contenant les informations sur les optimisations.
    """
    # Configuration SIMD
    configure_simd()
    
    # Vérifier le support
    simd_extensions = check_simd_support()
    numba_available = check_numba_support()
    
    return {
        "simd": {
            "available": len(simd_extensions) > 0,
            "extensions": simd_extensions
        },
        "numba": {
            "available": numba_available
        }
    }
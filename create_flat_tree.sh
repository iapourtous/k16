#!/bin/bash
# Script pour convertir un arbre existant en structure plate optimisée

source venv/bin/activate

# Chemin de l'arbre source (l'arbre normal)
SRC_TREE="models/tree.bsp"

# Chemin de l'arbre plat à créer
FLAT_TREE="models/tree.flat"

# Créer un arbre temporaire Python pour la conversion
cat > convert_tree.py << EOL
#!/usr/bin/env python3
"""
Script temporaire pour convertir un arbre K16 existant en structure plate optimisée.
"""

import os
import sys
from lib.io import TreeIO

def main():
    # Charger l'arbre existant
    tree_io = TreeIO()
    tree = tree_io.load_as_k16tree("${SRC_TREE}", use_flat_structure=True)
    
    # Sauvegarder l'arbre en format flat
    # Cela va automatiquement créer un fichier .flat
    tree_io.save_tree(tree, "${SRC_TREE}", save_flat=True)
    
    print(f"\n✅ Arbre plat créé avec succès : ${FLAT_TREE}")
    print("Lors de la prochaine utilisation, l'arbre plat sera automatiquement utilisé.")

if __name__ == "__main__":
    main()
EOL

# Exécuter le script temporaire
python convert_tree.py

# Nettoyer
rm convert_tree.py

echo "✓ Script terminé."
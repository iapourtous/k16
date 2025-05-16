#!/usr/bin/env bash
# Script d'installation et de configuration pour K16 Search

set -e  # Exit on error

# Vérifier qu'on utilise bien bash
if [ -z "$BASH_VERSION" ]; then
    echo "Ce script doit être exécuté avec bash, pas sh"
    echo "Utilisez: bash install.sh"
    exit 1
fi

# Couleurs pour l'affichage
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# En-tête
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}    K16 Search - Installation       ${NC}"
echo -e "${GREEN}====================================${NC}"
echo ""

# Vérifier la version de Python système
log_info "Vérification de Python système..."

# Chercher Python dans le système
SYSTEM_PYTHON=""
if command -v python3 &> /dev/null; then
    SYSTEM_PYTHON="python3"
elif command -v python &> /dev/null; then
    SYSTEM_PYTHON="python"
else
    log_error "Python 3 n'est pas installé. Veuillez installer Python 3.8 ou supérieur."
    exit 1
fi

log_info "Python système trouvé: $SYSTEM_PYTHON"
system_python_version=$($SYSTEM_PYTHON -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log_success "Python système $system_python_version détecté"

# Créer un environnement virtuel
log_info "Création de l'environnement virtuel..."
if [ ! -d "venv" ]; then
    $SYSTEM_PYTHON -m venv venv
    log_success "Environnement virtuel créé"
else
    log_warning "L'environnement virtuel existe déjà"
fi

# Activer l'environnement virtuel
log_info "Activation de l'environnement virtuel..."
if [ -f "venv/bin/activate" ]; then
    . venv/bin/activate
else
    log_error "Impossible d'activer l'environnement virtuel"
    exit 1
fi

# Vérifier le Python dans le venv
PYTHON_CMD="python"
python_version=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
log_success "Python $python_version activé dans l'environnement virtuel"

# Mettre à jour pip
log_info "Mise à jour de pip..."
$PYTHON_CMD -m pip install --upgrade pip

# Installer les dépendances
log_info "Installation des dépendances..."
pip install -r requirements.txt

# Créer requirements.txt s'il n'existe pas
if [ ! -f "requirements.txt" ]; then
    log_info "Création du fichier requirements.txt..."
    cat > requirements.txt << EOF
numpy>=1.21.0
sentence-transformers>=2.2.0
datasets>=2.14.0
tqdm>=4.62.0
streamlit>=1.25.0
PyYAML>=6.0
scikit-learn>=1.0.0
faiss-cpu>=1.7.0
psutil>=5.9.0
EOF
    pip install -r requirements.txt
fi

# Créer les répertoires nécessaires
log_info "Création des répertoires..."
mkdir -p data models logs

# Préparer les données
log_info "Préparation des données (téléchargement et encodage)..."
echo ""
echo "Cette étape peut prendre plusieurs minutes selon votre connexion internet"
echo "et la puissance de votre machine..."
echo ""

# Vérifier si les données existent déjà
if [ -f "data/qa.txt" ] && [ -f "data/data.bin" ]; then
    log_warning "Les données semblent déjà être générées."
    read -p "Voulez-vous les régénérer ? (o/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        $PYTHON_CMD src/prepareData.py
    else
        log_info "Utilisation des données existantes"
    fi
else
    $PYTHON_CMD src/prepareData.py
fi

log_success "Données préparées"

# Construire l'arbre
log_info "Construction de l'arbre de recherche..."
echo ""
echo "Cette étape peut prendre quelques minutes..."
echo ""

# Vérifier si l'arbre existe déjà
if [ -f "models/tree.bsp" ]; then
    log_warning "L'arbre semble déjà être construit."
    read -p "Voulez-vous le reconstruire ? (o/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Oo]$ ]]; then
        $PYTHON_CMD src/build_tree.py
    else
        log_info "Utilisation de l'arbre existant"
    fi
else
    $PYTHON_CMD src/build_tree.py
fi

log_success "Arbre construit"

# Créer des scripts de lancement
log_info "Création des scripts de lancement..."

# Script pour le test
cat > test.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
python src/test.py "$@"
EOF
chmod +x test.sh

# Script pour Streamlit
cat > search.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
streamlit run src/streamlit_search.py
EOF
chmod +x search.sh

log_success "Scripts créés"

# Afficher les instructions finales
echo ""
echo -e "${GREEN}====================================${NC}"
echo -e "${GREEN}    Installation terminée !         ${NC}"
echo -e "${GREEN}====================================${NC}"
echo ""
echo -e "${BLUE}Instructions pour utiliser K16 Search :${NC}"
echo ""
echo "1. Pour tester la recherche (évaluation des performances) :"
echo "   ./test.sh --k 100 --queries 100"
echo ""
echo "2. Pour lancer l'interface de recherche Streamlit :"
echo "   ./search.sh"
echo ""
echo "3. Pour reconstruire les données ou l'arbre :"
echo "   - Données : python src/prepareData.py"
echo "   - Arbre   : python src/build_tree.py"
echo ""
echo -e "${GREEN}L'interface Streamlit sera accessible à l'adresse : http://localhost:8501${NC}"
echo ""
log_info "N'oubliez pas d'activer l'environnement virtuel avant d'utiliser les scripts Python :"
echo "   source venv/bin/activate"
echo ""
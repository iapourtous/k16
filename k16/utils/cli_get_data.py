"""
Module pour le téléchargement et la préparation des données.
Fournit des fonctions pour télécharger le dataset Natural Questions et générer les embeddings.
"""

import os
import sys
import time
import datetime
import struct
import numpy as np
import argparse
from typing import List, Dict, Any
from tqdm.auto import tqdm

from k16.utils.config import ConfigManager

def format_time(seconds: float) -> str:
    """Formate le temps en heures, minutes, secondes."""
    return str(datetime.timedelta(seconds=int(seconds)))

def write_vectors(vecs: np.ndarray, path: str):
    """
    Écrit les vecteurs dans un binaire : header (n, d) + float32 data.
    
    Args:
        vecs: Vecteurs à écrire
        path: Chemin du fichier de sortie
    """
    n, d = vecs.shape
    start_time = time.time()
    print(f"⏳ Écriture de {n:,} vecteurs (dim {d}) vers {path}...")
    
    # S'assurer que le répertoire existe
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    # Convertir en float32 et s'assurer que les vecteurs sont normalisés
    vecs_float32 = vecs.astype(np.float32)
    
    # Format exact attendu par build_candidates
    with open(path, "wb") as f:
        # Utiliser QQ (uint64_t) pour assurer la compatibilité avec endianness explicite
        f.write(struct.pack("<QQ", n, d))
        f.write(vecs_float32.tobytes())
    
    elapsed = time.time() - start_time
    print(f"✓ {n:,} vecteurs (dim {d}) écrits dans {path} [terminé en {elapsed:.2f}s]")

def get_data_command(args: argparse.Namespace) -> int:
    """
    Commande pour télécharger et préparer les données.
    
    Args:
        args: Arguments de ligne de commande
        
    Returns:
        int: Code de retour (0 pour succès, autre pour erreur)
    """
    # Initialisation du gestionnaire de configuration
    config_manager = ConfigManager(args.config)
    
    # Récupération des paramètres pour la préparation des données
    prepare_data_config = config_manager.get_section("prepare_data")
    files_config = config_manager.get_section("files")
    
    # Enregistrer le temps de départ pour calculer la durée totale
    total_start_time = time.time()
    
    try:
        print(f"📥 Téléchargement et préparation des données...")
        print(f"  - Fichier QA: {args.out_text}")
        print(f"  - Fichier vecteurs: {args.out_vec}")
        print(f"  - Modèle: {args.model}")
        print(f"  - Taille de batch: {args.batch_size}")
        
        # 1. Télécharger NQ‑open (train)
        print("⏳ Téléchargement et préparation des données Natural Questions (open)...")
        print("  Cela peut prendre plusieurs minutes, veuillez patienter...")
        download_start = time.time()
        
        # Import dynamique pour éviter des dépendances inutiles
        try:
            from datasets import load_dataset
        except ImportError:
            print("⚠️ Bibliothèque 'datasets' non installée. Installation en cours...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            from datasets import load_dataset
            
        ds = load_dataset("nq_open", split="train")
        download_time = time.time() - download_start
        print(f"✓ Téléchargement terminé en {format_time(download_time)} - {len(ds):,} exemples chargés")
        
        # 2. Création du fichier QA
        print(f"⏳ Création du fichier QA {args.out_text}...")
        qa_start = time.time()
        os.makedirs(os.path.dirname(args.out_text) if os.path.dirname(args.out_text) else ".", exist_ok=True)
        
        lines = []
        with tqdm(total=len(ds), desc="Extraction Q&R") as pbar:
            for ex in ds:
                lines.append(f"{ex['question'].strip()} ||| {ex['answer'][0].strip()}")
                pbar.update(1)
        
        with open(args.out_text, "w", encoding="utf-8") as f_out:
            for i, ln in enumerate(tqdm(lines, desc="Écriture vers fichier")):
                f_out.write(ln.replace("\n", " ") + "\n")
                if (i+1) % 10000 == 0:
                    print(f"  → {i+1:,}/{len(lines):,} lignes écrites ({((i+1)/len(lines))*100:.1f}%)")
        
        qa_time = time.time() - qa_start
        print(f"✓ qa.txt écrit : {len(lines):,} lignes → {args.out_text} [terminé en {format_time(qa_time)}]")
        
        # 3. Embeddings
        # Vérifier si le fichier d'embeddings existe déjà
        recalculate = True
        if os.path.exists(args.out_vec):
            if not args.force:
                # Demander à l'utilisateur s'il veut recalculer les embeddings
                print(f"\nLe fichier d'embeddings {args.out_vec} existe déjà.")
                while True:
                    response = input("Voulez-vous recalculer les embeddings ? (o/n): ").lower()
                    if response in ['o', 'oui', 'y', 'yes']:
                        recalculate = True
                        break
                    elif response in ['n', 'non', 'no']:
                        recalculate = False
                        break
                    else:
                        print("Réponse non reconnue. Veuillez répondre par 'o' (oui) ou 'n' (non).")
            else:
                print(f"⚠️ Remplacement forcé du fichier d'embeddings existant: {args.out_vec}")
        
        encode_time = 0
        if recalculate:
            # Calculer les embeddings
            print(f"⏳ Chargement du modèle {args.model}...")
            encode_start = time.time()
            
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                print("⚠️ Bibliothèque 'sentence-transformers' non installée. Installation en cours...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
                from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(args.model)
            print(f"✓ Modèle chargé en {time.time() - encode_start:.2f}s")
            
            print(f"⏳ Encodage avec {args.model}...")
            total_batches = (len(lines) + args.batch_size - 1) // args.batch_size
            print(f"  → Encodage de {len(lines):,} lignes en {total_batches:,} batches de taille {args.batch_size}...")
            
            vecs = []
            encoded_examples = 0
            encode_start_time = time.time()
            
            for i in tqdm(range(0, len(lines), args.batch_size), desc="Batches", total=total_batches):
                batch = [f"passage: {t}" for t in lines[i : i + args.batch_size]]
                batch_size = len(batch)
                encoded_examples += batch_size
                
                batch_start = time.time()
                batch_vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                batch_time = time.time() - batch_start
                
                current_time = time.time() - encode_start_time
                examples_per_sec = encoded_examples / current_time if current_time > 0 else 0
                remaining = (len(lines) - encoded_examples) / examples_per_sec if examples_per_sec > 0 else 0
                
                vecs.append(batch_vecs)
                
                if (i + args.batch_size) % (args.batch_size * 10) == 0 or (i + batch_size) >= len(lines):
                    print(f"  → Batch {(i // args.batch_size) + 1}/{total_batches}: {batch_size} exemples en {batch_time:.2f}s "
                          f"({batch_size/batch_time:.1f} ex/s)")
                    print(f"  → Progrès: {encoded_examples:,}/{len(lines):,} exemples "
                          f"({encoded_examples/len(lines)*100:.1f}%) - Vitesse: {examples_per_sec:.1f} ex/s")
                    print(f"  → Temps écoulé: {format_time(current_time)} - Temps restant estimé: {format_time(remaining)}")
            
            vecs = np.vstack(vecs)
            encode_time = time.time() - encode_start_time
            print(f"✓ Encodage terminé en {format_time(encode_time)} - {len(lines):,} exemples @ {len(lines)/encode_time:.1f} ex/s")
            
            # Écrire les embeddings
            write_vectors(vecs, args.out_vec)
        else:
            print(f"Utilisation du fichier d'embeddings existant: {args.out_vec}")
        
        total_time = time.time() - total_start_time
        print("\n✓ Traitement terminé.")
        print(f"  - Configuration  : {args.config}")
        print(f"  - QA             : {args.out_text}")
        print(f"  - Embeddings     : {args.out_vec}")
        print(f"  - Modèle         : {args.model}")
        print(f"  - Batch size     : {args.batch_size}")
        print(f"  - Temps total    : {format_time(total_time)}")
        print(f"    ├─ Téléchargement : {format_time(download_time)} ({download_time/total_time*100:.1f}%)")
        print(f"    ├─ Création QA    : {format_time(qa_time)} ({qa_time/total_time*100:.1f}%)")
        if encode_time > 0:
            print(f"    └─ Encodage       : {format_time(encode_time)} ({encode_time/total_time*100:.1f}%)")
        else:
            print(f"    └─ Encodage       : (utilisé fichier existant)")
        
        # Instructions pour les étapes suivantes
        print("\nPour construire l'arbre avec ces vecteurs :")
        print(f"  python -m k16.cli build {args.out_vec} --config {args.config}")
        
    except Exception as e:
        print(f"\n❌ Erreur: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0
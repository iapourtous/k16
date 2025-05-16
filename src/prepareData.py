#!/usr/bin/env python3
"""
Prépare Natural Questions (open) – 307 k Q‑R – et génère :
  • qa.txt           (question ||| answer)
  • texts_sorted.txt (même contenu trié)
  • vectors.bin      (embeddings E5, float32)

Utilise un fichier de configuration YAML central.

Usage :
    python prepareData.py qa.txt vectors.bin \
           --model intfloat/multilingual-e5-large --batch-size 128
    python prepareData.py --config /path/to/config.yaml
"""

import os
import yaml

# Chemin par défaut du fichier de configuration
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")

# Chargement de la configuration
def load_config(config_path=None):
    """Charge la configuration depuis le fichier YAML."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement de la configuration: {str(e)}")
        print(f"⚠️ Utilisation des paramètres par défaut")
        return {
            "prepare_data": {
                "model": "intfloat/multilingual-e5-large",
                "batch_size": 128
            },
            "files": {
                "vectors_dir": ".",
                "default_qa": "qa.txt",
                "default_vectors": "vectors.bin"
            }
        }

# Chargement de la configuration
CONFIG = load_config()

# -------------------- paramètres par défaut (depuis config.yaml) -------------------- #
# Si la section n'existe pas dans le YAML, on créé des valeurs par défaut
if "prepare_data" not in CONFIG:
    CONFIG["prepare_data"] = {"model": "intfloat/multilingual-e5-large", "batch_size": 128}

DEF_MODEL = CONFIG["prepare_data"].get("model", "intfloat/multilingual-e5-large")
DEF_BATCH = CONFIG["prepare_data"].get("batch_size", 128)
# --------------------------------------------------------------------------------- #

import argparse, os, struct, numpy as np, time
from datasets import load_dataset
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import datetime

# ---------- helpers ---------- #
def write_bin(vecs: np.ndarray, path: str):
    """Écrit les vecteurs dans un binaire : header (n, d) + float32 data."""
    n, d = vecs.shape
    start_time = time.time()
    print(f"⏳ Écriture de {n:,} vecteurs (dim {d}) vers {path}...")

    # Convertir en float32 et s'assurer que les vecteurs sont normalisés
    vecs_float32 = vecs.astype(np.float32)

    # Format exact attendu par build_candidates
    with open(path, "wb") as f:
        # Utiliser QQ (uint64_t) pour assurer la compatibilité avec endianness explicite
        f.write(struct.pack("<QQ", n, d))
        f.write(vecs_float32.tobytes())

    elapsed = time.time() - start_time
    print(f"✓ {n:,} vecteurs (dim {d}) → {path} [terminé en {elapsed:.2f}s]")

def format_time(seconds):
    """Formate le temps en heures, minutes, secondes."""
    return str(datetime.timedelta(seconds=int(seconds)))

# ---------- main ---------- #
def main():
    global CONFIG
    total_start_time = time.time()

    # Définir les chemins par défaut à partir de la configuration
    default_qa_path = os.path.join(CONFIG["files"]["vectors_dir"], CONFIG["files"].get("default_qa", "qa.txt"))
    default_vectors_path = os.path.join(CONFIG["files"]["vectors_dir"], CONFIG["files"]["default_vectors"])

    ap = argparse.ArgumentParser("Natural Questions (open) → texte + embeddings")
    ap.add_argument("out_text", nargs="?", default=default_qa_path,
                    help=f"Fichier texte QA (par défaut: {default_qa_path})")
    ap.add_argument("out_vec", nargs="?", default=default_vectors_path,
                    help=f"Fichier binaire embeddings (par défaut: {default_vectors_path})")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                    help=f"Chemin vers le fichier de configuration (par défaut: {DEFAULT_CONFIG_PATH})")
    ap.add_argument("--model", default=DEF_MODEL,
                    help=f"Modèle d'embedding à utiliser (par défaut: {DEF_MODEL})")
    ap.add_argument("--batch-size", type=int, default=DEF_BATCH,
                    help=f"Taille des lots pour l'encodage (par défaut: {DEF_BATCH})")
    args = ap.parse_args()

    # Recharger la configuration si un fichier spécifique est fourni
    if args.config != DEFAULT_CONFIG_PATH:
        CONFIG = load_config(args.config)
        print(f"✓ Configuration chargée depuis: {args.config}")

    # 1. Télécharger NQ‑open (train)
    print("⏳ Téléchargement et préparation des données Natural Questions (open)...")
    print("  Cela peut prendre plusieurs minutes, veuillez patienter...")
    download_start = time.time()
    ds = load_dataset("nq_open", split="train")
    download_time = time.time() - download_start
    print(f"✓ Téléchargement terminé en {format_time(download_time)} - {len(ds):,} exemples chargés")

    # 2. création du fichier QA
    print(f"⏳ Création du fichier QA {args.out_text}...")
    qa_start = time.time()
    os.makedirs(os.path.dirname(args.out_text) or ".", exist_ok=True)
    
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


    # 4. Embeddings
    # Vérifier si le fichier d'embeddings existe déjà
    recalculate = True
    if os.path.exists(args.out_vec):
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

    if recalculate:
        # Calculer les embeddings
        print(f"⏳ Chargement du modèle {args.model}...")
        encode_start = time.time()
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
        write_bin(vecs, args.out_vec)
    else:
        print(f"Utilisation du fichier d'embeddings existant: {args.out_vec}")
        encode_time = 0  # Pas de temps d'encodage si on utilise le fichier existant

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
    print(f"  python src/build_tree.py {args.out_vec} --config {args.config}")
    print("ou")
    print(f"  python src/build_tree.py --config {args.config}")

if __name__ == "__main__":
    main()
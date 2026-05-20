#!/usr/bin/env python3
import sys
from collections import OrderedDict

def get_key(line: str) -> str:
    """
    Détermine la 'clé' d'une ligne, c'est-à-dire la partie qui sert
    à savoir si deux lignes 'commencent pareil'.

    Ici : on prend le premier mot (avant le premier espace).
    Tu peux adapter cette fonction selon ton format :
    - return line[:10]  # les 10 premiers caractères
    - return line.split(';')[0]  # ce qui est avant un ';'
    """
    # return line#[:20]
    cleaned = line.strip()
    if cleaned[0] in ["L", "G", "E"]:
        return cleaned[0]+" "+cleaned[1:].strip().split(" ")[0]
    elif "RHS1" in cleaned:
        return cleaned.split("RHS1")[0]+" "+cleaned.split("RHS1")[1].strip().split(" ")[0]
    elif "BND1" in cleaned:
        return cleaned.split("BND1")[0]+" "+cleaned.split("BND1")[1].strip().split(" ")[0]
    else :
        parts = cleaned.split("  ")
        if len(parts)==3:
            return cleaned.split("  ")[0]+" "+cleaned.split("  ")[1]
        else:
            return cleaned.split("  ")[0]
    # cleaned = line.replace("UP", "").strip()
    # cleaned = cleaned.replace("FX BND1", "").strip()
    # cleaned = cleaned.replace("L", "").strip()
    # cleaned = cleaned.replace("G", "").strip()
    # cleaned = cleaned.replace("E", "").strip()
    # cleaned = cleaned.replace("RHS1", "").strip()
    # if not cleaned:
    #     return ""
    # return cleaned.split(" ")[0]  # premier mot après nettoyage


def read_file_as_dict(path: str) -> OrderedDict:
    """
    Lit un fichier et retourne un dict (clé -> ligne complète).
    Si plusieurs lignes ont la même clé, la dernière écrase les précédentes.
    """
    d = OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            key = get_key(line)
            d[key] = line.rstrip('\n')  # on enlève juste le \n
    return d


def compare_files(file1: str, file2: str):
    data1 = read_file_as_dict(file1)  # clé -> ligne (ordre du fichier 1)
    data2 = read_file_as_dict(file2)  # clé -> ligne (ordre du fichier 2)

    keys1 = list(data1.keys())
    keys2 = list(data2.keys())

    # Lignes en moins : dans file1 mais pas dans file2, dans l'ordre de file1
    removed_keys = [k for k in keys1 if k not in data2]

    # Lignes en plus : dans file2 mais pas dans file1, dans l'ordre de file2
    added_keys = [k for k in keys2 if k not in data1]

    # Lignes modifiées : même clé dans les deux, contenu différent
    # On les liste dans l'ordre de file1
    modified_keys = [
        k for k in keys1
        if k in data2 and data1[k] != data2[k]
    ]

    filter_key = ["<0>","_0","t0"]

    print(f"Comparaison : {file1}  vs  {file2}\n")

    print("=== LIGNES EN MOINS (présentes seulement dans le premier fichier) ===")
    if not removed_keys:
        print("(aucune)")
    else:
        for k in removed_keys:
            if any([f in data1[k] for f in filter_key]):
                print(f"- {data1[k]}")

    print("\n=== LIGNES EN PLUS (présentes seulement dans le deuxième fichier) ===")
    if not added_keys:
        print("(aucune)")
    else:
        for k in added_keys:
            if any([f in data2[k] for f in filter_key]):
                print(f"+ {data2[k]}")

    print("\n=== LIGNES MODIFIÉES (même début, contenu différent) ===")
    if not modified_keys:
        print("(aucune)")
    else:
        for k in modified_keys:
            if any([f in data1[k] for f in filter_key]):
                print(f"* AVANT: {data1[k]}")
                print(f"  APRÈS: {data2[k]}")
                print("---")


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print(f"Usage : {sys.argv[0]} fichier1 fichier2")
    #     sys.exit(1)

    compare_files("C:/Users/jeannecor/Documents/1-PROJECTS/OPEN SOURCE/BP25/output/20260512-1834eco/problem-1-1--optim-nb-1.mps",
                  "C:/Users/jeannecor/Documents/1-PROJECTS/OPEN SOURCE/antares-studies-converted/BP25/output/20260512-1841eco/problem-1-1--optim-nb-1.mps")

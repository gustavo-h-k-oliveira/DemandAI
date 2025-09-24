import sys
from typing import List

import pandas as pd

CSV_PATH = "dataset_with_features.csv"

# Opções predefinidas (ajuste conforme o CSV)
PRESET_PRODUCTS = [
    "BOMBOM MORANGUETE 13G 160UN",
    "TETA BEL TRADICIONAL 50UN",
    "TOPBEL LEITE CONDENSADO 50UN",
    "TOPBEL TRADICIONAL 50UN",
]

def load_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="latin1")
    except UnicodeDecodeError:
        return pd.read_csv(path)

def choose_from_list(options: List[str], prompt: str = "Selecione uma opção: ") -> str:
    if not options:
        raise ValueError("Lista de opções vazia.")
    for idx, item in enumerate(options, 1):
        print(f"{idx:3d}) {item}")
    while True:
        sel = input(prompt).strip()
        if not sel.isdigit():
            print("Digite o número da opção.")
            continue
        i = int(sel)
        if 1 <= i <= len(options):
            return options[i - 1]
        print(f"Escolha um número entre 1 e {len(options)}.")

def main():
    # Carrega CSV
    df = load_csv(CSV_PATH)
    if "product" not in df.columns:
        print("Coluna 'product' não encontrada no CSV.", file=sys.stderr)
        sys.exit(1)

    # Lista todos os produtos e permite escolha por índice
    print("\n=== Seleção de Produto (por índice) ===")
    all_products = sorted(df["product"].dropna().unique())
    if not all_products:
        print("Nenhum produto disponível no CSV.")
        sys.exit(3)

    product = choose_from_list(all_products, prompt="Selecione o índice do produto: ")

    # Filtra dataset pelo produto escolhido e mostra pré-visualização
    dff = df[df["product"] == product].copy()
    if {"year", "month"}.issubset(dff.columns):
        dff = dff.sort_values(["year", "month"]).reset_index(drop=True)

    print(f"\nProduto selecionado: {product}")
    cols_preview = [c for c in ["product", "year", "month", "campaign", "seasonality", "quantity"] if c in dff.columns]
    print("\nPré-visualização:")
    try:
        # Mostra até 12 linhas para dar contexto
        print(dff[cols_preview].head(12).to_string(index=False))
    except Exception:
        # Fallback simples
        print(dff.head(12))

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import os
from typing import Optional

def try_import_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
        return pq
    except Exception:
        return None

def read_head_pandas(path: str, n: int, columns: Optional[list]):
    import pandas as pd
    if columns:
        return pd.read_parquet(path, columns=columns).head(n)
    return pd.read_parquet(path).head(n)

def stream_parquet_batches(path: str, batch_size: int, columns: Optional[list]):
    pq = try_import_pyarrow()
    if pq is None:
        raise RuntimeError("pyarrow não disponível para streaming. Instale with pip install pyarrow")
    pf = pq.ParquetFile(path)
    cols = columns if columns else None
    for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
        yield batch.to_pandas()

def print_schema(path: str):
    pq = try_import_pyarrow()
    if pq is not None:
        pf = pq.ParquetFile(path)
        print("Colunas (pyarrow):", pf.schema.names)
    else:
        # fallback: try pandas to get columns
        import pandas as pd
        try:
            df = pd.read_parquet(path, engine='pyarrow' if 'pyarrow' in pd.io.parquet.get_default_engine() else None, columns=None)
            print("Colunas (pandas):", list(df.columns))
        except Exception as e:
            print("Não foi possível ler esquema com pandas:", e)

def get_head_tail_count(path: str, n: int = 10, columns: Optional[list] = None):
    """Retorna (head_df, tail_df, total_rows) lendo de forma eficiente com pyarrow quando possível."""
    pq = try_import_pyarrow()
    import pandas as pd

    if pq is not None:
        pf = pq.ParquetFile(path)
        total = pf.metadata.num_rows if pf.metadata is not None else None

        # head: acumular até n rows
        head_rows = []
        head_count = 0

        # tail: manter deque de tamanho n
        from collections import deque
        tail_deque = deque(maxlen=n)

        cols = columns if columns else None
        for batch in pf.iter_batches(columns=cols, batch_size=100_000):
            batch_df = batch.to_pandas()
            # head
            if head_count < n:
                need = n - head_count
                head_rows.append(batch_df.head(need))
                head_count += min(need, len(batch_df))
            # tail (deque mantém último n)
            if len(batch_df) > 0:
                tail_deque.append(batch_df)

            # se já temos head e já avançamos bastante, podemos continuar apenas para tail
        # construir head_df e tail_df
        head_df = pd.concat(head_rows, ignore_index=True) if head_rows else pd.DataFrame(columns=cols or [])
        # construir tail concatenando os pedaços da deque e pegando últimas n linhas
        if tail_deque:
            tail_df = pd.concat(list(tail_deque), ignore_index=True).tail(n)
        else:
            tail_df = pd.DataFrame(columns=cols or [])
        return head_df.head(n), tail_df.tail(n), int(total) if total is not None else None

    # fallback pandas (pode consumir muita memória)
    df = pd.read_parquet(path, columns=columns) if columns else pd.read_parquet(path)
    total = len(df)
    return df.head(n), df.tail(n), total

def export_parquet_to_csv(path: str, out_csv: str, columns: Optional[list] = None, batch_size: int = 100_000, sep: str = ';'):
    """Exporta todo o conteúdo do parquet para CSV. Usa pyarrow em streaming quando possível."""
    pq = try_import_pyarrow()
    import pandas as pd

    # remove arquivo de saída se já existir (vamos reescrever)
    try:
        if os.path.exists(out_csv):
            os.remove(out_csv)
    except Exception:
        pass

    if pq is not None:
        pf = pq.ParquetFile(path)
        cols = columns if columns else None
        first = True
        for batch in pf.iter_batches(columns=cols, batch_size=batch_size):
            df = batch.to_pandas()
            # escreve incrementalmente, apenas header no primeiro chunk
            df.to_csv(out_csv, mode='a', index=False, header=first, sep=sep, encoding='utf-8')
            first = False
        return True
    # fallback pandas (pode consumir muita memória)
    df = pd.read_parquet(path, columns=columns) if columns else pd.read_parquet(path)
    df.to_csv(out_csv, index=False, sep=sep, encoding='utf-8')
    return True

def main():
    """Versão sem argumentos: caminhos e parâmetros hard-coded."""
    # --- CONFIGURAÇÃO hard-code ---
    PARQUET_PATH = r"C:\TCC\dados\rais\preprocessados\rais_RAIS_VINC_PUB_SUL_2023.txt_processado_ti.parquet"
    #PARQUET_PATH = r"C:\TCC\dados\rais\preprocessados\rais_DF2015_2015.txt_processado_ti.parquet"
    HEAD_SIZE = 10
    COLUMNS = None  # ex: ['ano','uf','eh_ti'] ou None para todas
    # -------------------------------

    if not os.path.isfile(PARQUET_PATH):
        print("Arquivo não encontrado:", PARQUET_PATH)
        sys.exit(2)

    try:
        head_df, tail_df, total = get_head_tail_count(PARQUET_PATH, n=HEAD_SIZE, columns=COLUMNS)
        print(f"Arquivo: {PARQUET_PATH}")
        print(f"Total de linhas: {total if total is not None else 'desconhecido'}\n")
        # imprimir head em formato CSV com ';' como separador
        print(f"Primeiras {len(head_df)} linhas (CSV, sep=';'):");
        print(head_df.to_csv(index=False, sep=';').strip())
        print("\n" + "-"*40 + "\n")
        # imprimir tail em formato CSV com ';' como separador
        print(f"Últimas {len(tail_df)} linhas (CSV, sep=';'):");
        print(tail_df.to_csv(index=False, sep=';').strip())

        # ==== EXPORTAR TODO O PARQUET PARA CSV ====
        base_name = os.path.splitext(os.path.basename(PARQUET_PATH))[0]
        out_dir = os.path.dirname(r"C:\TCC\dados\rais") or '.'
        full_csv = os.path.join(out_dir, f"{base_name}_full.csv")
        print(f"\nExportando todo o Parquet para CSV: {full_csv}")
        try:
            export_parquet_to_csv(PARQUET_PATH, full_csv, columns=COLUMNS, batch_size=100_000, sep=';')
            print("Exportação concluída.")
        except Exception as e:
            print("Erro ao exportar Parquet para CSV:", e)
        # ==========================================

    except Exception as e:
        print("Erro ao ler o arquivo Parquet:", e)
        if "pyarrow" in str(e).lower():
            print("Considere instalar pyarrow: pip install pyarrow")

if __name__ == "__main__":
    main()
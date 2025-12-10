"""Atalho simples para rodar o script RAIS original (mesma lógica).
Uso sugerido (PowerShell):
  python -m scripts.preditivo_rais_train --rais-dir C:\TCC\dados\rais\preprocessados
"""
import argparse
import sys

from modules.preditivo_rais.script import main as run_rais


def main():
    # Encaminha argumentos ao script original
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    run_rais()


if __name__ == "__main__":
    main()

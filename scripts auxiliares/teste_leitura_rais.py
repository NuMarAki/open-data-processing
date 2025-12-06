#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para testar a leitura do arquivo RAIS"""

import os
import pandas as pd
import re

def testar_leitura_rais():
    """Testa a leitura de um arquivo RAIS"""
    print("🔍 Testando leitura de arquivo RAIS...")
    
    # Procurar arquivo RAIS descompactado
    dir_descompactados = 'dados/rais/descompactados'
    if not os.path.exists(dir_descompactados):
        print(f"❌ Diretório não encontrado: {dir_descompactados}")
        return
    
    # Procurar primeiro arquivo .txt
    arquivos_txt = [f for f in os.listdir(dir_descompactados) if f.endswith('.txt')]
    if not arquivos_txt:
        print(f"❌ Nenhum arquivo .txt encontrado em {dir_descompactados}")
        return
    
    arquivo_teste = os.path.join(dir_descompactados, arquivos_txt[0])
    print(f"📁 Testando arquivo: {arquivos_txt[0]}")
    
    try:
        # Determinar o ano do arquivo pelo nome
        nome_arquivo = os.path.basename(arquivo_teste)
        ano_match = re.search(r'(\d{4})', nome_arquivo)
        ano = int(ano_match.group(1)) if ano_match else 2020
        
        print(f"📅 Ano detectado: {ano}")
        
        # Teste 1: Ler com header real
        print(f"\n🔄 Teste 1: Lendo com header real...")
        df = pd.read_csv(
            arquivo_teste,
            sep=';',
            encoding='latin1',
            dtype=str,
            na_values=['', ' '],
            low_memory=True,
            engine='c',
            skip_blank_lines=True,
            skipinitialspace=True,
            on_bad_lines='skip',
            header=0
        )
        
        print(f"✅ Arquivo lido com sucesso")
        print(f"📊 Shape: {df.shape}")
        print(f"📋 Total colunas: {len(df.columns)}")
        
        # Verificar se tem colunas Unnamed
        colunas_unnamed = df.columns.str.contains('^Unnamed').sum()
        if colunas_unnamed > 0:
            print(f"⚠️ Colunas Unnamed encontradas: {colunas_unnamed}")
        
        # Mostrar primeiras colunas
        print(f"📋 Primeiras 10 colunas:")
        for i, col in enumerate(df.columns[:10]):
            print(f"   {i}: {col}")
        
        # Procurar colunas essenciais
        print(f"\n🔍 Procurando colunas essenciais...")
        colunas_essenciais = ['idade', 'cbo_ocupacao', 'CBO Ocupação 2002', 'Idade']
        colunas_encontradas = [col for col in colunas_essenciais if col in df.columns]
        
        if colunas_encontradas:
            print(f"✅ Colunas essenciais encontradas: {colunas_encontradas}")
        else:
            print(f"⚠️ Nenhuma coluna essencial encontrada")
            
            # Procurar por padrões
            print(f"🔍 Procurando por padrões...")
            for col in df.columns:
                if 'idade' in col.lower():
                    print(f"   📋 Coluna de idade: {col}")
                if 'cbo' in col.lower():
                    print(f"   📋 Coluna de CBO: {col}")
        
        # Teste 2: Verificar primeiras linhas
        print(f"\n📄 Primeiras 3 linhas (primeiras 5 colunas):")
        print(df.iloc[:3, :5].to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return False


def main():
    """Executa teste"""
    print("="*60)
    print("TESTE DE LEITURA RAIS")
    print("="*60)
    
    sucesso = testar_leitura_rais()
    
    print("\n" + "="*60)
    if sucesso:
        print("✅ Teste concluído com sucesso")
    else:
        print("❌ Teste falhou")


if __name__ == "__main__":
    main() 
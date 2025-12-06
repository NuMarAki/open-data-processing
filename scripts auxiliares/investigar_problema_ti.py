#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para investigar o problema da classificação TI"""

import os
import pandas as pd
import numpy as np
from utils import CBOS_TI

def investigar_pnad():
    """Investiga dados do PNAD"""
    print("🔍 Investigando PNAD...")
    
    arquivo = 'resultados/pnad/dados_pnad_filtrado.parquet'
    if not os.path.exists(arquivo):
        print("   ❌ Arquivo PNAD não encontrado")
        return
    
    try:
        df = pd.read_parquet(arquivo, engine='pyarrow')
        print(f"   📊 Total registros: {len(df):,}")
        
        # Verificar colunas
        print(f"   📋 Colunas disponíveis: {list(df.columns)}")
        
        # Verificar CBOS
        if 'cbo_ocupacao' in df.columns:
            cbos_unicos = df['cbo_ocupacao'].astype(str).str[:4].unique()
            print(f"   📋 CBOS únicos encontrados: {len(cbos_unicos)}")
            print(f"   📋 Primeiros 10 CBOS: {cbos_unicos[:10].tolist()}")
            
            # Verificar quantos são TI
            cbos_ti_encontrados = [cbo for cbo in cbos_unicos if cbo in CBOS_TI]
            print(f"   📋 CBOS TI encontrados: {len(cbos_ti_encontrados)}")
            print(f"   📋 CBOS TI: {cbos_ti_encontrados}")
            
            # Verificar distribuição
            distribuicao_cbos = df['cbo_ocupacao'].astype(str).str[:4].value_counts().head(10)
            print(f"   📊 Top 10 CBOS por frequência:")
            for cbo, count in distribuicao_cbos.items():
                eh_ti = cbo in CBOS_TI
                status = "✅ TI" if eh_ti else "❌ Outros"
                print(f"      {cbo}: {count:,} registros {status}")
        
        # Verificar coluna eh_ti
        if 'eh_ti' in df.columns:
            ti_count = df['eh_ti'].sum()
            outros_count = (~df['eh_ti']).sum()
            print(f"   📊 Classificação atual:")
            print(f"      TI: {ti_count:,} ({ti_count/len(df)*100:.1f}%)")
            print(f"      Outros: {outros_count:,} ({outros_count/len(df)*100:.1f}%)")
            
            # Verificar se há inconsistência
            if ti_count == len(df):
                print(f"   ⚠️ PROBLEMA: Todos os registros estão marcados como TI!")
                
                # Recriar classificação para verificar
                print(f"   🔄 Recriando classificação...")
                df['eh_ti_novo'] = df['cbo_ocupacao'].astype(str).str[:4].isin(CBOS_TI)
                ti_novo = df['eh_ti_novo'].sum()
                outros_novo = (~df['eh_ti_novo']).sum()
                print(f"      Nova classificação:")
                print(f"      TI: {ti_novo:,} ({ti_novo/len(df)*100:.1f}%)")
                print(f"      Outros: {outros_novo:,} ({outros_novo/len(df)*100:.1f}%)")
                
                if ti_novo != ti_count:
                    print(f"   ✅ PROBLEMA IDENTIFICADO: Classificação inconsistente!")
                else:
                    print(f"   ❓ PROBLEMA NÃO IDENTIFICADO: Classificação parece correta")
        
    except Exception as e:
        print(f"   ❌ Erro ao investigar PNAD: {e}")


def investigar_caged():
    """Investiga dados do CAGED"""
    print("\n🔍 Investigando CAGED...")
    
    arquivo = 'resultados/caged/dados_caged_filtrado.parquet'
    if not os.path.exists(arquivo):
        print("   ❌ Arquivo CAGED não encontrado")
        return
    
    try:
        # Ler apenas uma amostra para não sobrecarregar memória
        df = pd.read_parquet(arquivo, engine='pyarrow')
        print(f"   📊 Total registros: {len(df):,}")
        
        # Verificar colunas
        print(f"   📋 Colunas disponíveis: {list(df.columns)}")
        
        # Verificar CBOS
        if 'cbo_ocupacao' in df.columns:
            # Amostra para análise
            amostra = df.sample(n=min(10000, len(df)), random_state=42)
            cbos_unicos = amostra['cbo_ocupacao'].astype(str).str[:4].unique()
            print(f"   📋 CBOS únicos na amostra: {len(cbos_unicos)}")
            print(f"   📋 Primeiros 10 CBOS: {cbos_unicos[:10].tolist()}")
            
            # Verificar quantos são TI
            cbos_ti_encontrados = [cbo for cbo in cbos_unicos if cbo in CBOS_TI]
            print(f"   📋 CBOS TI encontrados: {len(cbos_ti_encontrados)}")
            print(f"   📋 CBOS TI: {cbos_ti_encontrados}")
            
            # Verificar distribuição na amostra
            distribuicao_cbos = amostra['cbo_ocupacao'].astype(str).str[:4].value_counts().head(10)
            print(f"   📊 Top 10 CBOS por frequência (amostra):")
            for cbo, count in distribuicao_cbos.items():
                eh_ti = cbo in CBOS_TI
                status = "✅ TI" if eh_ti else "❌ Outros"
                print(f"      {cbo}: {count:,} registros {status}")
        
        # Verificar coluna eh_ti
        if 'eh_ti' in df.columns:
            ti_count = df['eh_ti'].sum()
            outros_count = (~df['eh_ti']).sum()
            print(f"   📊 Classificação atual:")
            print(f"      TI: {ti_count:,} ({ti_count/len(df)*100:.1f}%)")
            print(f"      Outros: {outros_count:,} ({outros_count/len(df)*100:.1f}%)")
            
            # Verificar se há inconsistência
            if ti_count == len(df):
                print(f"   ⚠️ PROBLEMA: Todos os registros estão marcados como TI!")
                
                # Recriar classificação para verificar
                print(f"   🔄 Recriando classificação...")
                df['eh_ti_novo'] = df['cbo_ocupacao'].astype(str).str[:4].isin(CBOS_TI)
                ti_novo = df['eh_ti_novo'].sum()
                outros_novo = (~df['eh_ti_novo']).sum()
                print(f"      Nova classificação:")
                print(f"      TI: {ti_novo:,} ({ti_novo/len(df)*100:.1f}%)")
                print(f"      Outros: {outros_novo:,} ({outros_novo/len(df)*100:.1f}%)")
                
                if ti_novo != ti_count:
                    print(f"   ✅ PROBLEMA IDENTIFICADO: Classificação inconsistente!")
                else:
                    print(f"   ❓ PROBLEMA NÃO IDENTIFICADO: Classificação parece correta")
        
    except Exception as e:
        print(f"   ❌ Erro ao investigar CAGED: {e}")


def verificar_processamento():
    """Verifica se há problema no processamento"""
    print("\n🔍 Verificando processamento...")
    
    # Verificar se os arquivos foram filtrados corretamente
    arquivos_filtrados = [
        'resultados/pnad/dados_pnad_filtrado.parquet',
        'resultados/caged/dados_caged_filtrado.parquet'
    ]
    
    for arquivo in arquivos_filtrados:
        if os.path.exists(arquivo):
            print(f"   📁 {arquivo}: {os.path.getsize(arquivo)/(1024*1024):.1f} MB")
        else:
            print(f"   ❌ {arquivo}: Não encontrado")
    
    # Verificar se há arquivos preprocessados
    print(f"\n   📁 Verificando arquivos preprocessados...")
    for base in ['pnad', 'caged']:
        diretorio = f'dados_preprocessados/{base}'
        if os.path.exists(diretorio):
            arquivos = [f for f in os.listdir(diretorio) if f.endswith(('.parquet', '.csv'))]
            print(f"      {base}: {len(arquivos)} arquivos")
        else:
            print(f"      {base}: Diretório não encontrado")


def main():
    """Executa investigação completa"""
    print("="*60)
    print("INVESTIGAÇÃO DO PROBLEMA DE CLASSIFICAÇÃO TI")
    print("="*60)
    
    investigar_pnad()
    investigar_caged()
    verificar_processamento()
    
    print("\n" + "="*60)
    print("CONCLUSÕES")
    print("="*60)
    
    print("💡 Possíveis causas do problema:")
    print("1. Arquivos já foram filtrados apenas para TI durante o processamento")
    print("2. Problema na lógica de classificação durante o processamento")
    print("3. CBOS nos dados não correspondem aos CBOS TI definidos")
    print("4. Problema na formatação dos CBOS nos dados")
    
    print("\n🔧 Soluções possíveis:")
    print("1. Reprocessar dados originais sem filtrar apenas TI")
    print("2. Verificar se o filtro TI foi aplicado incorretamente")
    print("3. Ajustar CBOS TI se necessário")
    print("4. Verificar formatação dos CBOS nos dados originais")


if __name__ == "__main__":
    main() 
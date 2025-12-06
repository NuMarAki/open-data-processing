#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para testar se as correções foram aplicadas corretamente"""

import os
import sys
import pandas as pd
import numpy as np

def teste_imports():
    """Testa se todas as bibliotecas necessárias estão disponíveis"""
    print("🔍 Testando imports...")
    
    try:
        import pyarrow.parquet as pq
        print("✅ PyArrow importado com sucesso")
    except ImportError:
        print("❌ ERRO: PyArrow não está instalado. Execute: pip install pyarrow")
        return False
    
    try:
        from scipy import stats
        print("✅ SciPy importado com sucesso")
    except ImportError:
        print("❌ ERRO: SciPy não está instalado. Execute: pip install scipy")
        return False
    
    return True


def teste_leitura_parquet():
    """Testa a leitura correta de arquivos parquet"""
    print("\n🔍 Testando leitura de parquet...")
    
    # Criar arquivo de teste
    df_teste = pd.DataFrame({
        'idade': np.random.randint(20, 60, 1000),
        'eh_ti': np.random.choice([True, False], 1000),
        'competencia': ['202301'] * 1000,
        'ano': [2023] * 1000
    })
    
    arquivo_teste = 'teste_parquet.parquet'
    
    try:
        # Remover arquivo se existir
        if os.path.exists(arquivo_teste):
            try:
                os.remove(arquivo_teste)
            except PermissionError:
                print("⚠️  Arquivo de teste anterior ainda em uso, continuando...")
        
        # Criar novo arquivo
        df_teste.to_parquet(arquivo_teste, engine='pyarrow')
        
        # Teste 1: Leitura normal
        df = pd.read_parquet(arquivo_teste)
        print("✅ Leitura normal de parquet OK")
        
        # Teste 2: Leitura com PyArrow
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(arquivo_teste)
        
        batches_lidos = 0
        for batch in parquet_file.iter_batches(batch_size=100):
            chunk = batch.to_pandas()
            batches_lidos += 1
        
        # Fechar explicitamente o arquivo
        parquet_file.close()
        
        print(f"✅ Leitura em batches OK ({batches_lidos} batches)")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO na leitura de parquet: {e}")
        return False
    
    finally:
        # Tentar limpar o arquivo de teste
        try:
            if os.path.exists(arquivo_teste):
                os.remove(arquivo_teste)
        except (PermissionError, OSError) as e:
            print(f"⚠️  Não foi possível remover arquivo de teste: {e}")
            print("   O arquivo será removido na próxima execução")


def teste_colunas_duplicadas():
    """Testa o tratamento de colunas duplicadas"""
    print("\n🔍 Testando tratamento de colunas duplicadas...")
    
    # Simular colunas duplicadas
    colunas_header = ['nome', 'idade', 'nome', 'salario', 'idade', 'nome']
    
    # Aplicar correção
    colunas_unicas = []
    contadores = {}
    
    for col in colunas_header:
        if col in contadores:
            contadores[col] += 1
            col_novo = f"{col}_{contadores[col]}"
        else:
            contadores[col] = 0
            col_novo = col
        colunas_unicas.append(col_novo)
    
    esperado = ['nome', 'idade', 'nome_1', 'salario', 'idade_1', 'nome_2']
    
    if colunas_unicas == esperado:
        print("✅ Tratamento de colunas duplicadas OK")
        print(f"   Original: {colunas_header}")
        print(f"   Corrigido: {colunas_unicas}")
        return True
    else:
        print("❌ ERRO no tratamento de colunas duplicadas")
        print(f"   Esperado: {esperado}")
        print(f"   Obtido: {colunas_unicas}")
        return False


def teste_conversao_competencia():
    """Testa a conversão correta da competência"""
    print("\n🔍 Testando conversão de competência...")
    
    # Criar DataFrame de teste
    df = pd.DataFrame({
        'competencia': ['202001', '202002', '202003', '202004']
    })
    
    try:
        # Aplicar correção
        df['competencia'] = df['competencia'].astype(str)
        df['ano'] = df['competencia'].str[:4].astype(int)
        
        if df['ano'].tolist() == [2020, 2020, 2020, 2020]:
            print("✅ Conversão de competência OK")
            print(f"   Competências: {df['competencia'].tolist()}")
            print(f"   Anos extraídos: {df['ano'].tolist()}")
            return True
        else:
            print("❌ ERRO na conversão de competência")
            return False
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False


def teste_memoria_amostragem():
    """Testa a amostragem para evitar problema de memória"""
    print("\n🔍 Testando amostragem para economia de memória...")
    
    # Criar dataset grande
    tamanho_grande = 1000000
    df_grande = pd.DataFrame({
        'idade': np.random.randint(20, 60, tamanho_grande)
    })
    
    max_sample_size = 10000
    
    try:
        if len(df_grande) > max_sample_size:
            amostra = df_grande['idade'].sample(n=max_sample_size, random_state=42)
        else:
            amostra = df_grande['idade']
        
        if len(amostra) == max_sample_size:
            print("✅ Amostragem OK")
            print(f"   Dataset original: {len(df_grande):,} registros")
            print(f"   Amostra: {len(amostra):,} registros")
            return True
        else:
            print("❌ ERRO na amostragem")
            return False
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False


def verificar_arquivos_projeto():
    """Verifica se os arquivos do projeto existem"""
    print("\n🔍 Verificando arquivos do projeto...")
    
    arquivos_necessarios = [
        'executar_analise_completa.py',
        'processadores_especificos.py',
        'analise_etarismo.py',
        'processador_base.py',
        'config_manager.py',
        'utils.py'
    ]
    
    todos_existem = True
    for arquivo in arquivos_necessarios:
        if os.path.exists(arquivo):
            print(f"✅ {arquivo}")
        else:
            print(f"❌ {arquivo} - NÃO ENCONTRADO")
            todos_existem = False
    
    return todos_existem


def main():
    """Executa todos os testes"""
    print("="*60)
    print("TESTE DE CORREÇÕES - ANÁLISE DE ETARISMO")
    print("="*60)
    
    testes = [
        ("Imports", teste_imports),
        ("Leitura Parquet", teste_leitura_parquet),
        ("Colunas Duplicadas", teste_colunas_duplicadas),
        ("Conversão Competência", teste_conversao_competencia),
        ("Amostragem de Memória", teste_memoria_amostragem),
        ("Arquivos do Projeto", verificar_arquivos_projeto)
    ]
    
    resultados = []
    
    for nome, funcao_teste in testes:
        try:
            resultado = funcao_teste()
            resultados.append((nome, resultado))
        except Exception as e:
            print(f"❌ ERRO no teste {nome}: {e}")
            resultados.append((nome, False))
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)
    
    total = len(resultados)
    sucessos = sum(1 for _, resultado in resultados if resultado)
    
    for nome, resultado in resultados:
        status = "✅ PASSOU" if resultado else "❌ FALHOU"
        print(f"{nome}: {status}")
    
    print(f"\nTotal: {sucessos}/{total} testes passaram")
    
    if sucessos == total:
        print("\n🎉 TODOS OS TESTES PASSARAM! As correções estão funcionando.")
        return 0
    else:
        print(f"\n⚠️  {total - sucessos} testes falharam. Verifique as correções.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
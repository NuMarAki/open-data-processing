#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script para testar a classificação TI nos dados"""

import os
import pandas as pd
import numpy as np
from utils import CBOS_TI, preparar_dados_ti

def testar_classificacao_ti():
    """Testa a classificação TI em dados de exemplo"""
    print("🔍 Testando classificação TI...")
    
    # Criar dados de teste
    dados_teste = {
        'idade': [25, 30, 35, 40, 45, 50],
        'cbo_ocupacao': ['1425', '2120', '3171', '9999', '2141', '1234']  # 4 TI, 2 não-TI
    }
    
    df_teste = pd.DataFrame(dados_teste)
    print(f"   📊 Dados de teste criados: {len(df_teste)} registros")
    print(f"   📋 CBOS: {dados_teste['cbo_ocupacao']}")
    
    # Aplicar classificação
    df_processado = preparar_dados_ti(df_teste)
    
    if 'eh_ti' in df_processado.columns:
        ti_count = df_processado['eh_ti'].sum()
        outros_count = (~df_processado['eh_ti']).sum()
        
        print(f"   ✅ Classificação aplicada:")
        print(f"      TI: {ti_count} registros")
        print(f"      Outros: {outros_count} registros")
        print(f"      Percentual TI: {(ti_count/len(df_processado)*100):.1f}%")
        
        # Verificar cada CBO
        print(f"   📋 Verificação por CBO:")
        for i, cbo in enumerate(dados_teste['cbo_ocupacao']):
            eh_ti = df_processado.iloc[i]['eh_ti']
            status = "✅ TI" if eh_ti else "❌ Outros"
            print(f"      {cbo}: {status}")
        
        return True
    else:
        print(f"   ❌ Coluna 'eh_ti' não foi criada")
        return False


def verificar_dados_reais():
    """Verifica dados reais se existirem"""
    print("\n🔍 Verificando dados reais...")
    
    # Procurar arquivos consolidados
    arquivos_consolidados = []
    for base in ['PNAD', 'RAIS', 'CAGED']:
        caminho = f'resultados/{base.lower()}/consolidado_{base.lower()}_consolidado.parquet'
        if os.path.exists(caminho):
            arquivos_consolidados.append((base, caminho))
    
    if not arquivos_consolidados:
        print("   ⚠️ Nenhum arquivo consolidado encontrado")
        return
    
    for base, arquivo in arquivos_consolidados:
        print(f"\n   📊 Verificando {base}...")
        try:
            # Ler apenas uma amostra para verificar
            df = pd.read_parquet(arquivo, engine='pyarrow')
            
            print(f"      Total registros: {len(df):,}")
            
            if 'eh_ti' in df.columns:
                ti_count = df['eh_ti'].sum()
                outros_count = (~df['eh_ti']).sum()
                total = len(df)
                
                print(f"      TI: {ti_count:,} ({ti_count/total*100:.1f}%)")
                print(f"      Outros: {outros_count:,} ({outros_count/total*100:.1f}%)")
                
                # Verificar se há problema
                if ti_count == total:
                    print(f"      ⚠️ PROBLEMA: Todos os registros estão marcados como TI!")
                elif ti_count == 0:
                    print(f"      ⚠️ PROBLEMA: Nenhum registro está marcado como TI!")
                else:
                    print(f"      ✅ Classificação parece normal")
                
                # Verificar CBOS únicos se disponível
                if 'cbo_ocupacao' in df.columns:
                    cbos_unicos = df['cbo_ocupacao'].astype(str).str[:4].unique()
                    cbos_ti_encontrados = [cbo for cbo in cbos_unicos if cbo in CBOS_TI]
                    print(f"      📋 CBOS únicos encontrados: {len(cbos_unicos)}")
                    print(f"      📋 CBOS TI encontrados: {len(cbos_ti_encontrados)}")
                    print(f"      📋 CBOS TI: {cbos_ti_encontrados[:10]}...")  # Mostrar apenas os primeiros
                
                if 'cbo_familia' in df.columns:
                    print(f"      ✅ cbo_familia presente. Amostra: {df['cbo_familia'].dropna().head(5).tolist()}")
                else:
                    print("      ⚠️ cbo_familia ausente – verificar pipeline PNAD.")

                
            else:
                print(f"      ❌ Coluna 'eh_ti' não encontrada")
                
        except Exception as e:
            print(f"      ❌ Erro ao verificar {base}: {e}")


def verificar_cbos_ti():
    """Verifica se os CBOS TI estão corretos"""
    print("\n🔍 Verificando CBOS TI...")
    
    print(f"   📋 CBOS TI definidos: {CBOS_TI}")
    print(f"   📊 Total CBOS TI: {len(CBOS_TI)}")
    
    # Verificar se são válidos (4 dígitos)
    cbos_invalidos = [cbo for cbo in CBOS_TI if not cbo.isdigit() or len(cbo) != 4]
    if cbos_invalidos:
        print(f"   ⚠️ CBOS inválidos encontrados: {cbos_invalidos}")
    else:
        print(f"   ✅ Todos os CBOS são válidos")


def main():
    """Executa todos os testes"""
    print("="*60)
    print("TESTE DE CLASSIFICAÇÃO TI")
    print("="*60)
    
    # Teste com dados sintéticos
    teste_sintetico = testar_classificacao_ti()
    
    # Verificar CBOS TI
    verificar_cbos_ti()
    
    # Verificar dados reais
    verificar_dados_reais()
    
    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)
    
    if teste_sintetico:
        print("✅ Teste sintético passou")
    else:
        print("❌ Teste sintético falhou")
    
    print("\n💡 Recomendações:")
    print("1. Se todos os registros estão marcados como TI, verificar:")
    print("   - Se a coluna CBO está correta")
    print("   - Se os CBOS TI estão corretos")
    print("   - Se há problema na lógica de classificação")
    print("2. Se nenhum registro está marcado como TI, verificar:")
    print("   - Se os CBOS nos dados correspondem aos CBOS TI")
    print("   - Se há problema na formatação dos CBOS")


if __name__ == "__main__":
    main() 
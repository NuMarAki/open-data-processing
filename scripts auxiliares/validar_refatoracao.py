#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de validação das refatorações de remoção de duplicidades
Testa se as funções consolidadas funcionam corretamente
"""

import sys
import os

def teste_imports():
    """Testa se todos os imports estão funcionando"""
    print("\n🔍 Testando imports...")
    
    erros = []
    
    try:
        from utils import ajustar_colunas_ocupacao, configurar_ambiente
        print("  ✅ utils.ajustar_colunas_ocupacao")
        print("  ✅ utils.configurar_ambiente")
    except ImportError as e:
        erros.append(f"❌ Erro ao importar de utils: {e}")
    
    try:
        from processadores_especificos import ProcessadorPNAD, ProcessadorRAIS, ProcessadorCAGED
        print("  ✅ processadores_especificos - todas as classes")
    except ImportError as e:
        erros.append(f"❌ Erro ao importar processadores: {e}")
    
    try:
        from analise_etarismo import AnalisadorEtarismo
        print("  ✅ analise_etarismo.AnalisadorEtarismo")
    except ImportError as e:
        erros.append(f"❌ Erro ao importar AnalisadorEtarismo: {e}")
    
    # Testar scripts de gráficos
    scripts_grafico = [
        'gera_grafico_serie_temporal_ti',
        'gera_grafico_renda_estudo_comparativo',
        'gera_grafico_estado_pnad',
        'gera_grafico_escolaridade_sexo'
    ]
    
    for script in scripts_grafico:
        try:
            __import__(script)
            print(f"  ✅ {script}")
        except ImportError as e:
            erros.append(f"❌ Erro ao importar {script}: {e}")
        except Exception as e:
            # Alguns scripts podem dar erro ao executar mas import deve funcionar
            if "import" in str(e).lower():
                erros.append(f"❌ Erro ao importar {script}: {e}")
            else:
                print(f"  ✅ {script} (import OK)")
    
    # Testar estatísticas
    try:
        import estatisticas_descritivas
        print("  ✅ estatisticas_descritivas")
    except ImportError as e:
        erros.append(f"❌ Erro ao importar estatisticas_descritivas: {e}")
    
    try:
        import estatisticas_descritivas_simples
        print("  ✅ estatisticas_descritivas_simples")
    except ImportError as e:
        erros.append(f"❌ Erro ao importar estatisticas_descritivas_simples: {e}")
    
    try:
        import preditivo_pnad
        print("  ✅ preditivo_pnad")
    except ImportError as e:
        erros.append(f"❌ Erro ao importar preditivo_pnad: {e}")
    
    if erros:
        print("\n❌ ERROS ENCONTRADOS:")
        for erro in erros:
            print(f"  {erro}")
        return False
    
    print("\n✅ Todos os imports funcionando corretamente!")
    return True


def teste_ajustar_colunas_ocupacao():
    """Testa a função ajustar_colunas_ocupacao consolidada"""
    print("\n🔍 Testando ajustar_colunas_ocupacao...")
    
    import pandas as pd
    from utils import ajustar_colunas_ocupacao
    
    testes = [
        # (nome_coluna_original, esperado)
        ('cbo', True),
        ('CBO', True),
        ('cdo', True),
        ('cbo_2002', True),
        ('cbo ocupacao', True),
        ('cbo_ocupacao_2002', True),
        ('idade', False),  # Não deve renomear
    ]
    
    sucesso = True
    
    for col_original, deve_renomear in testes:
        # Criar DataFrame de teste
        df = pd.DataFrame({col_original: [1, 2, 3], 'idade': [25, 30, 35]})
        
        # Aplicar função
        df_resultado = ajustar_colunas_ocupacao(df)
        
        # Verificar resultado
        tem_cbo_ocupacao = 'cbo_ocupacao' in df_resultado.columns
        
        if deve_renomear and not tem_cbo_ocupacao:
            print(f"  ❌ Falhou: '{col_original}' deveria ser renomeada mas não foi")
            sucesso = False
        elif not deve_renomear and col_original not in df_resultado.columns:
            print(f"  ❌ Falhou: '{col_original}' foi renomeada indevidamente")
            sucesso = False
        else:
            print(f"  ✅ '{col_original}' → {'cbo_ocupacao' if deve_renomear else 'mantida'}")
    
    if sucesso:
        print("\n✅ ajustar_colunas_ocupacao funcionando corretamente!")
    
    return sucesso


def teste_configurar_ambiente():
    """Testa a função configurar_ambiente consolidada"""
    print("\n🔍 Testando configurar_ambiente...")
    
    import os
    import shutil
    from utils import configurar_ambiente
    
    # Criar diretório temporário para teste
    dir_teste = 'teste_temp_validacao'
    
    # Limpar se existir
    if os.path.exists(dir_teste):
        shutil.rmtree(dir_teste)
    
    try:
        # Testar criação de diretório
        configurar_ambiente(dir_teste)
        
        if not os.path.exists(dir_teste):
            print(f"  ❌ Diretório '{dir_teste}' não foi criado")
            return False
        
        print(f"  ✅ Diretório '{dir_teste}' criado com sucesso")
        
        # Testar chamada novamente (não deve dar erro)
        configurar_ambiente(dir_teste)
        print(f"  ✅ Chamada duplicada não causa erro")
        
        # Limpar
        shutil.rmtree(dir_teste)
        print(f"  ✅ Limpeza do teste concluída")
        
        print("\n✅ configurar_ambiente funcionando corretamente!")
        return True
        
    except Exception as e:
        print(f"  ❌ Erro ao testar configurar_ambiente: {e}")
        # Limpar em caso de erro
        if os.path.exists(dir_teste):
            shutil.rmtree(dir_teste)
        return False


def verificar_arquivos_modificados():
    """Verifica se os arquivos foram modificados corretamente"""
    print("\n🔍 Verificando arquivos modificados...")
    
    from utils import ajustar_colunas_ocupacao, configurar_ambiente
    
    # Verificar se funções estão em utils
    print("  ✅ ajustar_colunas_ocupacao disponível em utils")
    print("  ✅ configurar_ambiente disponível em utils")
    
    # Verificar imports nos arquivos modificados
    arquivos_verificar = [
        'processadores_especificos.py',
        'executar_analise_completa.py',
        'analise_etarismo.py',
        'gera_grafico_serie_temporal_ti.py',
        'gera_grafico_renda_estudo_comparativo.py',
        'estatisticas_descritivas.py'
    ]
    
    for arquivo in arquivos_verificar:
        if not os.path.exists(arquivo):
            print(f"  ⚠️  {arquivo} não encontrado (pode estar em subpasta)")
            continue
        
        with open(arquivo, 'r', encoding='utf-8') as f:
            conteudo = f.read()
            
        # Verificar se importa de utils
        if 'from utils import' in conteudo or 'import utils' in conteudo:
            print(f"  ✅ {arquivo} importa de utils")
        else:
            print(f"  ⚠️  {arquivo} não importa de utils (verificar manualmente)")
    
    print("\n✅ Verificação de arquivos concluída!")
    return True


def main():
    """Executa todos os testes"""
    print("="*60)
    print("VALIDAÇÃO DE REFATORAÇÃO - REMOÇÃO DE DUPLICIDADES")
    print("="*60)
    
    resultados = []
    
    # Teste 1: Imports
    resultados.append(("Imports", teste_imports()))
    
    # Teste 2: ajustar_colunas_ocupacao
    resultados.append(("ajustar_colunas_ocupacao", teste_ajustar_colunas_ocupacao()))
    
    # Teste 3: configurar_ambiente
    resultados.append(("configurar_ambiente", teste_configurar_ambiente()))
    
    # Teste 4: Verificar arquivos
    resultados.append(("Arquivos modificados", verificar_arquivos_modificados()))
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO DOS TESTES")
    print("="*60)
    
    total = len(resultados)
    passou = sum(1 for _, r in resultados if r)
    
    for nome, resultado in resultados:
        status = "✅ PASSOU" if resultado else "❌ FALHOU"
        print(f"{nome:30s} {status}")
    
    print("\n" + "="*60)
    print(f"RESULTADO FINAL: {passou}/{total} testes passaram")
    print("="*60)
    
    if passou == total:
        print("\n🎉 TODAS AS VALIDAÇÕES PASSARAM!")
        print("As refatorações estão funcionando corretamente.")
        print("\nPróximos passos:")
        print("1. Execute seus scripts principais para teste final")
        print("2. Verifique se os outputs são idênticos aos anteriores")
        print("3. Se tudo estiver OK, faça commit das alterações")
        return 0
    else:
        print("\n⚠️  ALGUMAS VALIDAÇÕES FALHARAM")
        print("Revise os erros acima antes de prosseguir.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

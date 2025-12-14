#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processador de arquivo único PNAD para validar layout de posições
Usa o ProcessadorPNAD existente mas processa apenas 1 arquivo específico
"""

import os
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar classes do seu sistema
from processamento.processadores_especificos import ProcessadorPNAD
from scripts.config_manager import ConfigManager
from scripts.utils import logger, configurar_log

def processar_arquivo_unico_pnad(nome_arquivo, salvar_csv=True, mostrar_amostra=True):
    """
    Processa um único arquivo PNAD para validar layout de posições
    
    Args:
        nome_arquivo (str): Nome do arquivo (ex: "PNADC_2021_01.txt")
        salvar_csv (bool): Se deve salvar CSV processado
        mostrar_amostra (bool): Se deve mostrar amostra dos dados
        
    Returns:
        str: Caminho do arquivo processado gerado
    """
    print(f"🔍 PROCESSADOR DE ARQUIVO ÚNICO PNAD")
    print(f"📁 Arquivo: {nome_arquivo}")
    print("="*60)
    
    # Configurar log
    configurar_log()
    
    try:
        # Carregar configuração PNAD
        config_manager = ConfigManager()
        config = config_manager.carregar_configuracao('pnad', 'colunas_pnad.cfg')
        
        print(f"✅ Configuração carregada:")
        print(f"   • Layout: {len(config.layout)} posições definidas")
        
        # CBOs TI - usando a mesma definição dos seus processadores
        cbo_ti = [
            '1425',  # Gerentes de tecnologia da informação
            '2120', '2122', '2123', '2124', '2125',  # Profissionais de nível superior
            '3171', '3172', '3173'  # Técnicos
        ]
        print(f"   • CBOs TI: {cbo_ti}")
        print(f"   • Variáveis: {config.variaveis}")
        
        # Criar processador
        processador = ProcessadorPNAD(config)
        
        # Definir caminhos
        caminho_origem = Path("C:/TCC/dados/pnad/temp")
        arquivo_completo = caminho_origem / nome_arquivo
        
        # Verificar se arquivo existe
        if not arquivo_completo.exists():
            # Tentar encontrar arquivo similar
            arquivos_disponiveis = list(caminho_origem.glob("*.txt"))
            print(f"❌ Arquivo não encontrado: {arquivo_completo}")
            print(f"📂 Arquivos disponíveis em {caminho_origem}:")
            for arq in arquivos_disponiveis[:10]:  # Mostrar até 10
                print(f"   • {arq.name}")
            return None
        
        print(f"📂 Processando: {arquivo_completo}")
        print(f"📏 Tamanho: {arquivo_completo.stat().st_size / (1024*1024):.1f} MB")
        
        # PROCESSAR ARQUIVO USANDO SEU PROCESSADOR EXISTENTE
        df = processador.processar_arquivo(str(arquivo_completo))
        
        if df is None or df.empty:
            print("❌ Erro no processamento ou arquivo vazio")
            return None
        
        print(f"✅ Processamento concluído!")
        print(f"📊 Registros processados: {len(df):,}")
        print(f"📋 Colunas: {list(df.columns)}")
        
        # Mostrar estatísticas básicas
        if 'idade' in df.columns:
            idades_validas = df['idade'].dropna()
            if len(idades_validas) > 0:
                print(f"👥 Idades: {idades_validas.min()}-{idades_validas.max()} anos (média: {idades_validas.mean():.1f})")
        
        # Identificar coluna de CBO (pode ter nomes diferentes)
        coluna_cbo = None
        for col in df.columns:
            if 'cbo' in col.lower() or 'ocupacao' in col.lower() or 'v4010' in col.lower():
                coluna_cbo = col
                break
        
        if coluna_cbo:
            print(f"💼 Coluna CBO encontrada: {coluna_cbo}")
            cbos_validos = df[coluna_cbo].dropna()
            print(f"💼 CBOs identificados: {len(cbos_validos):,}")
            
            # CBOs de TI
            cbos_ti_df = df[df[coluna_cbo].isin(cbo_ti)]
            print(f"💻 Profissionais TI identificados: {len(cbos_ti_df):,}")
            
            if len(cbos_ti_df) > 0:
                print("💻 CBOs TI encontrados:")
                for cbo, count in cbos_ti_df[coluna_cbo].value_counts().items():
                    print(f"   • {cbo}: {count:,}")
            
            # Mostrar CBOs mais comuns
            print("📊 CBOs mais comuns (top 10):")
            for cbo, count in cbos_validos.value_counts().head(10).items():
                ti_flag = "🔥 TI" if str(cbo) in cbo_ti else ""
                print(f"   • {cbo}: {count:,} {ti_flag}")
        
        # Definir nome do arquivo de saída
        nome_base = nome_arquivo.replace('.txt', '')
        arquivo_saida = caminho_origem / f"{nome_base}_processado.csv"
        
        if salvar_csv:
            print(f"💾 Salvando: {arquivo_saida}")
            df.to_csv(arquivo_saida, sep=';', index=False, encoding='utf-8')
            print(f"✅ Arquivo salvo: {arquivo_saida}")
        
        # Mostrar amostra dos dados
        if mostrar_amostra and len(df) > 0:
            print(f"\n📋 AMOSTRA DOS DADOS (5 primeiros registros):")
            print("="*80)
            
            # Mostrar primeiras colunas importantes
            colunas_importantes = []
            for col in df.columns:
                if any(palavra in col.lower() for palavra in ['ano', 'trimestre', 'uf', 'idade', 'sexo', 'cbo', 'renda', 'curso']):
                    colunas_importantes.append(col)
            
            if colunas_importantes:
                amostra = df[colunas_importantes[:8]].head(5)  # Limitar a 8 colunas
                print(amostra.to_string(index=False))
            else:
                # Se não encontrar, mostrar primeiras 8 colunas
                amostra = df.iloc[:, :8].head(5)
                print(amostra.to_string(index=False))
        
        # Mostrar informações sobre layout aplicado
        print(f"\n🔧 LAYOUT APLICADO:")
        print("="*50)
        for var, posicoes in config.layout.items():
            inicio, fim = posicoes
            
            # Procurar coluna correspondente no DataFrame
            coluna_encontrada = None
            for col in df.columns:
                if var.lower() in col.lower() or col.lower() in var.lower():
                    coluna_encontrada = col
                    break
            
            if coluna_encontrada:
                print(f"   {var:12} = {inicio:3}-{fim:3} → ✅ {coluna_encontrada}")
            else:
                print(f"   {var:12} = {inicio:3}-{fim:3} → ❌ NÃO ENCONTRADO")
        
        # VALIDAÇÃO ESPECÍFICA PARA LAYOUT - usando nomes reais das colunas
        print(f"\n🎯 VALIDAÇÃO ESPECÍFICA DO LAYOUT:")
        print("="*50)
        
        # Encontrar colunas importantes e mostrar valores de amostra
        validacoes_importantes = [
            ('V2009', ['idade', 'v2009'], 'Idade'),
            ('V4010', ['cbo', 'ocupacao', 'v4010'], 'CBO Ocupação'),
            ('V2007', ['sexo', 'v2007'], 'Sexo'),
            ('V3009A', ['curso', 'estudo', 'v3009a'], 'Curso Elevado'),
            ('V1028', ['peso', 'v1028'], 'Peso Populacional'),
        ]
        
        for var_layout, palavras_chave, descricao in validacoes_importantes:
            if var_layout in config.layout:
                inicio, fim = config.layout[var_layout]
                
                # Procurar coluna correspondente
                coluna_encontrada = None
                for col in df.columns:
                    for palavra in palavras_chave:
                        if palavra.lower() in col.lower():
                            coluna_encontrada = col
                            break
                    if coluna_encontrada:
                        break
                
                if coluna_encontrada:
                    valores_sample = df[coluna_encontrada].dropna().head(5).tolist()
                    print(f"✅ {descricao:15} ({var_layout} pos.{inicio}-{fim}) → {coluna_encontrada}")
                    print(f"   Amostra: {valores_sample}")
                else:
                    print(f"❌ {descricao:15} ({var_layout} pos.{inicio}-{fim}) → NÃO ENCONTRADA")
        
        return str(arquivo_saida) if salvar_csv else None
        
    except Exception as e:
        print(f"❌ Erro no processamento: {e}")
        logger.error(f"Erro ao processar {nome_arquivo}: {e}")
        import traceback
        print(f"Detalhes do erro: {traceback.format_exc()}")
        return None

def main():
    """Função principal - permite executar via linha de comando"""
    
    if len(sys.argv) < 2:
        print("Uso: python processar_arquivo_unico.py <nome_do_arquivo>")
        print("Exemplo: python processar_arquivo_unico.py PNADC_012021.txt")
        
        # Listar arquivos disponíveis
        caminho_pnad = Path("C:/TCC/dados/pnad/temp")
        if caminho_pnad.exists():
            arquivos = list(caminho_pnad.glob("*.txt"))
            if arquivos:
                print(f"\n📂 Arquivos disponíveis:")
                for i, arquivo in enumerate(arquivos[:10], 1):
                    print(f"   {i:2}. {arquivo.name}")
        return
    
    nome_arquivo = sys.argv[1]
    
    # Processar arquivo
    arquivo_gerado = processar_arquivo_unico_pnad(
        nome_arquivo=nome_arquivo,
        salvar_csv=True,
        mostrar_amostra=True
    )
    
    if arquivo_gerado:
        print(f"\n🎉 PROCESSAMENTO CONCLUÍDO!")
        print(f"📁 Arquivo gerado: {arquivo_gerado}")
        print(f"\n💡 Para visualizar o resultado:")
        print(f"   • Abra o arquivo: {arquivo_gerado}")
        print(f"   • Use Excel, LibreOffice ou editor de texto")
        print(f"   • Separador: ponto e vírgula (;)")
    else:
        print(f"\n❌ PROCESSAMENTO FALHOU!")

if __name__ == "__main__":
    main()
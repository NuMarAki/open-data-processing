#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Menu interativo principal - Sistema refatorado mantendo a interface original"""

import os
import sys
from datetime import datetime
import shutil
import pandas as pd
import pyarrow.parquet as pq
import glob
from typing import Optional, List, Dict
import config_manager
from utils import CBOS_TI, criar_caminho, logger
import matplotlib.pyplot as plt
import gc
import configparser
import unicodedata
import re
from processadores_especificos import ProcessadorPNAD, ProcessadorRAIS, ProcessadorCAGED
from processador_base import ProcessadorSeguro
from analise_etarismo import AnalisadorEtarismo
from descompactador import Descompactador
from utils import (
    ler_arquivo_com_fallback, logger, configurar_log, monitorar_recursos, verificar_espaco_disco,
    limpar_diretorio, arquivo_existe_e_tamanho_ok, limpar_temp_personalizado, listar_erros_descompactacao,
    criar_caminho, CBOS_TI
)

diretorio_temp = r'Z:\TCC\TEMP'

try:
    os.makedirs(diretorio_temp, exist_ok=True)
    print(f"Diretório '{diretorio_temp}' garantido.")
except OSError as e:
    print(f"Erro ao criar o diretório '{diretorio_temp}': {e}")

os.environ['TEMP'] = diretorio_temp
os.environ['TMP'] = diretorio_temp

def _processar_chunk_para_metricas(chunk: pd.DataFrame, colunas_essenciais: List[str]) -> pd.DataFrame:
    """Normaliza um chunk e cria eh_ti se necessário."""
    # Adiciona colunas faltantes em lote (evita fragmentação)
    faltantes = [c for c in colunas_essenciais if c not in chunk.columns]
    if faltantes:
        chunk[faltantes] = None

    # Cria eh_ti se faltar, a partir de cbo_ocupacao
    if 'eh_ti' not in chunk.columns:
        if 'cbo_ocupacao' in chunk.columns:
            cbo = pd.to_numeric(chunk['cbo_ocupacao'], errors='coerce').astype('Int64').astype(str)
            chunk['eh_ti'] = cbo.isin(CBOS_TI)
        else:
            chunk['eh_ti'] = False

    return chunk

def _iterar_parquet_em_batches(arquivo: str, colunas: List[str], batch_size: int = 200_000):
    """Itera Parquet em batches usando PyArrow para reduzir pico de memória."""
    pf = pq.ParquetFile(arquivo)
    cols = [c for c in colunas if c in pf.schema.names]
    for batch in pf.iter_batches(columns=cols or None, batch_size=batch_size):
        yield batch.to_pandas(types_mapper=pd.ArrowDtype, self_destruct=True)

def verificar_arquivo_consolidado_existe(tipo):
    """Verifica se já existe arquivo consolidado para a base"""
    from utils import criar_caminho
    
    arquivo_consolidado = criar_caminho('C:/TCC/dados', tipo, f'dados_{tipo}_consolidados.csv')
    arquivo_parquet = criar_caminho('C:/TCC/dados', tipo, f'dados_{tipo}_consolidados.parquet')
    
    # Verificar se existe arquivo consolidado (CSV ou Parquet)
    if os.path.exists(arquivo_consolidado):
        tamanho_mb = os.path.getsize(arquivo_consolidado) / (1024*1024)
        if tamanho_mb > 1:  # Arquivo deve ter pelo menos 1MB
            return arquivo_consolidado, tamanho_mb
    
    if os.path.exists(arquivo_parquet):
        tamanho_mb = os.path.getsize(arquivo_parquet) / (1024*1024)
        if tamanho_mb > 1:  # Arquivo deve ter pelo menos 1MB
            return arquivo_parquet, tamanho_mb
    
    return None, 0

def verificar_configuracoes():
    """Verifica configurações e espaço, além de identificar duplicatas nos arquivos .cfg"""
    configs = ['colunas_pnad.cfg', 'colunas_rais.cfg', 'colunas_caged.cfg']
    config_ok = True
    
    for config_file in configs:
        if not os.path.exists(config_file):
            logger.error(f"Arquivo de configuração não encontrado: {config_file}")
            config_ok = False
            continue

        # Ler o arquivo .cfg com configparser
        try:
            parser = configparser.ConfigParser()
            parser.read(config_file, encoding='utf-8')
            
            # Verificar duplicatas dentro de cada seção
            for section in parser.sections():
                colunas = parser.options(section)
                vistos = set()
                duplicados = set()
                for col in colunas:
                    if col in vistos:
                        duplicados.add(col)
                    vistos.add(col)
                
                if duplicados:
                    logger.error(f"ERRO: Nomes de colunas duplicados encontrados na seção '{section}' em '{config_file}': {list(duplicados)}")
                    print(f"❌ ERRO: Nomes de colunas duplicados encontrados na seção '{section}' em '{config_file}': {list(duplicados)}")
                    config_ok = False
        except Exception as e:
            logger.error(f"Erro ao ler o arquivo de configuração '{config_file}': {e}")
            print(f"❌ Erro ao ler o arquivo de configuração '{config_file}': {e}")
            config_ok = False

    if not config_ok:
        return False  # Retorna imediatamente se houver erro de configuração

    # Verificar espaço
    tem_espaco, espaco_gb = verificar_espaco_disco()
    print(f"\n💾 Espaço disponível: {espaco_gb:.1f} GB")
    
    if espaco_gb < 5:
        print("⚠️  AVISO: Pouco espaço em disco!")
        print("   Considere limpar arquivos temporários ou descompactados.")
    
    return True

def ajustar_colunas_ocupacao(df):
    """
    Renomeia a primeira coluna que corresponda a variações de 'CBO' para 'cbo_ocupacao'.
    Normaliza acentos, remove caracteres estranhos e aplica .strip().
    """
    for col in list(df.columns):
        col_norm = unicodedata.normalize('NFKD', col).encode('ASCII', 'ignore').decode('ASCII').lower().strip()
        # remover caracteres não alfanuméricos exceto underscore e espaço
        col_norm = re.sub(r'[^a-z0-9_ ]', '', col_norm)
        if col_norm.startswith('cbo') or re.match(r'^cbo[_\s]', col_norm):
            if 'cbo_ocupacao' not in df.columns:
                df.rename(columns={col: 'cbo_ocupacao'}, inplace=True)
            break
    return df

def menu_principal():
    print("\n" + "="*60)
    print("ANÁLISE DE ETARISMO EM TI NO BRASIL - v2.0")
    print("TCC - MBA Data Science USP/Esalq")
    print("="*60)
    print("\nMENU PRINCIPAL")
    print("1. Descompactação de Arquivos")
    print("2. Processar Bases")
    print("3. Relatórios e Análises")
    print("4. Gerar diagnóstico dos dados")  # <-- Nova opção
    print("0. Sair")
    return input("\nEscolha uma opção: ").strip()


def submenu_processar_bases():
    print("\n--- Processar Bases ---")
    print("1. Processar PNAD")
    print("2. Processar RAIS")
    print("3. Processar CAGED")
    print("4. Processar TODAS")
    print("0. Voltar")
    return input("Escolha uma opção: ").strip()


def submenu_relatorios():
    print("\n--- Relatórios e Análises ---")
    print("1. Relatório PNAD")
    print("2. Relatório RAIS")
    print("3. Relatório CAGED")
    print("4. Relatório CONSOLIDADO (TODAS)")
    print("0. Voltar")
    return input("Escolha uma opção: ").strip()


def submenu_descompactacao():
    print("\n--- Descompactação de Arquivos ---")
    print("1. Descompactar PNAD")
    print("2. Descompactar RAIS")
    print("3. Descompactar CAGED")
    print("4. Descompactar TODAS")
    print("5. Listar arquivos com erro de descompactação")  # <-- Adicionada esta linha
    print("0. Voltar")
    return input("Escolha uma opção: ").strip()


def encontrar_partes_consolidado(arquivo_base):
    """Encontra todas as partes de um arquivo consolidado"""
    partes = []
    
    # Verificar arquivo base
    if os.path.exists(arquivo_base):
        partes.append(arquivo_base)
    
    # Procurar por partes parquet
    dir_base = os.path.dirname(arquivo_base)
    nome_base = os.path.basename(arquivo_base).replace('.csv', '')
    
    # Procurar parquet correspondente
    arquivo_parquet = arquivo_base.replace('.csv', '.parquet')
    if os.path.exists(arquivo_parquet):
        partes = [arquivo_parquet]  # Preferir parquet
        
    # Procurar por múltiplas partes
    for ext in ['.parquet', '.csv', '.csv.gz']:
        pattern = os.path.join(dir_base, f"{nome_base}_parte_*{ext}")
        partes.extend(glob.glob(pattern))
    
    return sorted(list(set(partes)))  # Remover duplicatas


def consolidar_base(tipo):
    """Consolida dados preprocessados de uma base de forma incremental, sem filtrar TI, para todos os tipos"""
    from utils import criar_caminho
    import pyarrow.parquet as pq
    import gc

    dir_preprocessados = criar_caminho('C:/TCC/dados', tipo, 'preprocessados')
    arquivo_saida = criar_caminho('C:/TCC/dados', tipo, f'dados_{tipo}_consolidados.csv')
    arquivo_cfg = f'colunas_{tipo}.cfg'  # Adicionado para carregar colunas

    if tipo not in ['pnad', 'rais', 'caged']:
        print(f"Tipo de base desconhecido: {tipo}")
        return

    arquivos = glob.glob(os.path.join(dir_preprocessados, '*.csv')) + \
               glob.glob(os.path.join(dir_preprocessados, '*.parquet'))

    if not arquivos:
        print(f"Nenhum arquivo preprocessado encontrado para {tipo.upper()} em {dir_preprocessados}")
        return

    print(f"\n[{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}] INICIANDO CONSOLIDAÇÃO DE {tipo.upper()}...")
    print(f"{tipo.upper()}: {len(arquivos)} arquivos encontrados para consolidação:")

    for arq in arquivos:
        print(f" - {os.path.basename(arq)}")

    # Carregar nomes de colunas do CFG como garantia
    colunas_cfg = None
    if os.path.exists(arquivo_cfg):
        with open(arquivo_cfg, encoding='utf-8') as f:
            colunas_cfg = [line.strip() for line in f if line.strip()]
            print(f"   ✅ Nomes de colunas carregados de {arquivo_cfg}")

    # Remover arquivo antigo, se existir
    if os.path.exists(arquivo_saida):
        os.remove(arquivo_saida)

    total = len(arquivos)
    colunas_padrao = colunas_cfg  # Usar colunas do CFG como padrão
    total_registros = 0
    primeiro = True
    chunk_size = 500_000

    print("\nIniciando consolidação incremental...")
    for i, arq in enumerate(arquivos, 1):
        try:
            if arq.endswith('.parquet'):
                # Parquet já tem colunas, a leitura é direta
                chunk = pd.read_parquet(arq, engine='pyarrow')
                # --- AJUSTE DE COLUNA DE OCUPAÇÃO ---
                chunk = ajustar_colunas_ocupacao(chunk)
                if colunas_padrao is None:
                    colunas_padrao = list(chunk.columns)
                # Alinhar colunas e garantir que colunas essenciais estejam presentes
                    colunas_essenciais = ['idade', 'cbo_ocupacao']
                    for col in colunas_padrao + colunas_essenciais:
                        if col not in chunk.columns:
                            chunk[col] = None  # Adicionar colunas ausentes
                chunk = chunk[colunas_padrao]
                # Criar a coluna 'eh_ti' com base em critérios (exemplo: ocupações relacionadas à TI)
                if 'eh_ti' not in chunk.columns:
                    chunk['eh_ti'] = chunk['cbo_ocupacao'].apply(lambda x: x in ['1234', '5678'] if pd.notnull(x) else False)

                chunk.to_csv(arquivo_saida, sep=';', index=False, header=primeiro, mode='a', encoding='utf-8')
                primeiro = False
                total_registros += len(chunk)
                del chunk
                gc.collect()
            else: # Para arquivos CSV
                # Forçar o uso das colunas do CFG
                for chunk in ler_arquivo_com_fallback(arq, sep=';', chunk_size=chunk_size, colunas=colunas_padrao):
                    # --- AJUSTE DE COLUNA DE OCUPAÇÃO ---
                    chunk = ajustar_colunas_ocupacao(chunk)

                    # Verificar e corrigir colunas duplicadas
                    if chunk.columns.duplicated().any():
                        logger.warning(f"Colunas duplicadas detectadas em {arq}. Corrigindo...")
                        chunk = corrigir_colunas_duplicadas(chunk)

                    # Adicionar colunas ausentes de uma vez
                    colunas_ausentes = [col for col in colunas_padrao if col not in chunk.columns]
                    if colunas_ausentes:
                        chunk = pd.concat([chunk, pd.DataFrame(columns=colunas_ausentes)], axis=1)

                    # Reordenar as colunas
                    chunk = chunk[colunas_padrao]

                    # Alinhar colunas e garantir que colunas essenciais estejam presentes
                    colunas_essenciais = ['idade', 'cbo_ocupacao']
                    for col in colunas_padrao + colunas_essenciais:
                        if col not in chunk.columns:
                            chunk[col] = None  # Adicionar colunas ausentes

                    # Criar a coluna 'eh_ti' com base em critérios (exemplo: ocupações relacionadas à TI)
                    if 'eh_ti' not in chunk.columns:
                        chunk['eh_ti'] = chunk['cbo_ocupacao'].apply(lambda x: x in ['1234', '5678'] if pd.notnull(x) else False)

                    # Salvar o chunk no arquivo consolidado
                    chunk.to_csv(arquivo_saida, sep=';', index=False, header=primeiro, mode='a', encoding='utf-8')
                    primeiro = False
                    total_registros += len(chunk)
                    del chunk
                    gc.collect()
            porcentagem = (i / total) * 100
            print(f"[{i}/{total}] {os.path.basename(arq)} - {total_registros:,} registros acumulados ({porcentagem:.1f}% concluído)")
        except Exception as e:
            print(f"Erro ao processar {arq}: {e}")
    print(f"\n✅ Consolidação incremental concluída!")
    print(f"   Total de registros: {total_registros:,}")
    print(f"   Arquivo salvo: {arquivo_saida}")


def contar_linhas_fallback_encoding(arquivo, encodings=['utf-8', 'latin1', 'utf-8']):
    for enc in encodings:
        try:
            with open(arquivo, 'r', encoding=enc) as f:
                total = sum(1 for _ in f) - 1
                print(f"[INFO] Contagem de linhas em {arquivo} com encoding: {enc}")
                return total
        except Exception as e:
            print(f"[WARN] Falha ao contar linhas em {arquivo} com encoding {enc}: {e}")
    print(f"[ERRO] Não foi possível contar linhas em {arquivo} com nenhum encoding conhecido.")
    return None


# Função para carregar colunas de um arquivo .cfg respeitando as seções
def carregar_colunas_cfg(arquivo_cfg):
    """Carrega as colunas de um arquivo .cfg, organizadas por seção."""
    parser = configparser.ConfigParser()
    parser.read(arquivo_cfg, encoding='utf-8')
    
    colunas_por_secao = {}
    for section in parser.sections():
        colunas_por_secao[section] = parser.options(section)
    return colunas_por_secao

# Ajuste na função processar_base_completa
def processar_base_completa(nome_base, ProcessadorClasse, arquivo_cfg):
    """Processa uma base de dados completa"""
    from utils import criar_caminho
    
    print(f"\n{'='*60}")
    print(f"PROCESSANDO {nome_base}")
    print(f"{'='*60}")
    
    # PRIMEIRA VERIFICAÇÃO: Arquivo consolidado já existe?
    arquivo_existente, tamanho_mb = verificar_arquivo_consolidado_existe(nome_base.lower())
    
    if arquivo_existente:
        print(f"\n✅ {nome_base} já possui arquivo consolidado:")
        print(f"   • {os.path.basename(arquivo_existente)} ({tamanho_mb:.1f} MB)")
        
        # Perguntar se quer reprocessar
        resposta = input(f"\nDeseja reprocessar {nome_base} do zero? (S/N): ").strip().upper()
        if resposta != 'S':
            print(f"📊 Executando análise com dados existentes...")
            
            # Ir direto para análise
            dir_resultado = criar_caminho('resultados', nome_base.lower())
            os.makedirs(dir_resultado, exist_ok=True)
            
            analisador = AnalisadorEtarismo(dir_resultado)
            analisador.executar_analise_completa(arquivo_existente)
            
            print(f"✅ Análise de {nome_base} concluída!")
            return True
        else:
            print(f"🔄 Reprocessando {nome_base} do zero...")
            # Remover arquivo consolidado existente para forçar reprocessamento
            try:
                if os.path.exists(arquivo_existente):
                    os.remove(arquivo_existente)
                    print(f"   Arquivo consolidado removido: {os.path.basename(arquivo_existente)}")
            except Exception as e:
                print(f"   ⚠️ Erro ao remover arquivo existente: {e}")
    
    # SEGUNDA VERIFICAÇÃO: Verificar se há partes consolidadas (sistema antigo)
    arquivo_consolidado = criar_caminho('dados', nome_base.lower(), f'dados_{nome_base.lower()}_consolidados.csv')
    partes = encontrar_partes_consolidado(arquivo_consolidado)
    
    if partes and not arquivo_existente:  # Se não foi detectado na primeira verificação
        print(f"\n✅ {nome_base} possui dados em partes:")
        for parte in partes:
            tamanho_mb = os.path.getsize(parte) / (1024*1024)
            print(f"   • {os.path.basename(parte)} ({tamanho_mb:.1f} MB)")
            
        resposta = input(f"\nDeseja reprocessar {nome_base}? (S/N): ").strip().upper()
        if resposta != 'S':
            print(f"📊 Executando análise com partes existentes...")
            
            dir_resultado = criar_caminho('resultados', nome_base.lower())
            os.makedirs(dir_resultado, exist_ok=True)
            
            for idx, parte in enumerate(partes, 1):
                if os.path.exists(parte):
                    print(f"\n➡️  Processando arquivo {idx}/{len(partes)}: {os.path.basename(parte)}")
                    analisador = AnalisadorEtarismo(dir_resultado)
                    analisador.executar_analise_completa(parte)
                    
            return True
        else:
            # Limpar partes existentes
            for parte in partes:
                try:
                    if os.path.exists(parte):
                        os.remove(parte)
                        print(f"   Parte removida: {os.path.basename(parte)}")
                except Exception as e:
                    print(f"   ⚠️ Erro ao remover {parte}: {e}")
    
    # TERCEIRA PARTE: Processamento completo (descompactação + preprocessamento + consolidação)
    print(f"\n🔄 Iniciando processamento completo de {nome_base}...")
    
    # Verificar espaço antes
    tem_espaco, espaco_gb = verificar_espaco_disco()
    print(f"   Espaço disponível: {espaco_gb:.1f} GB")
    
    if espaco_gb < 5:
        print("   ⚠️ Limpando temporários...")
        limpar_temporarios(criar_caminho('dados', nome_base.lower(), 'temp'))
    
    print(f"   🔄 Reprocessamento automático de falhas ativado para {nome_base}")
    
    try:
        # Carregar configuração
        config = config_manager.ConfigManager().carregar_configuracao(nome_base.lower(), arquivo_cfg)
        
        # Forçar processamento sequencial se usar_paralelo=False
        if not config.usar_paralelo:
            print(f"📝 Processamento sequencial ativado (usar_paralelo=False)")
            config.usar_paralelo = False
            config.max_workers = 1
        
        # Criar processador
        processador = ProcessadorClasse(config)
        
        # Adicionar o caminho do CFG ao processador para que ele possa usá-lo
        processador.arquivo_cfg = arquivo_cfg

        # Processar com segurança
        with ProcessadorSeguro(processador) as proc:
            df = proc.processar_periodo_completo()
        
        if df is not None:
            print(f"✅ {nome_base} processada com sucesso!")
            print(f"   Total de registros: {len(df):,}")
            
            # Usar o caminho onde o processador realmente salvou
            arquivo_consolidado_real = criar_caminho('C:/TCC/dados', nome_base.lower(), f'dados_{nome_base.lower()}_consolidados.csv')

            # Salvar análises em resultados
            dir_resultado = criar_caminho('resultados', nome_base.lower())
            os.makedirs(dir_resultado, exist_ok=True)

            # Usar o arquivo real para análise
            arquivo_saida = arquivo_consolidado_real
            
            print(f"📊 Executando análise de etarismo...")
            analisador = AnalisadorEtarismo(dir_resultado)
            analisador.executar_analise_completa(arquivo_saida)
            
            return True
        else:
            print(f"❌ Falha no processamento de {nome_base}")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        logger.error(f"Erro ao processar {nome_base}: {e}")
        return False

def verificar_arquivos_descompactados(tipo):
    from utils import criar_caminho
    dir_descompactados = criar_caminho('dados', tipo, 'descompactados')
    if not os.path.exists(dir_descompactados):
        return False
    arquivos = [f for f in os.listdir(dir_descompactados) if f.endswith('.txt')]
    return len(arquivos) > 0

def verificar_arquivos_preprocessados(tipo):
    from utils import criar_caminho
    dir_preprocessados = criar_caminho('dados', tipo, 'preprocessados')
    if not os.path.exists(dir_preprocessados):
        return False
    arquivos = glob.glob(os.path.join(dir_preprocessados, '*.csv')) + \
               glob.glob(os.path.join(dir_preprocessados, '*.parquet'))
    return len(arquivos) > 0

def descompactar_base(tipo, ProcessadorClasse, arquivo_cfg):
    print(f"\nDescompactando {tipo.upper()}...")
    try:
        config = config_manager.ConfigManager().carregar_configuracao(tipo, arquivo_cfg)
        
        # Forçar descompactação sequencial se usar_paralelo=False
        if not config.usar_paralelo:
            print(f"📝 Descompactação sequencial ativada (usar_paralelo=False)")
            config.usar_paralelo = False
            config.max_workers = 1
        
        processador = ProcessadorClasse(config)
        descompactador = Descompactador(tipo, config.caminho_destino)
        arquivos = processador.descobrir_arquivos_compactados()
        print(f"Encontrados {len(arquivos)} arquivos para descompactar")
        sucesso = 0
        
        # Descompactação sequencial
        for i, arquivo in enumerate(arquivos, 1):
            print(f"📁 Descompactando arquivo {i}/{len(arquivos)}: {os.path.basename(arquivo)}")
            if descompactador.descompactar_arquivo(arquivo):
                sucesso += 1
                
        print(f"✅ {tipo.upper()}: {sucesso}/{len(arquivos)} arquivos descompactados")
    except Exception as e:
        print(f"❌ Erro ao descompactar {tipo.upper()}: {e}")
        logger.error(f"Erro ao descompactar {tipo.upper()}: {e}")

def processar_base_se_preciso(tipo, ProcessadorClasse, arquivo_cfg):
    """Processa base apenas se necessário, verificando primeiro se já existe consolidado"""
    
    # Verificar se já existe arquivo consolidado
    arquivo_existente, tamanho_mb = verificar_arquivo_consolidado_existe(tipo)
    
    if arquivo_existente:
        print(f"\n✅ {tipo.upper()} já possui arquivo consolidado ({tamanho_mb:.1f} MB)")
        print(f"   Pulando descompactação e preprocessamento...")
        
        # Ir direto para análise
        from utils import criar_caminho
        dir_resultado = criar_caminho('resultados', tipo)
        os.makedirs(dir_resultado, exist_ok=True)
        
        print(f"📊 Executando análise de etarismo para {tipo.upper()}...")
        analisador = AnalisadorEtarismo(dir_resultado)
        analisador.executar_analise_completa(arquivo_existente)
        
        return True
    
    # Se não existe consolidado, verificar se precisa descompactar
    if not verificar_arquivos_descompactados(tipo):
        print(f"⚠️ Arquivos descompactados de {tipo.upper()} não encontrados. Iniciando descompactação...")
        descompactar_base(tipo, ProcessadorClasse, arquivo_cfg)
    
    # Processar base completa
    return processar_base_completa(tipo.upper(), ProcessadorClasse, arquivo_cfg)

def relatorio_base_se_preciso(tipo, ProcessadorClasse, arquivo_cfg, func_relatorio):
    if not verificar_arquivos_preprocessados(tipo):
        print(f"⚠️ Arquivos preprocessados de {tipo.upper()} não encontrados. Iniciando processamento de base...")
        processar_base_se_preciso(tipo, ProcessadorClasse, arquivo_cfg)
    func_relatorio()

def salvar_csv_com_extensao_correta(df, caminho_arquivo, compressao=False, **kwargs):
    """Salva DataFrame como CSV, garantindo extensão .csv.gz se compressão gzip for usada."""
    if compressao:
        if not caminho_arquivo.endswith('.csv.gz'):
            caminho_arquivo = caminho_arquivo.replace('.csv', '') + '.csv.gz'
        df.to_csv(caminho_arquivo, sep=';', index=False, encoding='utf-8', compression='gzip', **kwargs)
    else:
        if not caminho_arquivo.endswith('.csv'):
            caminho_arquivo += '.csv'
        df.to_csv(caminho_arquivo, sep=';', index=False, encoding='utf-8', **kwargs)
    return caminho_arquivo


def processar_base_completa(nome_base, ProcessadorClasse, arquivo_cfg):
    """Processa uma base de dados completa"""
    from utils import criar_caminho
    
    print(f"\n{'='*60}")
    print(f"PROCESSANDO {nome_base}")
    print(f"{'='*60}")
    
    # PRIMEIRA VERIFICAÇÃO: Arquivo consolidado já existe?
    arquivo_existente, tamanho_mb = verificar_arquivo_consolidado_existe(nome_base.lower())
    
    if arquivo_existente:
        print(f"\n✅ {nome_base} já possui arquivo consolidado:")
        print(f"   • {os.path.basename(arquivo_existente)} ({tamanho_mb:.1f} MB)")
        
        # Perguntar se quer reprocessar
        resposta = input(f"\nDeseja reprocessar {nome_base} do zero? (S/N): ").strip().upper()
        if resposta != 'S':
            print(f"📊 Executando análise com dados existentes...")
            
            # Ir direto para análise
            dir_resultado = criar_caminho('resultados', nome_base.lower())
            os.makedirs(dir_resultado, exist_ok=True)
            
            analisador = AnalisadorEtarismo(dir_resultado)
            analisador.executar_analise_completa(arquivo_existente)
            
            print(f"✅ Análise de {nome_base} concluída!")
            return True
        else:
            print(f"🔄 Reprocessando {nome_base} do zero...")
            # Remover arquivo consolidado existente para forçar reprocessamento
            try:
                if os.path.exists(arquivo_existente):
                    os.remove(arquivo_existente)
                    print(f"   Arquivo consolidado removido: {os.path.basename(arquivo_existente)}")
            except Exception as e:
                print(f"   ⚠️ Erro ao remover arquivo existente: {e}")
    
    # SEGUNDA VERIFICAÇÃO: Verificar se há partes consolidadas (sistema antigo)
    arquivo_consolidado = criar_caminho('dados', nome_base.lower(), f'dados_{nome_base.lower()}_consolidados.csv')
    partes = encontrar_partes_consolidado(arquivo_consolidado)
    
    if partes and not arquivo_existente:  # Se não foi detectado na primeira verificação
        print(f"\n✅ {nome_base} possui dados em partes:")
        for parte in partes:
            tamanho_mb = os.path.getsize(parte) / (1024*1024)
            print(f"   • {os.path.basename(parte)} ({tamanho_mb:.1f} MB)")
            
        resposta = input(f"\nDeseja reprocessar {nome_base}? (S/N): ").strip().upper()
        if resposta != 'S':
            print(f"📊 Executando análise com partes existentes...")
            
            dir_resultado = criar_caminho('resultados', nome_base.lower())
            os.makedirs(dir_resultado, exist_ok=True)
            
            for idx, parte in enumerate(partes, 1):
                if os.path.exists(parte):
                    print(f"\n➡️  Processando arquivo {idx}/{len(partes)}: {os.path.basename(parte)}")
                    analisador = AnalisadorEtarismo(dir_resultado)
                    analisador.executar_analise_completa(parte)
                    
            return True
        else:
            # Limpar partes existentes
            for parte in partes:
                try:
                    if os.path.exists(parte):
                        os.remove(parte)
                        print(f"   Parte removida: {os.path.basename(parte)}")
                except Exception as e:
                    print(f"   ⚠️ Erro ao remover {parte}: {e}")
    
    # TERCEIRA PARTE: Processamento completo (descompactação + preprocessamento + consolidação)
    print(f"\n🔄 Iniciando processamento completo de {nome_base}...")
    
    # Verificar espaço antes
    tem_espaco, espaco_gb = verificar_espaco_disco()
    print(f"   Espaço disponível: {espaco_gb:.1f} GB")
    
    if espaco_gb < 5:
        print("   ⚠️ Limpando temporários...")
        limpar_temporarios(criar_caminho('dados', nome_base.lower(), 'temp'))
    
    print(f"   🔄 Reprocessamento automático de falhas ativado para {nome_base}")
    
    try:
        # Carregar configuração
        config = config_manager.ConfigManager().carregar_configuracao(nome_base.lower(), arquivo_cfg)
        
        # Forçar processamento sequencial se usar_paralelo=False
        if not config.usar_paralelo:
            print(f"📝 Processamento sequencial ativado (usar_paralelo=False)")
            config.usar_paralelo = False
            config.max_workers = 1
        
        # Criar processador
        processador = ProcessadorClasse(config)
        
        # Adicionar o caminho do CFG ao processador para que ele possa usá-lo
        processador.arquivo_cfg = arquivo_cfg

        # Processar com segurança
        with ProcessadorSeguro(processador) as proc:
            df = proc.processar_periodo_completo()
        
        if df is not None:
            print(f"✅ {nome_base} processada com sucesso!")
            print(f"   Total de registros: {len(df):,}")
            
            # Usar o caminho onde o processador realmente salvou
            arquivo_consolidado_real = criar_caminho('C:/TCC/dados', nome_base.lower(), f'dados_{nome_base.lower()}_consolidados.csv')

            # Salvar análises em resultados
            dir_resultado = criar_caminho('resultados', nome_base.lower())
            os.makedirs(dir_resultado, exist_ok=True)

            # Usar o arquivo real para análise
            arquivo_saida = arquivo_consolidado_real
            
            print(f"📊 Executando análise de etarismo...")
            analisador = AnalisadorEtarismo(dir_resultado)
            analisador.executar_analise_completa(arquivo_saida)
            
            return True
        else:
            print(f"❌ Falha no processamento de {nome_base}")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        logger.error(f"Erro ao processar {nome_base}: {e}")
        return False


def limpar_temporarios(diretorio_base='C:/TCC/dados'):
    """Limpa arquivos temporários"""
    # Importação explícita para garantir disponibilidade
    from utils import criar_caminho
    
    diretorios_temp = [
        criar_caminho(diretorio_base, 'rais', 'temp'),
        criar_caminho(diretorio_base, 'caged', 'temp'), 
        criar_caminho(diretorio_base, 'pnad', 'temp'),
        'temp'
    ]
    
    for diretorio in diretorios_temp:
        if os.path.exists(diretorio):
            try:
                limpar_diretorio(diretorio, manter_diretorio=True)
                print(f"✅ Limpo: {diretorio}")
            except Exception as e:
                print(f"⚠️  Erro ao limpar {diretorio}: {e}")


def executar_analise_consolidada(bases: Optional[List[str]] = None):
    """Executa a análise consolidada apenas nas bases informadas (['pnad','rais','caged'])."""
    # Inicializa resultados para evitar NameError
    resultados: Dict[str, dict] = {}

    validas = {'pnad', 'rais', 'caged'}
    if bases is None:
        selecionadas = ['pnad', 'rais', 'caged']
    else:
        if isinstance(bases, str):
            bases = [bases]
        selecionadas = [b.lower() for b in bases if b.lower() in validas]

    logger.info(f"Bases selecionadas para análise: {selecionadas if selecionadas else 'nenhuma'}")
    if not selecionadas:
        logger.warning("Nenhuma base válida selecionada. Abortando análise consolidada.")
        return resultados

    bases_dados = {
        'pnad': {'dir': criar_caminho('C:/TCC/dados', 'pnad', 'preprocessados')},
        'rais': {'dir': criar_caminho('C:/TCC/dados', 'rais', 'preprocessados')},
        'caged': {'dir': criar_caminho('C:/TCC/dados', 'caged', 'preprocessados')},
    }

    def _ler_csv_em_chunks(caminho: str, cols_desejadas: List[str], chunksize=500_000):
        # Lê cabeçalho para montar usecols seguro
        try:
            header_cols = list(pd.read_csv(caminho, sep=';', nrows=0).columns)
            usecols = [c for c in cols_desejadas if c in header_cols]
        except Exception:
            usecols = None
        for chunk in pd.read_csv(caminho, sep=';', usecols=usecols, chunksize=chunksize, low_memory=False):
            yield chunk

    for base in selecionadas:
        dir_pre = bases_dados[base]['dir']
        logger.info(f"Iniciando análise consolidada da base: {base.upper()} (dir={dir_pre})")

        arquivos = sorted(
            glob.glob(os.path.join(dir_pre, '*.csv')) +
            glob.glob(os.path.join(dir_pre, '*.parquet'))
        )
        if not arquivos:
            logger.warning(f"Sem arquivos preprocessados para {base.upper()} em {dir_pre}")
            continue

        # Acumuladores da base
        total_registros = 0
        ti_count = 0
        outros_count = 0
        idade_soma_ti = 0.0
        idade_soma_outros = 0.0
        anos_set = set()

        colunas_essenciais = ['ano', 'idade', 'cbo_ocupacao', 'eh_ti']

        for i, arq in enumerate(arquivos, 1):
            print(f"   Arquivo {i}/{len(arquivos)}: {os.path.basename(arq)}")
            try:
                if arq.lower().endswith('.parquet'):
                    iterador = _iterar_parquet_em_batches(arq, colunas_essenciais, batch_size=200_000)
                else:
                    iterador = _ler_csv_em_chunks(arq, colunas_essenciais, chunksize=500_000)

                for chunk in iterador:
                    chunk = _processar_chunk_para_metricas(chunk, colunas_essenciais)

                    ti_mask = chunk['eh_ti'] == True
                    n_ti = int(ti_mask.sum())
                    n_total = len(chunk)
                    ti_count += n_ti
                    total_registros += n_total
                    outros_count += (n_total - n_ti)

                    if 'idade' in chunk.columns:
                        idade_soma_ti += pd.to_numeric(chunk.loc[ti_mask, 'idade'], errors='coerce').fillna(0).sum()
                        idade_soma_outros += pd.to_numeric(chunk.loc[~ti_mask, 'idade'], errors='coerce').fillna(0).sum()

                    if 'ano' in chunk.columns:
                        anos_set.update(pd.to_numeric(chunk['ano'], errors='coerce').dropna().unique())

                    del chunk
                    gc.collect()

            except Exception as e:
                logger.error(f"Erro ao processar arquivo {arq}: {e}")
                print(f"❌ Erro ao processar arquivo {arq}: {e}")

        idade_media_ti = (idade_soma_ti / ti_count) if ti_count > 0 else 0.0
        idade_media_outros = (idade_soma_outros / outros_count) if outros_count > 0 else 0.0

        resultados[base.upper()] = {
            'total_registros': total_registros,
            'ti_count': ti_count,
            'outros_count': outros_count,
            'idade_media_ti': idade_media_ti,
            'idade_media_outros': idade_media_outros,
            'anos': sorted(list(anos_set)),
            'dir_preprocessados': dir_pre,
        }

        # Salva relatório dessa base na pasta-mãe (ex.: C:/TCC/dados/pnad/)
        base_dir_mae = os.path.dirname(dir_pre)  # remove "/preprocessados"
        os.makedirs(base_dir_mae, exist_ok=True)
        rel_path = os.path.join(base_dir_mae, f"relatorio_consolidado_{base}.txt")
        with open(rel_path, 'w', encoding='utf-8') as f:
            anos_txt = f"{int(min(anos_set))}-{int(max(anos_set))}" if anos_set else "Não disponível"
            f.write("=== RELATÓRIO DE ANÁLISE CONSOLIDADA ===\n\n")
            f.write(f"Base: {base.upper()}\n")
            f.write(f"Total de registros processados: {total_registros:,}\n")
            f.write(f"Profissionais TI identificados: {ti_count:,}\n")
            f.write(f"Outros profissionais: {outros_count:,}\n")
            perc_ti = (ti_count / total_registros * 100) if total_registros > 0 else 0.0
            f.write(f"Percentual TI: {perc_ti:.2f}%\n")
            f.write(f"Idade média TI: {idade_media_ti:.1f}\n")
            f.write(f"Idade média Outros: {idade_media_outros:.1f}\n")
            f.write(f"Anos de cobertura: {anos_txt}\n")

        logger.info(f"✅ Análise consolidada da base {base.upper()} concluída!")
        print(f"Relatório salvo em: {rel_path}")

    # Resumo no console
    if resultados:
        print("\n=== RESUMO CONSOLIDADO ===")
        for base, res in resultados.items():
            anos = res['anos']
            periodo = f"{int(min(anos))}-{int(max(anos))}" if anos else "ND"
            print(f"- {base}: total={res['total_registros']:,} TI={res['ti_count']:,} Outros={res['outros_count']:,} "
                  f"Idade TI={res['idade_media_ti']:.1f} Idade Outros={res['idade_media_outros']:.1f} Período={periodo}")
    else:
        logger.error("Nenhum dado disponível para análise consolidada")

    return resultados

def main():
    configurar_log()
    if not verificar_configuracoes():
        print("\n❌ Verifique as configurações antes de continuar.")
        return
    while True:
        opcao = menu_principal()
        if opcao == '0':
            print("\nEncerrando o programa...")
            break
        elif opcao == '1':
            while True:
                sub = submenu_descompactacao()
                if sub == '0':
                    break
                elif sub == '1':
                    descompactar_base('pnad', ProcessadorPNAD, 'colunas_pnad.cfg')
                elif sub == '2':
                    descompactar_base('rais', ProcessadorRAIS, 'colunas_rais.cfg')
                elif sub == '3':
                    descompactar_base('caged', ProcessadorCAGED, 'colunas_caged.cfg')
                elif sub == '4':
                    descompactar_base('pnad', ProcessadorPNAD, 'colunas_pnad.cfg')
                    descompactar_base('rais', ProcessadorRAIS, 'colunas_rais.cfg')
                    descompactar_base('caged', ProcessadorCAGED, 'colunas_caged.cfg')
                else:
                    print("Opção inválida!")
        elif opcao == '2':
            while True:
                sub = submenu_processar_bases()
                if sub == '0':
                    break
                elif sub == '1':
                    processar_base_se_preciso('pnad', ProcessadorPNAD, 'colunas_pnad.cfg')
                elif sub == '2':
                    processar_base_se_preciso('rais', ProcessadorRAIS, 'colunas_rais.cfg')
                elif sub == '3':
                    processar_base_se_preciso('caged', ProcessadorCAGED, 'colunas_caged.cfg')
                elif sub == '4':
                    processar_base_se_preciso('pnad', ProcessadorPNAD, 'colunas_pnad.cfg')
                    processar_base_se_preciso('rais', ProcessadorRAIS, 'colunas_rais.cfg')
                    processar_base_se_preciso('caged', ProcessadorCAGED, 'colunas_caged.cfg')
                else:
                    print("Opção inválida!")
        elif opcao == '3':
            while True:
                sub = submenu_relatorios()
                if sub == '0':
                    break
                elif sub == '1':
                    relatorio_base_se_preciso('pnad', ProcessadorPNAD, 'colunas_pnad.cfg', lambda: executar_analise_consolidada(['pnad']))
                elif sub == '2':
                    relatorio_base_se_preciso('rais', ProcessadorRAIS, 'colunas_rais.cfg', lambda: executar_analise_consolidada(['rais']))
                elif sub == '3':
                    relatorio_base_se_preciso('caged', ProcessadorCAGED, 'colunas_caged.cfg', lambda: executar_analise_consolidada(['caged']))
                elif sub == '4':
                    relatorio_base_se_preciso('todos', ProcessadorPNAD, 'colunas_pnad.cfg', lambda: executar_analise_consolidada(['pnad','rais','caged']))
                else:
                    print("Opção inválida!")
        else:
            print("Opção inválida!")
        input("\nPressione ENTER para continuar...")


if __name__ == '__main__':
    main()


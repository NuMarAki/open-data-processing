# -*- coding: utf-8 -*-
"""Processador base refatorado com responsabilidades bem definidas"""

import os
import glob
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config_manager import ConfigBase
from processamento.descompactador import Descompactador
from scripts.utils import (
    logger, monitorar_recursos, limpar_memoria_forcado,
    criar_caminho, validar_caminho, preparar_dados_ti
)

# Importação explícita adicional para garantir disponibilidade
from scripts.utils import criar_caminho as _criar_caminho

def _criar_caminho(*partes):
    from scripts.utils import criar_caminho
    return criar_caminho(*partes)

class ProcessadorBase(ABC):
    """Processador base com responsabilidades bem definidas"""
    
    def __init__(self, config: ConfigBase):
        # Importação explícita para garantir disponibilidade
        from scripts.utils import criar_caminho, validar_caminho
        
        self.config = config
        self.tipo = config.tipo
        
        # Delegar descompactação
        self.descompactador = Descompactador(self.tipo, config.caminho_destino)
        
        # Diretórios
        self.dir_preprocessados = _criar_caminho(config.caminho_destino, 'preprocessados')
        validar_caminho(self.dir_preprocessados, criar_se_nao_existir=True)
        
        logger.info(f"Processador {self.tipo} inicializado")
    
    @abstractmethod
    def processar_arquivo(self, arquivo_txt: str) -> Optional[pd.DataFrame]:
        """Processa arquivo TXT específico do formato"""
        pass

    @abstractmethod
    def descobrir_arquivos(self) -> List[str]:
        """Descobre arquivos disponíveis para processamento"""
        pass

    def processar_periodo_completo(self) -> Optional[pd.DataFrame]:
        """Processa período completo com paralelização adaptativa"""
        logger.info(f"Processamento {self.tipo}: {self.config.ano_inicio}-{self.config.ano_fim}")
        
        # Descobrir arquivos
        arquivos = self.descobrir_arquivos()
        if not arquivos:
            logger.error(f"Nenhum arquivo {self.tipo} encontrado")
            return None
        
        logger.info(f"Encontrados {len(arquivos)} arquivos para processar")
        
        # Forçar processamento sequencial para RAIS (evitar problemas de memória)
        if self.tipo == 'rais':
            logger.info(f"📝 Forçando processamento sequencial para RAIS (evitar problemas de memória)")
            df_final = self._processar_sequencial(arquivos)
        else:
            # Processar com paralelização para outros tipos
            df_final = self._processar_paralelo(arquivos)
        
        return df_final

    def _processar_paralelo(self, arquivos: List[str]) -> Optional[pd.DataFrame]:
        """Processa arquivos em paralelo"""
        logger.info(f"🚀 Processamento paralelo de {len(arquivos)} arquivos")
        
        # Preprocessar arquivos
        arquivos_preprocessados = []
        max_workers = self.config.max_workers or min(4, len(arquivos))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._preprocessar_arquivo, arq): arq 
                for arq in arquivos
            }
            
            with tqdm(total=len(arquivos), desc=f"Processando {self.tipo}") as pbar:
                for future in as_completed(futures):
                    try:
                        resultado = future.result(timeout=300)
                        if resultado:
                            arquivos_preprocessados.append(resultado)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Erro no processamento: {e}")
                        pbar.update(1)
        
        # Carregar e consolidar
        return self._carregar_preprocessados(arquivos_preprocessados)

    def _processar_sequencial(self, arquivos: List[str]) -> Optional[pd.DataFrame]:
        """Processa arquivos sequencialmente"""
        logger.info(f"📝 Processamento sequencial de {len(arquivos)} arquivos")
        
        arquivos_preprocessados = []
        
        for i, arquivo in enumerate(arquivos, 1):
            logger.info(f"📁 Processando arquivo {i}/{len(arquivos)}: {os.path.basename(arquivo)}")
            
            # Fallback defensivo: se a instância não tiver o método (ambientes antigos),
            # redireciona para a função da classe base.
            if not hasattr(self, "_preprocessar_arquivo"):
                logger.warning("Método _preprocessar_arquivo não encontrado na instância; aplicando fallback da classe base.")
                resultado = ProcessadorBase._preprocessar_arquivo(self, arquivo)  # type: ignore
            else:
                resultado = self._preprocessar_arquivo(arquivo)
            if resultado:
                arquivos_preprocessados.append(resultado)
                logger.info(f"✅ Arquivo {i}/{len(arquivos)} processado com sucesso")
            else:
                logger.warning(f"⚠️ Arquivo {i}/{len(arquivos)} falhou no processamento")
            
            # Limpar memória a cada arquivo processado
            limpar_memoria_forcado()
        
        logger.info(f"📊 Total de arquivos processados com sucesso: {len(arquivos_preprocessados)}/{len(arquivos)}")
        
        return self._carregar_preprocessados(arquivos_preprocessados)

    def _preprocessar_arquivo(self, arquivo_compactado: str) -> Optional[str]:
        """Preprocessa arquivo individual"""
        # Importação explícita para garantir disponibilidade
        from scripts.utils import criar_caminho
        
        try:
            # Descompactar
            logger.info(f"🗜️ Descompactando: {os.path.basename(arquivo_compactado)}")
            arquivo_txt = self.descompactador.descompactar_arquivo(arquivo_compactado)
            if not arquivo_txt:
                logger.error(f"❌ Falha na descompactação: {os.path.basename(arquivo_compactado)}")
                return None
            
            # Gerar nome do preprocessado
            nome_base = os.path.basename(arquivo_compactado)
            nome_preprocessado = f"{self.tipo}_{nome_base.replace('.7z', '').replace('.zip', '')}_processado.csv"
            arquivo_preprocessado = _criar_caminho(self.dir_preprocessados, nome_preprocessado)
            
            # Verificar se já existe
            if os.path.exists(arquivo_preprocessado):
                logger.info(f"✅ Já preprocessado: {nome_preprocessado}")
                # Mesmo que o arquivo principal já exista, tentaremos gerar TI/Não TI se estiverem ausentes
                gerar_splits = True
            else:
                gerar_splits = True
            
            # Processar
            logger.info(f"🔄 Processando: {os.path.basename(arquivo_txt)}")
            df = self.processar_arquivo(arquivo_txt)
            
            if df is None or len(df) == 0:
                logger.warning(f"⚠️ Nenhum dado extraído de {arquivo_txt}")
                return None
            
            logger.info(f"📊 Dados extraídos: {len(df):,} registros")
            
            # Aplicar amostragem se configurado
            if self.config.amostra_registros > 0 and len(df) > self.config.amostra_registros:
                df = df.sample(n=self.config.amostra_registros, random_state=42)
                logger.info(f"📊 Amostra aplicada: {len(df):,} registros")
            else:
                logger.info(f"📊 Processamento completo: {len(df):,} registros (sem amostragem)")
            
            # Salvar (principal)
            tamanho_df = len(df)
            arquivo_final = arquivo_preprocessado

            if tamanho_df > 100000:  # DataFrames grandes
                logger.info("💾 Salvando com compressão para otimizar I/O...")
                
                # Usar parquet para melhor performance (se disponível)
                try:
                    arquivo_parquet = arquivo_preprocessado.replace('.csv', '.parquet')
                    df.to_parquet(
                        arquivo_parquet,
                        engine='pyarrow',
                        compression='snappy',  # Compressão rápida
                        index=False
                    )
                    arquivo_final = arquivo_parquet
                    logger.info(f"✅ Salvo em formato Parquet (mais eficiente)")
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao salvar Parquet: {e}, usando CSV comprimido...")
                    # Fallback para CSV comprimido
                    df.to_csv(
                        arquivo_preprocessado, 
                        sep=';',
                        index=False,
                        compression={'method': 'gzip', 'compresslevel': 1},
                        chunksize=50000,
                        encoding='utf-8'
                    )
                    arquivo_final = arquivo_preprocessado + '.gz'
                    os.rename(arquivo_preprocessado, arquivo_final)
            else:
                df.to_csv(arquivo_preprocessado, sep=';', index=False, encoding='utf-8')
                logger.info(f"✅ Salvo em CSV")

            try:
                file_size = os.path.getsize(arquivo_final) / (1024*1024)  # MB
                logger.info(f"💾 Salvo: {os.path.basename(arquivo_final)} ({tamanho_df:,} registros, {file_size:.1f} MB)")
            except Exception:
                pass

            # NOVO: gerar splits TI e Não TI por preprocessado (no mesmo diretório)
            if gerar_splits:
                try:
                    try:
                        from scripts.utils import preparar_dados_ti
                    except Exception:
                        preparar_dados_ti = None
                    if preparar_dados_ti is None:
                        logger.warning("preparar_dados_ti não disponível — não será possível gerar arquivos TI/Não TI.")
                    else:
                        df_cls = preparar_dados_ti(
                            df.copy(),
                            cbo_ti=getattr(self.config, 'cbo_ti', None),
                            skip_age_filter=True
                        )
                        if 'eh_ti' not in df_cls.columns:
                            logger.warning("Coluna 'eh_ti' ausente após preparar_dados_ti — pulando geração de TI/Não TI.")
                        else:
                            prefer_parquet = str(arquivo_final).endswith('.parquet')
                            prefixo = os.path.splitext(arquivo_preprocessado)[0]
                            caminho_ti_parquet = prefixo + '_ti.parquet'
                            caminho_nao_ti_parquet = prefixo + '_nao_ti.parquet'
                            caminho_ti_csv = prefixo + '_ti.csv'
                            caminho_nao_ti_csv = prefixo + '_nao_ti.csv'

                            df_ti = df_cls[df_cls['eh_ti'] == True]
                            df_nao = df_cls[df_cls['eh_ti'] == False]

                            if prefer_parquet:
                                try:
                                    df_ti.to_parquet(caminho_ti_parquet, engine='pyarrow', compression='snappy', index=False)
                                    df_nao.to_parquet(caminho_nao_ti_parquet, engine='pyarrow', compression='snappy', index=False)
                                    logger.info(f"[PREPROCESSADO SPLIT] Salvo TI: {os.path.basename(caminho_ti_parquet)} ({len(df_ti):,})")
                                    logger.info(f"[PREPROCESSADO SPLIT] Salvo Não TI: {os.path.basename(caminho_nao_ti_parquet)} ({len(df_nao):,})")
                                except Exception as e:
                                    logger.warning(f"Falha ao salvar splits em Parquet: {e} — salvando em CSV.")
                                    df_ti.to_csv(caminho_ti_csv, sep=';', index=False, encoding='utf-8')
                                    df_nao.to_csv(caminho_nao_ti_csv, sep=';', index=False, encoding='utf-8')
                                    logger.info(f"[PREPROCESSADO SPLIT] Salvo TI: {os.path.basename(caminho_ti_csv)} ({len(df_ti):,})")
                                    logger.info(f"[PREPROCESSADO SPLIT] Salvo Não TI: {os.path.basename(caminho_nao_ti_csv)} ({len(df_nao):,})")
                            else:
                                df_ti.to_csv(caminho_ti_csv, sep=';', index=False, encoding='utf-8')
                                df_nao.to_csv(caminho_nao_ti_csv, sep=';', index=False, encoding='utf-8')
                                logger.info(f"[PREPROCESSADO SPLIT] Salvo TI: {os.path.basename(caminho_ti_csv)} ({len(df_ti):,})")
                                logger.info(f"[PREPROCESSADO SPLIT] Salvo Não TI: {os.path.basename(caminho_nao_ti_csv)} ({len(df_nao):,})")
                except Exception as e:
                    logger.warning(f"Falha na geração de arquivos TI/Não TI para {os.path.basename(arquivo_preprocessado)}: {e}")

            # Limpar memória
            del df
            limpar_memoria_forcado()
            
            return arquivo_final  # Retornar o caminho do arquivo final (Parquet ou CSV)
            
        except Exception as e:
            logger.error(f"❌ Erro ao preprocessar {arquivo_compactado}: {e}")
            return None

    def _carregar_preprocessados(self, arquivos: List[str]) -> Optional[str]:
        """Carrega e consolida arquivos preprocessados de forma incremental (sem estourar a memória)"""
        from scripts.utils import criar_caminho
        import pyarrow
        import pyarrow.parquet as pq
        
        if not arquivos:
            logger.error("Nenhum arquivo preprocessado para carregar")
            return None
        
        logger.info(f"Consolidação incremental de {len(arquivos)} arquivos preprocessados...")
        
        # Definir caminho do arquivo consolidado
        arquivo_consolidado = _criar_caminho(
            self.config.caminho_destino,
            f'dados_{self.tipo}_consolidados.csv'
        )
        primeiro = True
        total_registros = 0
        colunas_padrao = None
        
        # Remover arquivo antigo, se existir
        if os.path.exists(arquivo_consolidado):
            os.remove(arquivo_consolidado)
        
        for arquivo in arquivos:
            try:
                if arquivo.endswith('.parquet'):
                    df = pd.read_parquet(arquivo, engine='pyarrow')
                elif arquivo.endswith('.csv.gz'):
                    df = pd.read_csv(arquivo, compression='gzip', low_memory=False, sep=';')
                else:
                    df = pd.read_csv(arquivo, low_memory=False, sep=';')
                
                if len(df) == 0:
                    continue
                
                if colunas_padrao is None:
                    colunas_padrao = list(df.columns)
                    # Garante que idade e cbo_ocupacao estejam presentes no CAGED
                    if self.tipo == 'caged':
                        for col in ['idade', 'cbo_ocupacao']:
                            if col not in colunas_padrao:
                                colunas_padrao.append(col)
                # Garantir as mesmas colunas
                for col in colunas_padrao:
                    if col not in df.columns:
                        df[col] = None
                df = df[colunas_padrao]
                
                df.to_csv(arquivo_consolidado, sep=';', index=False, encoding='utf-8', mode='a', header=primeiro)
                primeiro = False
                total_registros += len(df)
                logger.info(f"Consolidado: {os.path.basename(arquivo)} ({len(df):,} registros)")
                del df
            except Exception as e:
                logger.warning(f"Erro ao consolidar {arquivo}: {e}")
        
        logger.info(f"Consolidação incremental concluída: {total_registros:,} registros totais")
        logger.info(f"Dados consolidados salvos: {arquivo_consolidado}")
        return arquivo_consolidado if total_registros > 0 else None
    

class ProcessadorSeguro:
    """Context manager simplificado para processamento seguro"""
    
    def __init__(self, processador: ProcessadorBase):
        self.processador = processador
        self.recursos_iniciais = None
    
    def __enter__(self):
        logger.info("Iniciando processamento seguro...")
        self.recursos_iniciais = monitorar_recursos()
        
        if self.recursos_iniciais:
            logger.info(f"RAM inicial: {self.recursos_iniciais['memoria_percent']:.1f}%")
        
        # Limpar falhas do cache
        falhas = self.processador.descompactador.cache.forcar_reprocessamento_falhas()
        if falhas > 0:
            logger.info(f"🔄 {falhas} falhas serão reprocessadas")
        
        return self.processador
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("Finalizando processamento...")
        
        # Limpeza
        if hasattr(self.processador, 'limpar_recursos'):
            self.processador.limpar_recursos()
        
        limpar_memoria_forcado()
        
        # Log final
        recursos_finais = monitorar_recursos()
        if recursos_finais and self.recursos_iniciais:
            diff = recursos_finais['memoria_percent'] - self.recursos_iniciais['memoria_percent']
            logger.info(f"Diferença RAM: {diff:+.1f}%")
        
        if exc_type:
            logger.error(f"Erro durante processamento: {exc_val}")
        
        return False  # Re-raise exceptions

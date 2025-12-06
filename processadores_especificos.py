# -*- coding: utf-8 -*-
"""Processadores específicos simplificados para cada base de dados"""
PROCESSAMENTO_TI_APLICADO_FLAG = "_flag_ti_aplicado"
MIN_IDADE_PNAD = 14
MAX_IDADE_PNAD = 80

import hashlib
import os
import re
import glob
import pandas as pd
import gc
import unicodedata
from typing import Optional, List

from processador_base import ProcessadorBase
from config_manager import ConfigBase
from utils import logger, criar_caminho, listar_arquivos_por_extensao, preparar_dados_ti


def ajustar_colunas_ocupacao(df: pd.DataFrame):
    """Renomeia qualquer coluna que comece com 'cbo' ou 'cdo' para 'cbo_ocupacao'."""
    try:
        cols = list(df.columns)
        alvo = None
        for c in cols:
            cl = c.lower().strip()
            if cl.startswith('cbo') or cl.startswith('cdo'):
                alvo = c
                break
            # alguns RAIS trazem 'cbo ocupacao 2002' etc
            if 'cbo' in cl and 'ocup' in cl:
                alvo = c
                break
        if alvo and 'cbo_ocupacao' not in df.columns:
            df = df.rename(columns={alvo: 'cbo_ocupacao'})
        return df
    except Exception as e:
        logger.warning(f"Falha ao ajustar coluna de ocupação: {e}")
        return df

class ProcessadorPNAD(ProcessadorBase):
    """
    Processador específico para PNAD Contínua (layout posicional / FWF).
    Refatorações implementadas:
      - Uso de única função processar_arquivo (removidas duplicadas anteriores).
      - Filtro de idade (14–80) aplicado uma única vez antes da classificação TI.
      - Classificação TI aplicada só uma vez (flag de atributo evita reprocesso).
      - Removido salvamento Parquet (somente CSV utf-8-sig).
      - Garantida criação de cbo_familia (a partir de cbo_ocupacao) após renomeação.
      - Cálculo e log de proporções TI (geral simples, geral ponderada, ocupados simples, ocupados ponderada).
      - Log de hash do layout.
      - Código antigo para segunda aplicação de preparar_dados_ti e Parquet marcado como DESATIVADO.
    """

    def __init__(self, config: ConfigBase):
        super().__init__(config)
        self.nome_base = 'PNAD'
        logger.info("Processador PNAD inicializado (refatorado)")
        self.variaveis = getattr(config, 'variaveis', [])
        self.layout = getattr(config, 'layout', {})
        self.renomeacao = getattr(config, 'renomeacao', {})
        self.amostra_por_trimestre = getattr(config, 'amostra_por_trimestre', 0)
        self.cbo_ti = getattr(config, 'cbo_ti', None)

    # -----------------------------------------------------------------
    def descobrir_arquivos(self) -> List[str]:
        arquivos = []
        dir_descompactados = os.path.join(self.config.caminho_destino, 'descompactados')
        if not os.path.exists(dir_descompactados):
            logger.error(f"Diretório de descompactados não encontrado: {dir_descompactados}")
            return []
        for arquivo in os.listdir(dir_descompactados):
            if arquivo.endswith('.txt'):
                arquivos.append(os.path.join(dir_descompactados, arquivo))
        logger.info(f"{self.nome_base}: {len(arquivos)} arquivos descompactados encontrados")
        return arquivos

    # -----------------------------------------------------------------
    def _montar_colspecs(self):
        colspecs, nomes = [], []
        for var, posicoes in self.layout.items():
            try:
                if isinstance(posicoes, tuple):
                    inicio, fim = posicoes
                elif isinstance(posicoes, str):
                    inicio, fim = map(int, posicoes.split(','))
                else:
                    inicio, fim = posicoes
                # Layout IBGE é 1-based com fim inclusivo; pandas read_fwf aceita (start, end)
                colspecs.append((inicio - 1, fim))
                nomes.append(var)
            except Exception as e:
                logger.warning(f"Erro ao processar layout da variável {var}: {e}")
        return colspecs, nomes

    # -----------------------------------------------------------------
    def _processar_arquivo_grande(self, arquivo_txt: str, colspecs: List, nomes: List) -> Optional[pd.DataFrame]:
        logger.info("Usando processamento otimizado para arquivo grande (FWF em lote único)")
        try:
            df = pd.read_fwf(
                arquivo_txt,
                colspecs=colspecs,
                names=nomes,
                encoding='latin1',
                dtype=str,
                na_values=['', ' '],
                keep_default_na=False
            )
            if len(df) > 500_000:
                logger.info("PNAD: Processados 500,000 registros...")
            logger.info(f"Total processado: {len(df):,} registros")
            return df
        except Exception as e:
            logger.error(f"Erro no processamento em arquivo grande: {e}")
            return None

    def processar_arquivo(self, arquivo: str) -> Optional[pd.DataFrame]:
        try:
            if not arquivo.endswith('.txt'):
                logger.error(f"Arquivo não é um .txt descompactado: {arquivo}")
                return None

            logger.info(f"Lendo arquivo descompactado: {os.path.basename(arquivo)}")
            colspecs, nomes = self._montar_colspecs()
            if not colspecs:
                logger.error("Nenhuma especificação de coluna válida encontrada no layout")
                return None

            logger.info(f"Extraindo {len(colspecs)} colunas do layout: {nomes}")
            tamanho_mb = os.path.getsize(arquivo) / (1024 * 1024)
            if tamanho_mb > 500:
                df = self._processar_arquivo_grande(arquivo, colspecs, nomes)
            else:
                df = pd.read_fwf(
                    arquivo,
                    colspecs=colspecs,
                    names=nomes,
                    encoding='latin1',
                    dtype=str,
                    na_values=['', ' '],
                    keep_default_na=False
                )

            if df is None or df.empty:
                logger.error(f"Nenhum dado extraído do arquivo: {arquivo}")
                return None

            logger.info(f"[ANALISE] Dados brutos extraídos: {len(df):,} registros com {len(df.columns)} colunas")
            logger.info(f"[ANALISE] Colunas extraídas: {list(df.columns)}")

            if self.renomeacao:
                logger.info("🔧 Aplicando renomeações de colunas...")
                renomeacoes_aplicadas = {old: new for old, new in self.renomeacao.items() if old in df.columns}
                logger.info(f"Renomeações aplicadas: {renomeacoes_aplicadas}")
                df = df.rename(columns=renomeacoes_aplicadas)
                logger.info(f"Colunas após renomeação: {list(df.columns)}")

            logger.info("🔧 Convertendo tipos de dados...")
            campos_numericos = {
                'idade': 'int64', 'ano': 'int64', 'trimestre': 'int64', 'ano_nascimento': 'int64',
                'peso_populacional': 'float64',
                'rendimento_trabalho_principal': 'float64', 'rendimento_bruto_mensal': 'float64',
                'horas_trabalhadas_semana': 'int64', 'anos_estudo': 'int64'
            }
            for campo, tipo in campos_numericos.items():
                if campo in df.columns:
                    try:
                        if 'float' in tipo:
                            df[campo] = pd.to_numeric(df[campo], errors='coerce')
                        else:
                            df[campo] = pd.to_numeric(df[campo], errors='coerce').astype('Int64')
                        logger.info(f"✅ {campo} convertido para {tipo}")
                    except Exception as e:
                        logger.warning(f"⚠️ Erro ao converter {campo}: {e}")

            # Ajustar ocupação
            df = ajustar_colunas_ocupacao(df)



            # Filtro idade
            if 'idade' in df.columns:
                antes = len(df)
                df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
                df = df[(df['idade'] >= MIN_IDADE_PNAD) & (df['idade'] <= MAX_IDADE_PNAD)]
                logger.info(f"Filtro idade {MIN_IDADE_PNAD}-{MAX_IDADE_PNAD}: {antes:,} → {len(df):,} registros")

            # Obter lista CBO TI
            cbos_ti_cfg = getattr(self.config, 'cbo_ti', None)
            if not cbos_ti_cfg:
                logger.warning("Lista cbo_ti não encontrada no config — classificação pode gerar 0 TI.")
            else:
                logger.info(f"Lista CBO TI (config): {cbos_ti_cfg}")

            # Diagnóstico inicial de CBO antes da classificação
            if 'cbo_ocupacao' in df.columns:
                amostra_cbo = df['cbo_ocupacao'].dropna().astype(str).str.strip()
                # Extrair família (4 dígitos) mesmo antes do preparar_dados_ti
                df['cbo_familia'] = amostra_cbo.str.replace(r'[^0-9]', '', regex=True).str.zfill(4).str[:4]
                top_familias = df['cbo_familia'].value_counts().head(15)
                logger.info(f"Top 15 famílias CBO (pré-classificação): {dict(top_familias)}")
                if not cbos_ti_cfg:
                    # Sugestão: mostrar interseção potencial
                    familias_21xx = df['cbo_familia'].dropna().str.startswith(('14','21','31')).value_counts()
                    logger.info(f"Diagnóstico famílias iniciando em 14/21/31: {familias_21xx.to_dict()}")
            else:
                logger.warning("Coluna cbo_ocupacao não encontrada antes da classificação.")

            if PROCESSAMENTO_TI_APLICADO_FLAG not in df.attrs:
                try:
                    from utils import preparar_dados_ti, normalizar_cbo_familia
                except ImportError:
                    from utils_refatorado import preparar_dados_ti, normalizar_cbo_familia

                logger.info("🔧 Aplicando processamento TI (refatorado)...")
                df = preparar_dados_ti(
                    df,
                    cbo_ti=cbos_ti_cfg,
                    min_idade=MIN_IDADE_PNAD,
                    max_idade=MAX_IDADE_PNAD,
                    skip_age_filter=True
                )
                df.attrs[PROCESSAMENTO_TI_APLICADO_FLAG] = True

                if 'cbo_familia' not in df.columns and 'cbo_ocupacao' in df.columns:
                    try:
                        df['cbo_familia'] = normalizar_cbo_familia(df['cbo_ocupacao'])
                    except Exception:
                        pass
                if 'cbo_familia' in df.columns:
                    total = len(df)
                    missing = int(df['cbo_familia'].isna().sum())
                    pct_missing = missing / total * 100 if total else 0.0
                    logger.info(f"[CBO] cbo_familia ausente/invalidos: {missing:,} / {total:,} ({pct_missing:.2f}%)")
                    if pct_missing > 30.0:
                        logger.warning("[CBO] Alta proporção de cbo_familia ausente (>30%). Verifique layout/renomeacao/posicoes no cfg.")
            else:
                logger.info("🔧 Processamento TI já aplicado - pulando.")

            # Proporções gerais
            if 'eh_ti' in df.columns:
                prop_ti = df['eh_ti'].mean()
                logger.info(f"Proporção TI (geral simples): {prop_ti*100:.2f}% [TI={df['eh_ti'].sum():,}/Total={len(df):,}]")
                if 'peso_populacional' in df.columns:
                    try:
                        w = pd.to_numeric(df['peso_populacional'], errors='coerce')
                        mask = w.notna() & (w > 0)
                        if mask.any():
                            import numpy as np
                            prop_ti_w = float((df.loc[mask, 'eh_ti'] * w[mask]).sum() / w[mask].sum())
                            logger.info(f"Proporção TI (geral ponderada): {prop_ti_w*100:.2f}%")
                    except Exception as e:
                        logger.warning(f"Falha proporção TI ponderada: {e}")

            # Ocupados
            if all(c in df.columns for c in ['ocupado', 'eh_ti']):
                df_occ = df[df['ocupado'] == 1]
                if len(df_occ):
                    prop_occ = df_occ['eh_ti'].mean()
                    logger.info(f"Proporção TI (ocupados simples): {prop_occ*100:.2f}% "
                                f"[TI={df_occ['eh_ti'].sum():,}/Ocupados={len(df_occ):,}]")
                    if 'peso_populacional' in df_occ.columns:
                        try:
                            w_occ = pd.to_numeric(df_occ['peso_populacional'], errors='coerce')
                            mask_o = w_occ.notna() & (w_occ > 0)
                            if mask_o.any():
                                import numpy as np
                                prop_occ_w = float((df_occ.loc[mask_o, 'eh_ti'] * w_occ[mask_o]).sum() / w_occ[mask_o].sum())
                                logger.info(f"Proporção TI (ocupados ponderada): {prop_occ_w*100:.2f}%")
                        except Exception as e:
                            logger.warning(f"Falha proporção ocupados ponderada: {e}")

            # Hash layout
            try:
                hash_layout = hashlib.md5(str(sorted(self.layout.items())).encode()).hexdigest()
                logger.info(f"Hash layout PNAD: {hash_layout}")
            except Exception:
                pass

            for col in ['faixa_etaria', 'geracao']:
                if col in df.columns:
                    df = df.drop(columns=[col])
                    logger.info(f"Coluna {col} removida dos dados")

            dir_preprocessados = os.path.join(self.config.caminho_destino, 'preprocessados')
            os.makedirs(dir_preprocessados, exist_ok=True)
            nome_base = os.path.splitext(os.path.basename(arquivo))[0]
            arquivo_csv = os.path.join(dir_preprocessados, f"{nome_base}_preprocessado.csv")
            logger.info(f"Salvando {len(df):,} registros em: {arquivo_csv}")
            df.to_csv(
                arquivo_csv,
                sep=';',
                index=False,
                encoding='utf-8-sig',
                lineterminator='\n'   # <<< CORREÇÃO AQUI
            )
            logger.info(f"Arquivo CSV salvo: {arquivo_csv}")

            logger.info("✅ Processamento concluído:")
            logger.info(f"   • Total registros: {len(df):,}")
            logger.info(f"   • Total colunas: {len(df.columns)}")
            logger.info(f"   • Arquivo salvo em: preprocessados/")
            logger.info(f"   • Colunas finais: {list(df.columns)}")

            gc.collect()
            return df

        except Exception as e:
            logger.error(f"Erro ao processar {os.path.basename(arquivo)}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _extrair_ano(self, df: pd.DataFrame) -> pd.Series:
        """Extrai ano dos dados PNAD (fallback simples)."""
        if 'Ano' in df.columns:
            return pd.to_numeric(df['Ano'], errors='coerce')
        if 'ano' in df.columns:
            return pd.to_numeric(df['ano'], errors='coerce')
        return pd.Series([self.config.ano_inicio] * len(df))

class ProcessadorRAIS(ProcessadorBase):
    """Processador específico para RAIS"""
    
    def __init__(self, config: ConfigBase):
        super().__init__(config)
        self.nome_base = 'RAIS'
        logger.info("Processador rais inicializado")
        self.renomeacao = getattr(config, 'renomeacao', {})

    # Fallback defensivo: garante que _preprocessar_arquivo exista na instância
    # mesmo que o ambiente não exponha o método herdado (evita AttributeError do log).
    def _preprocessar_arquivo(self, arquivo_compactado: str) -> Optional[str]:
        try:
            return super()._preprocessar_arquivo(arquivo_compactado)
        except AttributeError:
            logger.error("Método base _preprocessar_arquivo não encontrado; verifique a versão de processador_base.py")
            return None
    
    def descobrir_arquivos(self) -> List[str]:
        """Descobre arquivos RAIS para processamento"""
        arquivos = []
        dir_descompactados = os.path.join(self.config.caminho_destino, 'descompactados')
        if not os.path.exists(dir_descompactados):
            logger.error(f"Diretório de descompactados não encontrado: {dir_descompactados}")
            return []
        for arquivo in os.listdir(dir_descompactados):
            if arquivo.endswith('.txt'):
                caminho_completo = os.path.join(dir_descompactados, arquivo)
                arquivos.append(caminho_completo)
        logger.info(f"{self.nome_base}: {len(arquivos)} arquivos descompactados encontrados")
        return arquivos
    
    def processar_arquivo(self, arquivo_txt: str) -> Optional[pd.DataFrame]:
        """Processa arquivo TXT específico do formato RAIS, com normalização de colunas."""
        try:
            logger.info(f"Lendo arquivo descompactado: {os.path.basename(arquivo_txt)}")
            encodings_tentar = ['utf-8', 'latin1', 'iso-8859-1']
            encoding_usado = None
            for enc in encodings_tentar:
                try:
                    with open(arquivo_txt, 'r', encoding=enc) as f:
                        _ = f.readline().strip()
                        encoding_usado = enc
                        break
                except UnicodeDecodeError:
                    continue
            if not encoding_usado:
                logger.error(f"Não foi possível determinar o encoding do arquivo {arquivo_txt}.")
                return None
            logger.info(f"Arquivo {arquivo_txt} lido com encoding: {encoding_usado}")

            df = pd.read_csv(
                arquivo_txt,
                sep=';',
                encoding=encoding_usado,
                dtype=str,
                low_memory=False,
                skip_blank_lines=True,
                skipinitialspace=True,
                on_bad_lines='skip'
            )
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.normalize('NFKD')
                .str.encode('ascii', errors='ignore')
                .str.decode('utf-8')
                .str.replace(r'\s+', '_', regex=True)
            )
            logger.info(f"Colunas normalizadas: {list(df.columns)}")

            df = ajustar_colunas_ocupacao(df)

            if self.renomeacao:
                renomeacao_valida = {k.lower(): v for k, v in self.renomeacao.items()}
                df = df.rename(columns=renomeacao_valida)

            colunas_essenciais = ['idade', 'cbo_ocupacao']
            colunas_faltantes = [col for col in colunas_essenciais if col not in df.columns]
            if colunas_faltantes:
                logger.error(f"Colunas essenciais ausentes: {colunas_faltantes}")
            else:
                logger.info(f"Colunas essenciais encontradas: {colunas_essenciais}")

            return df

        except Exception as e:
            logger.error(f"Erro ao processar {os.path.basename(arquivo_txt)}: {e}")
            return None

class ProcessadorCAGED(ProcessadorBase):
    """Processador específico para CAGED"""
    
    def __init__(self, config: ConfigBase):
        super().__init__(config)
        self.nome_base = 'CAGED'
        logger.info("Processador caged inicializado")
        self.tipos_processar = getattr(config, 'tipos_processar', {'MOV': True, 'FOR': True, 'EXC': False})
        self.renomeacao = getattr(config, 'renomeacao', {})
    
    def descobrir_arquivos(self) -> List[str]:
        """Descobre arquivos CAGED para processamento"""
        arquivos = []
        
        # Usar diretório de descompactados
        dir_descompactados = os.path.join(self.config.caminho_destino, 'descompactados')
        if not os.path.exists(dir_descompactados):
            logger.error(f"Diretório de descompactados não encontrado: {dir_descompactados}")
            return []
            
        # Procurar arquivos .txt descompactados
        for arquivo in os.listdir(dir_descompactados):
            if arquivo.endswith('.txt'):
                caminho_completo = os.path.join(dir_descompactados, arquivo)
                arquivos.append(caminho_completo)
        
        logger.info(f"{self.nome_base}: {len(arquivos)} arquivos descompactados encontrados")
        return arquivos
    
    def processar_arquivo(self, arquivo: str) -> Optional[pd.DataFrame]:
        """Processa um único arquivo CAGED"""
        try:
            # Verificar se é arquivo descompactado
            if not arquivo.endswith('.txt'):
                logger.error(f"Arquivo não é um .txt descompactado: {arquivo}")
                return None
                
            logger.info(f"Lendo arquivo descompactado: {os.path.basename(arquivo)}")
            
            # Ler arquivo com buffer otimizado
            df = pd.read_csv(
                arquivo,
                sep=self.config.delimitador,
                encoding=self.config.encoding,
                dtype=str,
                na_values=['', ' '],
                low_memory=True,
                engine='c',
                skip_blank_lines=True,
                skipinitialspace=True,
                on_bad_lines='skip'
            )
            
            # Padronizar nomes de colunas
            df.columns = df.columns.str.strip()
            
            # Aplicar renomeação
            if self.renomeacao:
                df = df.rename(columns=self.renomeacao)
            
            # Garantir que competencia seja string
            if 'competencia' in df.columns:
                df['competencia'] = df['competencia'].astype(str)
            
            # Converter competencia para ano se possível
            if 'competencia' in df.columns:
                df['ano'] = df['competencia'].str[:4].astype(int)
            
            # Aplicar filtros básicos
            if 'idade' in df.columns:
                antes = len(df)
                df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
                df = df[(df['idade'] >= MIN_IDADE_PNAD) & (df['idade'] <= MAX_IDADE_PNAD)]
                logger.info(f"Filtro idade {MIN_IDADE_PNAD}-{MAX_IDADE_PNAD}: {antes:,} → {len(df):,}")
            
            # Processar salário
            if 'salario' in df.columns:
                df['salario'] = pd.to_numeric(df['salario'].str.replace(',', '.'), errors='coerce')
            
            if PROCESSAMENTO_TI_APLICADO_FLAG not in df.attrs:
                logger.info("🔧 Aplicando processamento TI (primeira vez)...")
                from utils import preparar_dados_ti
                df = preparar_dados_ti(
                    df,
                    cbo_ti=cbos_ti_cfg,
                    min_idade=MIN_IDADE_PNAD,
                    max_idade=MAX_IDADE_PNAD  # aqui não filtra novamente se alterarmos preparar_dados_ti
                )
                df.attrs[PROCESSAMENTO_TI_APLICADO_FLAG] = True
            else:
                logger.info("🔧 Processamento TI já aplicado - pulando.")

            return df
            
        except Exception as e:
            logger.error(f"Erro ao processar {os.path.basename(arquivo)}: {e}")
            return None
    
    def _extrair_ano(self, df: pd.DataFrame) -> pd.Series:
        """Extrai ano dos dados CAGED"""
        if 'ano' in df.columns:
            return pd.to_numeric(df['ano'], errors='coerce')
        elif 'competencia' in df.columns:
            # Competência no formato AAAAMM
            return pd.to_numeric(df['competencia'].astype(str).str[:4], errors='coerce')
        else:
            return pd.Series([self.config.ano_inicio] * len(df))

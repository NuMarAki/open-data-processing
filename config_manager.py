# -*- coding: utf-8 -*-
"""Gerenciador centralizado de configurações"""

import configparser
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from utils import logger


@dataclass
class ConfigBase:
    """Configuração base para processadores"""
    tipo: str
    caminho_arquivos_compactados: str
    caminho_destino: str
    delimitador: str = ';'
    encoding: str = 'utf-8'
    batch_size: int = 3
    amostra_registros: int = 0
    ano_inicio: int = 2012
    ano_fim: int = 2025
    usar_paralelo: bool = True
    max_workers: Optional[int] = None
    # Campos específicos da PNAD
    variaveis: Optional[list] = None
    layout: Optional[dict] = None
    renomeacao: Optional[dict] = None
    amostra_por_trimestre: int = 0
    tipos_processar: Optional[dict] = None
    # Campos específicos da RAIS
    colunas_rais: Optional[dict] = None
    cbo_ti: Optional[List[str]] = None
    # Novos campos de processamento
    forcar_sequencial: bool = False
    usar_chunking: bool = False
    tamanho_chunk: int = 100000
    # Limites de memória
    limite_memoria_mb: int = 8192
    forcar_limpeza_memoria: bool = True
    percentual_max_memoria: int = 80
    # Caminhos adicionais
    caminho_resultados: Optional[str] = None
    rotulo_fonte: Optional[str] = None
    # Filtros de análise
    filtros_analise: Optional[Dict[str, Any]] = None
    faixas_etarias: Optional[Dict[str, tuple]] = None
    

class ConfigManager:
    """Gerenciador unificado de configurações"""
    
    _instance = None
    _configs = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def carregar_configuracao(self, tipo: str, arquivo_cfg: str) -> ConfigBase:
        """Carrega configuração de arquivo .cfg"""
        if tipo in self._configs:
            return self._configs[tipo]
        
        if not os.path.exists(arquivo_cfg):
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {arquivo_cfg}")
        
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(arquivo_cfg, encoding='utf-8')
        
        # Extrair configurações comuns
        config_dict = self._extrair_config_comum(config, tipo)
        
        # Configurações específicas por tipo
        if tipo == 'pnad':
            config_dict.update(self._config_pnad(config))
        elif tipo == 'rais':
            config_dict.update(self._config_rais(config))
        elif tipo == 'caged':
            config_dict.update(self._config_caged(config))
        
        config_obj = ConfigBase(**config_dict)
        self._configs[tipo] = config_obj
        
        logger.info(f"Configuração {tipo} carregada: {arquivo_cfg}")
        return config_obj
    
    def _extrair_config_comum(self, config: configparser.ConfigParser, tipo: str) -> Dict[str, Any]:
        """Extrai configurações comuns"""
        base_config = {
            'tipo': tipo,
            'caminho_arquivos_compactados': config.get('caminhos', 'caminho_arquivos_compactados', 
                                                       fallback=config.get('caminhos_processamento', 'caminho_arquivos_compactados', fallback='')),
            'caminho_destino': config.get('caminhos', 'caminho_descompactacao',
                                         fallback=config.get('caminhos_processamento', 'caminho_descompactacao', fallback=f'dados/{tipo}')),
        }
        
        # Caminhos adicionais
        if config.has_section('caminhos'):
            base_config['caminho_resultados'] = config.get('caminhos', 'caminho_resultados', fallback=None)
            base_config['rotulo_fonte'] = config.get('caminhos', 'rotulo_fonte', fallback=None)
        
        # Parâmetros de leitura
        if config.has_section('parametros_leitura'):
            base_config.update({
                'delimitador': config.get('parametros_leitura', 'delimitador', fallback=';'),
                'encoding': config.get('parametros_leitura', 'encoding', fallback='utf-8'),
                'usar_chunking': config.getboolean('parametros_leitura', 'usar_chunking', fallback=False),
                'tamanho_chunk': config.getint('parametros_leitura', 'tamanho_chunk', fallback=100000)
            })
        elif config.has_section('parametros_leitura_txt'):
            base_config.update({
                'delimitador': config.get('parametros_leitura_txt', 'delimitador', fallback=';'),
                'encoding': config.get('parametros_leitura_txt', 'encoding', fallback='utf-8')
            })
        
        # Parâmetros de processamento
        if config.has_section('parametros_processamento'):
            base_config.update({
                'usar_paralelo': config.getboolean('parametros_processamento', 'usar_paralelo', fallback=True),
                'max_workers': config.getint('parametros_processamento', 'max_workers', fallback=4) or None,
                'batch_size': config.getint('parametros_processamento', 'batch_size', fallback=3),
                'amostra_registros': config.getint('parametros_processamento', 'amostra_registros', fallback=0),
                'forcar_sequencial': config.getboolean('parametros_processamento', 'forcar_sequencial', fallback=False)
            })
        
        # Período de análise
        if config.has_section('periodo_analise'):
            base_config.update({
                'ano_inicio': config.getint('periodo_analise', 'ano_inicio', fallback=2012),
                'ano_fim': config.getint('periodo_analise', 'ano_fim', fallback=2025)
            })
        
        # Limites de memória
        if config.has_section('memoria'):
            base_config.update({
                'limite_memoria_mb': config.getint('memoria', 'limite_memoria_mb', fallback=8192),
                'forcar_limpeza_memoria': config.getboolean('memoria', 'forcar_limpeza_memoria', fallback=True),
                'percentual_max_memoria': config.getint('memoria', 'percentual_max_memoria', fallback=80)
            })
        
        # Faixas etárias
        if config.has_section('faixas_etarias'):
            faixas = {}
            for faixa, valores in config.items('faixas_etarias'):
                try:
                    min_idade, max_idade = [int(v.strip()) for v in valores.split(',')]
                    faixas[faixa] = (min_idade, max_idade)
                except:
                    logger.warning(f"Faixa etária inválida: {faixa} = {valores}")
            base_config['faixas_etarias'] = faixas
        
        # Filtros de análise
        if config.has_section('filtros_analise'):
            base_config['filtros_analise'] = dict(config.items('filtros_analise'))
        
        # Debug/Otimização (fallback para compatibilidade)
        if config.has_section('debug'):
            base_config.update({
                'amostra_registros': config.getint('debug', 'amostra_registros', fallback=base_config.get('amostra_registros', 0)),
                'max_workers': config.getint('debug', 'max_workers_otimizado', fallback=base_config.get('max_workers')),
                'usar_paralelo': config.getboolean('debug', 'usar_paralelo', fallback=base_config.get('usar_paralelo', True))
            })
        
        return base_config
    
    def _config_pnad(self, config: configparser.ConfigParser) -> Dict[str, Any]:
        """Configurações específicas PNAD"""
        pnad_config: Dict[str, Any] = {}

        # 1. Variáveis selecionadas (se usar subset em algum processamento específico)
        if config.has_section('variaveis'):
            variaveis = [v.strip() for v in config.get('variaveis', 'variaveis').split(',') if v.strip()]
            pnad_config['variaveis'] = variaveis

        # 2. Layout posicional (converter para dict {nome: (inicio,fim)})
        if config.has_section('layout'):
            layout_raw = dict(config.items('layout'))
            layout_proc = {}
            for k, v in layout_raw.items():
                # Aceita formatos: "5,10" ou "5-10" ou "5:10"
                v_norm = v.replace('-', ',').replace(':', ',')
                try:
                    inicio, fim = [int(x.strip()) for x in v_norm.split(',') if x.strip()]
                    if inicio > fim:
                        inicio, fim = fim, inicio
                    layout_proc[k] = (inicio, fim)
                except Exception:
                    logger.warning(f"[CONFIG][PNAD] Layout inválido para {k}: {v}")
            pnad_config['layout'] = layout_proc

        # 3. Renomeações (ex.: V2009 -> idade)
        if config.has_section('renomeacao'):
            pnad_config['renomeacao'] = dict(config.items('renomeacao'))

        # 4. Amostragem opcional por trimestre
        if config.has_option('periodo_analise', 'amostra_registros_por_trimestre'):
            pnad_config['amostra_por_trimestre'] = config.getint(
                'periodo_analise',
                'amostra_registros_por_trimestre'
            )

        # 5. CBO TI – lê seção [cbo_ti] e registra em ambas as chaves para compatibilidade
        if config.has_section('cbo_ti'):
            try:
                raw_codigos = config.get('cbo_ti', 'codigos', fallback='')
                cbos = [c.strip() for c in raw_codigos.split(',') if c.strip()]
                cbos_familia = []
                for c in cbos:
                    digits = ''.join(ch for ch in c if ch.isdigit())
                    if len(digits) >= 4:
                        cbos_familia.append(digits[:4])
                cbos_familia = sorted(set(cbos_familia))
                # garantir retrocompatibilidade: setar tanto 'cbo_ti' quanto 'cbo_ti'
                pnad_config['cbo_ti'] = cbos_familia
                pnad_config['cbo_ti'] = cbos_familia
                logger.info(f"[CONFIG][PNAD] cbo_ti carregados ({len(cbos_familia)}): {cbos_familia}")
            except Exception as e:
                logger.warning(f"[CONFIG][PNAD] Falha ao carregar cbo_ti: {e}")
        else:
            logger.warning("[CONFIG][PNAD] Seção [cbo_ti] ausente no cfg – classificação TI ficará vazia.")

        return pnad_config

    def _config_rais(self, config: configparser.ConfigParser) -> Dict[str, Any]:
        """Configurações específicas RAIS"""
        rais_config = {
            'encoding': 'latin1'  # RAIS geralmente usa latin1
        }
        
        # Carregar configurações de colunas
        if config.has_section('colunas_rais'):
            rais_config['colunas_rais'] = dict(config.items('colunas_rais'))
        
        # Carregar configurações de renomeação
        if config.has_section('renomeacao'):
            rais_config['renomeacao'] = dict(config.items('renomeacao'))
        
        return rais_config
    
    def _config_caged(self, config: configparser.ConfigParser) -> Dict[str, Any]:
        """Configurações específicas CAGED"""
        caged_config = {}
        # Carregar tipos de arquivo
        if config.has_section('tipos_arquivo_caged'):
            caged_config['tipos_processar'] = {
                'MOV': config.getboolean('tipos_arquivo_caged', 'processar_mov', fallback=True),
                'FOR': config.getboolean('tipos_arquivo_caged', 'processar_for', fallback=True),
                'EXC': config.getboolean('tipos_arquivo_caged', 'processar_exc', fallback=False)
            }
        # Carregar renomeação de colunas (unificar colunas_caged e renomeacao)
        renomeacao = {}
        if config.has_section('colunas_caged'):
            renomeacao.update(dict(config.items('colunas_caged')))
        if config.has_section('renomeacao'):
            renomeacao.update(dict(config.items('renomeacao')))
        if renomeacao:
            caged_config['renomeacao'] = renomeacao
        return caged_config
    
    def get_config(self, tipo: str) -> Optional[ConfigBase]:
        """Retorna configuração carregada"""
        return self._configs.get(tipo)
    
    def get_all_configs(self) -> Dict[str, ConfigBase]:
        """Retorna todas as configurações"""
        return self._configs.copy()


# Singleton global
config_manager = ConfigManager()
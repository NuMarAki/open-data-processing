# -*- coding: utf-8 -*-
"""Utilitários comuns refatorados e simplificados"""

import os
import json
import logging
import hashlib
import shutil
import gc
import psutil
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import shutil
import hashlib
import unicodedata
import re

# Configuração de logging
_logger_instance = None

def configurar_log(nome_arquivo='analise.log') -> logging.Logger:
    """Configura logging simplificado"""
    global _logger_instance
    
    # Se já existe uma instância, retorna ela
    if _logger_instance is not None:
        return _logger_instance
    
    # Criar logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Limpar handlers existentes para evitar duplicação
    logger.handlers = []
    
    # Formato das mensagens
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Handler para arquivo com UTF-8
    file_handler = logging.FileHandler(nome_arquivo, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para console com tratamento de caracteres especiais
    class ConsoleHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                # Substituir emojis por texto simples
                msg = msg.replace('🔄', '[PROCESSANDO]')
                msg = msg.replace('✅', '[OK]')
                msg = msg.replace('❌', '[ERRO]')
                msg = msg.replace('⚠️', '[ALERTA]')
                msg = msg.replace('📊', '[ANALISE]')
                msg = msg.replace('📦', '[ARQUIVO]')
                msg = msg.replace('💾', '[DISCO]')
                msg = msg.replace('💻', '[SISTEMA]')
                msg = msg.replace('🗜️', '[COMPACTACAO]')
                msg = msg.replace('📋', '[STATUS]')
                msg = msg.replace('🧹', '[LIMPEZA]')
                msg = msg.replace('🚀', '[PERFORMANCE]')
                msg = msg.replace('🆕', '[NOVO]')
                msg = msg.replace('💡', '[DICA]')
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)
    
    console_handler = ConsoleHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Salvar instância
    _logger_instance = logger
    return logger


logger = configurar_log()


# CBOs de TI padronizados
CBOS_TI = [
    '1425',  # Gerentes de TI
    '2120', '2122', '2123', '2124',  # Profissionais de TI
    '2141', '2142', '2143', '2149',  # Analistas e desenvolvedores
    '3171', '3172', '3173'  # Técnicos
]

# Faixas etárias padronizadas
FAIXAS_ETARIAS = {
    '18-24': (18, 24),
    '25-34': (25, 34),
    '35-44': (35, 44),
    '45-54': (45, 54),
    '55-64': (55, 64),
    '65+': (65, 100)
}

def salvar_arquivo_preprocessado(df, caminho_arquivo, formato='csv'):
    """Salva o DataFrame no formato especificado, garantindo a extensão correta."""
    if formato == 'csv':
        if not caminho_arquivo.endswith('.csv'):
            caminho_arquivo += '.csv'
        df.to_csv(caminho_arquivo, sep=';', index=False, encoding='utf-8')
    else:
        raise ValueError(f"Formato desconhecido: {formato}")
    print(f"✅ Arquivo salvo: {caminho_arquivo}")
    
def ler_arquivo_com_fallback(arquivo, sep=';', chunk_size=None, colunas=None):
    """Lê um arquivo com fallback de codificação e garante que os chunks sejam DataFrames."""
    encodings_tentar = ['utf-8', 'latin1', 'iso-8859-1']
    for enc in encodings_tentar:
        try:
            if chunk_size:
                # Leitura em chunks
                for chunk in pd.read_csv(
                    arquivo,
                    sep=sep,
                    encoding=enc,
                    chunksize=chunk_size,
                    low_memory=False,
                    names=colunas,
                    header=None
                ):
                    # Garantir que o chunk seja um DataFrame com colunas
                    if colunas:
                        chunk.columns = colunas
                    yield chunk
            else:
                # Leitura completa
                df = pd.read_csv(
                    arquivo,
                    sep=sep,
                    encoding=enc,
                    low_memory=False,
                    names=colunas,
                    header=None
                )
                if colunas:
                    df.columns = colunas
                return df
        except UnicodeDecodeError:
            logger.warning(f"Falha ao ler {arquivo} com encoding {enc}. Tentando próximo...")
    raise Exception(f"Não foi possível ler o arquivo {arquivo} com os encodings conhecidos.")


# Funções de sistema de arquivos
def normalizar_caminho(caminho: str) -> str:
    """Normaliza caminhos para o sistema operacional"""
    return os.path.normpath(caminho) if caminho else caminho


def criar_caminho(*partes: str) -> str:
    """Cria caminho de forma segura"""
    return normalizar_caminho(os.path.join(*partes))


def validar_caminho(caminho: str, criar_se_nao_existir: bool = False) -> bool:
    """Valida se caminho existe e opcionalmente cria"""
    if not caminho:
        return False
    
    caminho_normalizado = normalizar_caminho(caminho)
    
    if os.path.exists(caminho_normalizado):
        return True
    
    if criar_se_nao_existir:
        try:
            os.makedirs(caminho_normalizado, exist_ok=True)
            return True
        except Exception:
            return False
    
    return False


def obter_tamanho_arquivo_mb(arquivo: str) -> float:
    """Retorna tamanho do arquivo em MB"""
    try:
        return os.path.getsize(arquivo) / (1024 * 1024)
    except:
        return 0


def listar_arquivos_por_extensao(diretorio: str, extensoes: List[str]) -> List[str]:
    """Lista arquivos por extensões específicas"""
    arquivos = []
    if not os.path.exists(diretorio):
        return arquivos
    
    extensoes_lower = [ext.lower() for ext in extensoes]
    
    for root, _, files in os.walk(diretorio):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensoes_lower):
                arquivos.append(os.path.join(root, file))
    
    return arquivos


def calcular_hash_arquivo(arquivo: str) -> Optional[str]:
    """Calcula hash MD5 de arquivo"""
    try:
        hash_md5 = hashlib.md5()
        with open(arquivo, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None


# Funções de processamento de dados
def configurar_ambiente(diretorio='graficos'):
    """
    Cria o diretório especificado para salvar outputs, se não existir.
    Args:
        diretorio: Nome do diretório a criar (default: 'graficos')
    """
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)
    print(f"Diretório '{diretorio}' pronto.")

def ajustar_colunas_ocupacao(df):
    """
    Renomeia qualquer coluna que corresponda a 'cbo' ou 'cdo' para 'cbo_ocupacao'.
    Trata variações: 'cbo', 'cdo', 'cbo ocupacao', 'cbo_ocupacao_2002', etc.
    Versão consolidada e unificada.
    """
    if df is None or df.empty:
        return df
    
    try:
        cols = list(df.columns)
        alvo = None
        
        for col in cols:
            # Normalizar nome da coluna removendo acentos e caracteres especiais
            col_norm = unicodedata.normalize('NFKD', str(col))
            col_norm = col_norm.encode('ASCII', 'ignore').decode('ASCII')
            col_norm = col_norm.lower().strip()
            # Remover caracteres não alfanuméricos exceto underscore e espaço
            col_norm = re.sub(r'[^a-z0-9_ ]', '', col_norm)
            
            # Verificar se começa com 'cbo' ou 'cdo'
            if col_norm.startswith('cbo') or col_norm.startswith('cdo'):
                alvo = col
                break
            # Verificar se contém 'cbo' e 'ocup' (ex: 'cbo ocupacao 2002')
            if 'cbo' in col_norm and 'ocup' in col_norm:
                alvo = col
                break
        
        # Renomear se encontrou e 'cbo_ocupacao' ainda não existe
        if alvo and 'cbo_ocupacao' not in df.columns:
            df.rename(columns={alvo: 'cbo_ocupacao'}, inplace=True)
            logger.info(f"Coluna '{alvo}' renomeada para 'cbo_ocupacao'")
        
        return df
    except Exception as e:
        logger.warning(f"Falha ao ajustar coluna de ocupação: {e}")
        return df

def preparar_dados_ti(df, cbo_ti=None, min_idade=14, max_idade=80, skip_age_filter=False):
    """
    Versão refatorada que:
      - cria cbo_familia com normalização robusta acima
      - loga cobertura antes/depois e percentuais de miss
      - usa apenas lista cbos_ti do cfg para classificar
    """
    try:
        _df = df.copy()

        # Identificar coluna de CBO
        coluna_cbo = None
        for cand in ['cbo_ocupacao', 'cbo', 'CBO', 'ocupacao', 'v4010', 'V4010']:
            if cand in _df.columns:
                coluna_cbo = cand
                break

        if coluna_cbo:
            before_nonnull = _df[coluna_cbo].notna().sum()
            logger.info(f"[CBO] Coluna de origem para CBO: {coluna_cbo} (valores não nulos: {before_nonnull:,})")
            _df['cbo_familia'] = normalizar_cbo_familia(_df[coluna_cbo])
            after_nonnull = _df['cbo_familia'].notna().sum()
            logger.info(f"[CBO] cbo_familia criada: {after_nonnull:,} registros com família válida "
                        f"({after_nonnull/len(_df)*100:.3f}% do arquivo)")
        else:
            _df['cbo_familia'] = None
            logger.warning("[CBO] Nenhuma coluna de CBO encontrada para normalização.")

        # Filtrar idade apenas se necessário
        if not skip_age_filter and 'idade' in _df.columns:
            _df = filtrar_idade(_df, 'idade', min_idade, max_idade, logger=logger)

        # Classificar TI usando somente lista do cfg
        if cbo_ti is None:
            try:
                cbo_ti = CBOS_TI  # fallback global se existir
                logger.info("[CBO] Usando lista CBOS_TI fallback")
            except Exception:
                cbo_ti = []
                logger.warning("[CBO] Nenhuma lista CBO TI fornecida (cbos_ti vazia).")

        # Cobertura e log antes de classificar
        if cbo_ti:
            total_fams = _df['cbo_familia'].notna().sum()
            matched = _df['cbo_familia'].isin(cbo_ti).sum()
            logger.info(f"[CBO] Cobertura lista cfg: {matched:,} registros (entre {total_fams:,} com família válida) "
                        f"→ {matched/len(_df)*100:.6f}% do arquivo total")
        else:
            logger.warning("[CBO] Lista cbos_ti vazia — classificação resultará em 0 TI.")

        # Criar eh_ti booleano (apenas match com a lista)
        _df['eh_ti'] = _df['cbo_familia'].isin(cbo_ti) if cbo_ti else False

        # Normalizar rendimentos (remanejar como antes)
        for col in _df.columns:
            if 'rendimento' in col.lower() or 'renda' in col.lower():
                _df[col] = pd.to_numeric(_df[col], errors='coerce')

        # Estatísticas básicas logadas
        ti_count = int(_df['eh_ti'].sum()) if 'eh_ti' in _df.columns else 0
        logger.info(f"[RESUMO] Preparação TI: total={len(_df):,} TI={ti_count:,} ({(ti_count/len(_df))*100 if len(_df) else 0:.6f}%)")
        if 'peso_populacional' in _df.columns and 'eh_ti' in _df.columns:
            try:
                prop_w = weighted_proportion(_df['eh_ti'], _df['peso_populacional'])
                logger.info(f"[RESUMO] Proporção TI ponderada: {prop_w*100:.6f}%")
            except Exception:
                logger.warning("[RESUMO] Falha ao calcular proporção ponderada dentro de preparar_dados_ti.")

        return _df
    except Exception as e:
        logger.error(f"Erro em preparar_dados_ti (refatorado): {e}")
        return df
    
def classificar_faixa_etaria(idade: float) -> str:
    """Classifica idade em faixa etária"""
    for faixa, (min_age, max_age) in FAIXAS_ETARIAS.items():
        if min_age <= idade <= max_age:
            return faixa
    return 'Outros'


def gerar_estatisticas_basicas(df_ti: pd.DataFrame, 
                              df_outros: pd.DataFrame) -> Dict[str, Any]:
    """Gera estatísticas básicas comparativas"""
    stats = {
        'total_ti': len(df_ti),
        'total_outros': len(df_outros),
        'idade_media_ti': df_ti['idade'].mean() if len(df_ti) > 0 else 0,
        'idade_media_outros': df_outros['idade'].mean() if len(df_outros) > 0 else 0,
        'perc_ti': len(df_ti) / (len(df_ti) + len(df_outros)) * 100 
                   if (len(df_ti) + len(df_outros)) > 0 else 0
    }
    
    # Distribuição por faixas
    for faixa in FAIXAS_ETARIAS.keys():
        stats[f'ti_{faixa}'] = len(df_ti[df_ti['faixa_etaria'] == faixa])
        stats[f'outros_{faixa}'] = len(df_outros[df_outros['faixa_etaria'] == faixa])
    
    return stats


# Gerenciamento de recursos
def monitorar_recursos() -> Optional[Dict[str, Any]]:
    """Monitora uso de recursos do sistema"""
    try:
        mem = psutil.virtual_memory()
        
        return {
            'memoria_usada_gb': (mem.total - mem.available) / (1024**3),
            'memoria_total_gb': mem.total / (1024**3),
            'memoria_percent': mem.percent,
            'cpu_percent': psutil.cpu_percent(interval=1),
            'threads_ativas': len(psutil.Process().threads())
        }
    except Exception as e:
        logger.warning(f"Erro ao monitorar recursos: {e}")
        return None


def limpar_memoria_forcado():
    """Força limpeza de memória"""
    logger.info("Executando limpeza de memória...")
    
    # Garbage collection
    for i in range(3):
        n = gc.collect()
        logger.debug(f"GC round {i+1}: {n} objetos coletados")
    
    # Limpar caches do pandas se possível
    try:
        pd.core.computation.expressions.set_use_numexpr(False)
        pd.core.computation.expressions.set_use_numexpr(True)
    except:
        pass


def verificar_espaco_disco(caminho: str = '.') -> Tuple[bool, float]:
    """Verifica espaço disponível em disco"""
    try:
        total, usado, livre = shutil.disk_usage(caminho)
        livre_gb = livre / (1024**3)
        return True, livre_gb
    except Exception:
        return False, 0


def limpar_diretorio(caminho: str, manter_diretorio: bool = True) -> bool:
    """Limpa conteúdo de um diretório"""
    if not os.path.exists(caminho):
        return True
    
    try:
        for item in os.listdir(caminho):
            item_path = os.path.join(caminho, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        
        if not manter_diretorio:
            os.rmdir(caminho)
        
        return True
    except Exception as e:
        logger.error(f"Erro ao limpar diretório {caminho}: {e}")
        return False


# Classe CacheManager simplificada (mantida pois é essencial)
class CacheManager:
    """Gerenciador de cache simplificado"""
    
    def __init__(self, arquivo_cache: str):
        self.arquivo = arquivo_cache
        self.cache = self._carregar()
    
    def _carregar(self) -> Dict[str, Any]:
        """Carrega cache do disco"""
        try:
            if os.path.exists(self.arquivo):
                with open(self.arquivo, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {e}")
        return {}
    
    def salvar(self):
        """Salva cache no disco"""
        try:
            os.makedirs(os.path.dirname(self.arquivo), exist_ok=True)
            with open(self.arquivo, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {e}")
    
    def get(self, chave: str, default: Any = None) -> Any:
        """Obtém valor do cache"""
        return self.cache.get(chave, default)
    
    def set(self, chave: str, valor: Any):
        """Define valor no cache"""
        self.cache[chave] = valor
        self.salvar()
    
    def invalidar(self, chave: str):
        """Remove entrada do cache"""
        if chave in self.cache:
            del self.cache[chave]
            self.salvar()
    def gerar_chave_unica(self, arquivo_origem: str, tipo: str) -> str:
        """Gera chave única baseada no caminho completo"""
        # Normalizar caminho
        caminho_normalizado = os.path.normpath(arquivo_origem)
        
        # Extrair informações relevantes
        nome_arquivo = os.path.basename(caminho_normalizado)
        diretorio_pai = os.path.basename(os.path.dirname(caminho_normalizado))
        
        # Gerar hash curto do caminho completo para garantir unicidade
        hash_caminho = hashlib.md5(caminho_normalizado.encode('utf-8')).hexdigest()[:8]
        
        # Criar chave legível: nome_arquivo_diretorioPai_hashCurto_tipo
        chave = f"{os.path.splitext(nome_arquivo)[0]}_{diretorio_pai}_{hash_caminho}_{tipo}"
        
        return chave

    def listar_falhas_por_origem(self) -> Dict[str, List[str]]:
        """Lista falhas agrupadas por origem"""
        falhas = {}
        
        for chave, valor in self.cache.items():
            if isinstance(valor, dict) and valor.get('status') == 'failed':
                origem = valor.get('arquivo_origem', 'Origem desconhecida')
                if origem not in falhas:
                    falhas[origem] = []
                falhas[origem].append({
                    'chave': chave,
                    'erro': valor.get('erro', ''),
                    'tentativas': valor.get('tentativas', 0),
                    'timestamp': valor.get('timestamp', 0)
                })
        
        return falhas
    
    def set_failed(self, chave: str, erro: str, arquivo_origem: str = None):
        """Registra falha no cache"""
        self.cache[chave] = {
            'status': 'failed',
            'erro': str(erro),
            'arquivo_origem': arquivo_origem,
            'timestamp': datetime.now().timestamp(),
            'tentativas': self.cache.get(chave, {}).get('tentativas', 0) + 1
        }
        self.salvar()

    def set_success(self, chave: str, arquivo_origem: str, arquivo_resultado: str):
        """Registra sucesso no cache"""
        self.cache[chave] = {
            'status': 'success',
            'hash': calcular_hash_arquivo(arquivo_origem),
            'arquivo_origem': os.path.abspath(arquivo_origem),
            'arquivo_txt': os.path.abspath(arquivo_resultado),
            'timestamp': datetime.now().timestamp(),
            'tamanho_mb': obter_tamanho_arquivo_mb(arquivo_resultado)
        }
        self.salvar()

    def should_skip(self, chave: str, arquivo_origem: str, 
               arquivo_esperado: Optional[str] = None) -> Tuple[bool, str]:
        """Verifica se deve pular processamento"""
        entrada = self.get(chave)
        
        if not entrada:
            return False, "Sem entrada no cache"
        
        # Verificar se é o mesmo arquivo origem
        origem_cache = entrada.get('arquivo_origem')
        if origem_cache and os.path.abspath(origem_cache) != os.path.abspath(arquivo_origem):
            return False, f"Arquivo origem diferente: {origem_cache} vs {arquivo_origem}"
        
        if entrada.get('status') == 'failed':
            # MUDANÇA: Sempre tentar reprocessar falhas
            tentativas = entrada.get('tentativas', 0)
            logger.info(f"Reprocessando falha anterior: {tentativas} tentativas - {origem_cache}")
            return False, f"Falha anterior ({tentativas} tentativas) - tentando novamente"
        
        if entrada.get('status') == 'success':
            arquivo_cache = entrada.get('arquivo_txt')
            if arquivo_cache and os.path.exists(arquivo_cache):
                return True, f"Arquivo já processado e válido - Origem: {origem_cache}"
        
        return False, "Deve processar"
    
    def forcar_reprocessamento_falhas(self) -> int:
        """Remove falhas do cache para forçar reprocessamento"""
        chaves_falhas = [
            k for k, v in self.cache.items() 
            if isinstance(v, dict) and v.get('status') == 'failed'
        ]
        
        for chave in chaves_falhas:
            del self.cache[chave]
        
        if chaves_falhas:
            self.salvar()
            logger.info(f"🔄 {len(chaves_falhas)} falhas removidas do cache")
        
        return len(chaves_falhas)
    
    def limpar_cache_antigo(self, dias: int = 30):
        """Remove entradas antigas do cache"""
        limite = datetime.now().timestamp() - (dias * 24 * 3600)
        chaves_remover = []
        
        for chave, valor in self.cache.items():
            if isinstance(valor, dict) and valor.get('timestamp', 0) < limite:
                chaves_remover.append(chave)
        
        for chave in chaves_remover:
            del self.cache[chave]
        
        if chaves_remover:
            self.salvar()
            logger.info(f"Cache limpo: {len(chaves_remover)} entradas antigas removidas")

def arquivo_existe_e_tamanho_ok(caminho, tamanho_esperado=None):
    """Verifica se o arquivo existe e, se informado, se o tamanho bate."""
    if not os.path.exists(caminho):
        return False
    if tamanho_esperado is not None:
        try:
            return os.path.getsize(caminho) == tamanho_esperado
        except Exception:
            return False
    return True

def limpar_temp_personalizado():
    temp_dir = os.environ.get('TEMP', None)
    if temp_dir and os.path.exists(temp_dir):
        for nome in os.listdir(temp_dir):
            if nome.startswith('temp_rais_') or nome.startswith('temp_pnad_'):
                caminho = os.path.join(temp_dir, nome)
                try:
                    shutil.rmtree(caminho)
                except Exception as e:
                    print(f'Erro ao remover {caminho}: {e}')

def listar_erros_descompactacao():
    """Lista erros de descompactação com informações detalhadas de origem"""
    import glob
    import json
    import os
    
    print("\n📋 ARQUIVOS COM ERRO DE DESCOMPACTAÇÃO (DETALHADO)")
    print("=" * 80)
    
    arquivos_cache = glob.glob(os.path.join('.cache', 'cache_*.json'))
    total_erros = 0
    
    for arquivo_cache in arquivos_cache:
        base = os.path.basename(arquivo_cache).replace('cache_', '').replace('.json', '').upper()
        
        try:
            with open(arquivo_cache, 'r', encoding='utf-8') as f:
                dados = json.load(f)
        except Exception as e:
            print(f'Erro ao ler cache {base}: {e}')
            continue
        
        # Agrupar erros por origem
        erros_por_origem = {}
        for chave, valor in dados.items():
            if isinstance(valor, dict) and valor.get('status') == 'failed':
                origem = valor.get('arquivo_origem', 'Origem desconhecida')
                if origem not in erros_por_origem:
                    erros_por_origem[origem] = []
                erros_por_origem[origem].append({
                    'chave': chave,
                    'erro': valor.get('erro', ''),
                    'tentativas': valor.get('tentativas', 0)
                })
        
        if erros_por_origem:
            print(f'\n{base}:')
            for origem, erros in erros_por_origem.items():
                print(f'  📁 Origem: {origem}')
                for erro_info in erros:
                    print(f'    ❌ {erro_info["erro"]} (tentativas: {erro_info["tentativas"]})')
                total_erros += len(erros)
    
    if total_erros == 0:
        print("Nenhum erro encontrado nos caches.")
    else:
        print(f"\n📊 Total de arquivos com erro: {total_erros}")
        print("\n💡 Para reprocessar falhas, use a opção de limpeza de cache ou delete manualmente as entradas com falha.")

def diagnosticar_conflitos_cache():
    """Diagnostica e oferece correção de conflitos de cache entre bases"""
    import glob
    import json
    
    print("\n🔍 DIAGNÓSTICO DE CONFLITOS DE CACHE")
    print("=" * 60)
    
    conflitos_encontrados = {}
    cache_legacy_detectado = False
    
    for arquivo_cache in glob.glob('.cache/cache_*.json'):
        base = os.path.basename(arquivo_cache).replace('cache_', '').replace('.json', '').upper()
        
        try:
            with open(arquivo_cache, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            # Detectar estrutura legacy (sem arquivo_origem)
            legacy_count = 0
            new_count = 0
            
            # Agrupar por nome base do arquivo
            nomes_base = {}
            for chave, valor in dados.items():
                if isinstance(valor, dict):
                    origem = valor.get('arquivo_origem', '')
                    
                    if not origem:
                        legacy_count += 1
                        cache_legacy_detectado = True
                    else:
                        new_count += 1
                    
                    nome_arquivo = os.path.basename(origem) if origem else chave.split('_')[0]
                    
                    if nome_arquivo not in nomes_base:
                        nomes_base[nome_arquivo] = []
                    nomes_base[nome_arquivo].append({
                        'chave': chave,
                        'origem': origem if origem else 'LEGACY (sem origem)',
                        'status': valor.get('status', 'unknown'),
                        'is_legacy': not bool(origem)
                    })
            
            print(f"\n{base}:")
            print(f"  📊 Total de entradas: {len(dados)}")
            print(f"  🆕 Estrutura nova: {new_count}")
            if legacy_count > 0:
                print(f"  🗂️  Estrutura legacy: {legacy_count}")
            
            # Detectar conflitos potenciais
            conflitos_base = {}
            for nome, entradas in nomes_base.items():
                if len(entradas) > 1:
                    conflitos_base[nome] = entradas
            
            if conflitos_base:
                conflitos_encontrados[base] = conflitos_base
                print(f"  ⚠️  Conflitos detectados: {len(conflitos_base)}")
                
                for nome_arquivo, entradas in conflitos_base.items():
                    print(f"    📁 {nome_arquivo} ({len(entradas)} entradas):")
                    for entrada in entradas:
                        status_icon = "✅" if entrada['status'] == 'success' else "❌" if entrada['status'] == 'failed' else "⚠️"
                        legacy_tag = " [LEGACY]" if entrada['is_legacy'] else ""
                        print(f"      {status_icon} {entrada['origem']}{legacy_tag}")
            else:
                print(f"  ✅ Nenhum conflito detectado")
        
        except Exception as e:
            print(f"  ❌ Erro ao analisar cache: {e}")
    
    # Oferecer soluções
    if cache_legacy_detectado or conflitos_encontrados:
        print(f"\n💡 SOLUÇÕES RECOMENDADAS:")
        
        if cache_legacy_detectado:
            print("  🗂️  Cache legacy detectado:")
            print("     - Entradas antigas sem rastreamento de origem")
            print("     - Recomenda-se limpar e reprocessar")
        
        if conflitos_encontrados:
            print("  ⚠️  Conflitos de cache detectados:")
            print("     - Múltiplas entradas para o mesmo nome de arquivo")
            print("     - Pode causar comportamento inconsistente")
        
        print(f"\n🔧 AÇÕES DISPONÍVEIS:")
        print("  1. Limpar entradas legacy (recomendado)")
        print("  2. Limpar apenas entradas com falha")
        print("  3. Limpar todo o cache (reprocessamento completo)")
        print("  4. Voltar ao menu principal")
        
        escolha = input("\nEscolha uma ação (1-4): ").strip()
        
        if escolha == '1':
            _limpar_cache_legacy()
        elif escolha == '2':
            _limpar_cache_falhas()
        elif escolha == '3':
            _limpar_cache_completo()
        elif escolha == '4':
            return
        else:
            print("❌ Opção inválida!")
    else:
        print("\n✅ CACHE SAUDÁVEL")
        print("  - Nenhum conflito detectado")
        print("  - Estrutura atualizada")
        print("  - Sistema funcionando corretamente")

def _limpar_cache_legacy():
    """Remove entradas legacy do cache"""
    print("\n🗂️ Limpando entradas legacy...")
    
    for arquivo_cache in glob.glob('.cache/cache_*.json'):
        try:
            with open(arquivo_cache, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            # Remover entradas sem arquivo_origem
            dados_limpos = {
                k: v for k, v in dados.items() 
                if isinstance(v, dict) and v.get('arquivo_origem')
            }
            
            removidas = len(dados) - len(dados_limpos)
            
            if removidas > 0:
                with open(arquivo_cache, 'w', encoding='utf-8') as f:
                    json.dump(dados_limpos, f, indent=2, ensure_ascii=False)
                
                base = os.path.basename(arquivo_cache).replace('cache_', '').replace('.json', '').upper()
                print(f"  ✅ {base}: {removidas} entradas legacy removidas")
        
        except Exception as e:
            print(f"  ❌ Erro ao limpar {arquivo_cache}: {e}")
    
    print("✅ Limpeza de cache legacy concluída!")

def _limpar_cache_falhas():
    """Remove apenas entradas com falha do cache"""
    print("\n❌ Limpando entradas com falha...")
    
    total_removidas = 0
    
    for arquivo_cache in glob.glob('.cache/cache_*.json'):
        try:
            with open(arquivo_cache, 'r', encoding='utf-8') as f:
                dados = json.load(f)
            
            # Remover entradas com status failed
            dados_limpos = {
                k: v for k, v in dados.items() 
                if not (isinstance(v, dict) and v.get('status') == 'failed')
            }
            
            removidas = len(dados) - len(dados_limpos)
            total_removidas += removidas
            
            if removidas > 0:
                with open(arquivo_cache, 'w', encoding='utf-8') as f:
                    json.dump(dados_limpos, f, indent=2, ensure_ascii=False)
                
                base = os.path.basename(arquivo_cache).replace('cache_', '').replace('.json', '').upper()
                print(f"  ✅ {base}: {removidas} falhas removidas")
        
        except Exception as e:
            print(f"  ❌ Erro ao limpar {arquivo_cache}: {e}")
    
    print(f"✅ {total_removidas} entradas com falha removidas!")

def _limpar_cache_completo():
    """Remove todo o cache"""
    print("\n🧹 ATENÇÃO: Isso removerá TODO o cache!")
    confirmacao = input("Tem certeza? Digite 'CONFIRMAR' para continuar: ")
    
    if confirmacao.upper() == 'CONFIRMAR':
        try:
            import shutil
            if os.path.exists('.cache'):
                shutil.rmtree('.cache')
                print("✅ Cache completamente removido!")
                print("   Próximo processamento recriará cache limpo")
            else:
                print("ℹ️  Diretório de cache não existe")
        except Exception as e:
            print(f"❌ Erro ao remover cache: {e}")
    else:
        print("❌ Operação cancelada")

def validar_colunas_essenciais(df, colunas_essenciais):
    """Valida se as colunas essenciais estão presentes no DataFrame"""
    try:
        # PRIMEIRO: Ajustar colunas de ocupação
        df = ajustar_colunas_ocupacao(df)
        
        # Verificar colunas ausentes
        colunas_ausentes = [col for col in colunas_essenciais if col not in df.columns]
        
        if colunas_ausentes:
            logger.error(f"Colunas não encontradas:")
            logger.error(f"  Procurando: {', '.join(colunas_essenciais)}")
            logger.error(f"  Disponíveis: {list(df.columns)}")
            
            # Para colunas essenciais ausentes, criar com valores None
            for col in colunas_ausentes:
                df[col] = None
                logger.warning(f"Coluna '{col}' criada com valores None")
        
        return df, colunas_ausentes
        
    except Exception as e:
        logger.error(f"Erro na validação de colunas: {e}")
        return df, colunas_essenciais
    
def filtrar_idade(df, col='idade', min_idade=14, max_idade=80, logger=None):
    if col not in df.columns:
        if logger: logger.warning(f"Coluna {col} não encontrada para filtro de idade.")
        return df
    antes = len(df)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[(df[col] >= min_idade) & (df[col] <= max_idade)]
    if logger:
        logger.info(f"Filtro idade {min_idade}-{max_idade}: {antes:,} → {len(df):,} (removidos {antes-len(df):,})")
    return df


def normalizar_cbo_familia(series):
    """
    Normaliza uma série contendo CBOs para a 'família' de 4 dígitos.
    Regras:
      - Remove quaisquer caracteres não numéricos
      - Se resultado vazio ou '0000' -> retorna None (NA)
      - Garante 4 dígitos via zfill e corta os primeiros 4 caracteres
    """
    s = series.astype(str).fillna('').str.strip()
    # Manter apenas dígitos
    s = s.str.replace(r'[^0-9]', '', regex=True)
    # Se vazio, normalizar para None
    s = s.where(s.str.len() > 0, other=None)
    # Garantir pelo menos 4 dígitos e cortar
    s = s.map(lambda x: None if x is None else x.zfill(4)[:4])
    # Rejeitar '0000' que sinaliza ausência
    s = s.where(lambda x: x != '0000', other=None)
    # Finalmente, validar 4 dígitos numéricos
    s = s.where(s.str.match(r'^[0-9]{4}$'), other=None)
    return s


def classificar_ti(df, coluna_cbo_familia='cbo_familia', cbo_ti=None, logger=None):
    if cbo_ti is None:
        try:
            from scripts.utils import CBOS_TI  # se existir global
            cbo_ti = CBOS_TI
            if logger: logger.info("Lista CBOS_TI global utilizada.")
        except Exception:
            cbo_ti = []
            if logger: logger.warning("Lista de CBO TI não fornecida nem encontrada globalmente.")
    if coluna_cbo_familia not in df.columns:
        if logger: logger.error(f"Coluna {coluna_cbo_familia} não encontrada para classificação TI.")
        df['eh_ti'] = False
        return df
    df['eh_ti'] = df[coluna_cbo_familia].isin(cbo_ti)
    if logger:
        logger.info(f"Classificação TI aplicada: {df['eh_ti'].sum():,} registros TI ({df['eh_ti'].mean()*100:.2f}%)")
    return df
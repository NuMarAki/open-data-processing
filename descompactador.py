# -*- coding: utf-8 -*-
"""Módulo especializado em descompactação de arquivos"""

import os
import py7zr
import zipfile
import tempfile
from typing import Optional, List, Tuple
from utils import logger, obter_tamanho_arquivo_mb, CacheManager, arquivo_existe_e_tamanho_ok, criar_caminho
import re


class Descompactador:
    """Gerencia descompactação de arquivos com cache inteligente"""
    
    def __init__(self, tipo_base: str, destino_base: str):
        self.tipo = tipo_base
        self.destino_base = destino_base
        self.cache = CacheManager(f'.cache/cache_{tipo_base}.json')
        
        # Diretório para arquivos descompactados
        self.dir_descompactados = criar_caminho(destino_base, 'descompactados')
        os.makedirs(self.dir_descompactados, exist_ok=True)
    
    def descompactar_arquivo(self, arquivo_compactado: str) -> Optional[str]:
        """Descompacta arquivo com verificação de cache"""
        if not os.path.exists(arquivo_compactado):
            logger.error(f"Arquivo não encontrado: {arquivo_compactado}")
            return None

        # Gerar chave única baseada no caminho completo
        cache_key = self.cache.gerar_chave_unica(arquivo_compactado, self.tipo)
        arquivo_final = self._gerar_nome_descompactado(arquivo_compactado)
        
        # Verificar se já existe
        if self._validar_arquivo_existente(arquivo_final, arquivo_compactado, cache_key):
            return arquivo_final
        
        # Verificar cache
        if self._verificar_cache(cache_key, arquivo_compactado, arquivo_final):
            return arquivo_final
        
        # Descompactar
        return self._executar_descompactacao(arquivo_compactado, arquivo_final, cache_key)
    
    def _validar_arquivo_existente(self, arquivo: str, origem: str, cache_key: str) -> bool:
        """Valida se arquivo já existe e é válido"""
        if arquivo_existe_e_tamanho_ok(arquivo):
            tamanho = obter_tamanho_arquivo_mb(arquivo)
            logger.info(f"📁 Arquivo já existe: {os.path.basename(arquivo)} ({tamanho:.1f}MB) - Origem: {origem}")
            self.cache.set_success(cache_key, origem, arquivo)
            return True
        return False
    
    def _verificar_cache(self, cache_key: str, origem: str, destino: str) -> bool:
        """Verifica informações do cache"""
        should_skip, motivo = self.cache.should_skip(cache_key, origem, destino)
        if should_skip and "já processado e válido" in motivo:
            cached_info = self.cache.get(cache_key)
            arquivo_cache = cached_info.get('arquivo_txt')
            if arquivo_cache and os.path.exists(arquivo_cache):
                logger.info(f"📁 Usando do cache: {os.path.basename(arquivo_cache)} - {motivo}")
                return True
        elif should_skip and "Falhou" in motivo:
            logger.warning(f"⚠️ Pulando arquivo que já falhou: {motivo}")
            return True
        return False
    
    def _executar_descompactacao(self, origem: str, destino: str, cache_key: str) -> Optional[str]:
        """Executa a descompactação propriamente dita"""
        if arquivo_existe_e_tamanho_ok(destino):
            tamanho = obter_tamanho_arquivo_mb(destino)
            logger.info(f"📁 Arquivo final já existe: {os.path.abspath(destino)} ({tamanho:.1f}MB)")
            self.cache.set_success(cache_key, origem, destino)
            return destino

        logger.info(f"🔄 Descompactando de: {os.path.abspath(origem)} para: {os.path.abspath(destino)}")
        
        with tempfile.TemporaryDirectory(prefix=f"temp_{self.tipo}_") as temp_dir:
            try:
                # Descompactar
                if origem.lower().endswith('.7z'):
                    sucesso = self._descompactar_7z(origem, temp_dir)
                elif origem.lower().endswith('.zip'):
                    sucesso = self._descompactar_zip(origem, temp_dir)
                else:
                    logger.error(f"Formato não suportado: {origem}")
                    self.cache.set_failed(cache_key, "Formato não suportado")
                    return None
                
                if not sucesso:
                    self.cache.set_failed(cache_key, "Falha na descompactação", origem)
                    return None
                
                # Encontrar e mover arquivo principal
                arquivo_txt = self._processar_arquivos_descompactados(temp_dir, destino)
                
                if arquivo_txt:
                    self.cache.set_success(cache_key, origem, arquivo_txt)
                    tamanho = obter_tamanho_arquivo_mb(arquivo_txt)
                    logger.info(f"✅ Descompactado: {os.path.basename(arquivo_txt)} ({tamanho:.1f}MB)")
                    return arquivo_txt
                else:
                    self.cache.set_failed(cache_key, "Nenhum arquivo válido encontrado", origem)
                    return None
                    
            except Exception as e:
                logger.error(f"Erro na descompactação: {e}")
                self.cache.set_failed(cache_key, str(e))
                return None
    
    def _descompactar_7z(self, arquivo: str, destino: str) -> bool:
        """Descompacta arquivo 7z"""
        try:
            with py7zr.SevenZipFile(arquivo, 'r') as archive:
                archive.extractall(destino)
            return True
        except Exception as e:
            logger.error(f"Erro ao descompactar 7z: {e}")
            return False
    
    def _descompactar_zip(self, arquivo: str, destino: str) -> bool:
        """Descompacta arquivo zip"""
        try:
            with zipfile.ZipFile(arquivo, 'r') as archive:
                archive.extractall(destino)
            return True
        except Exception as e:
            logger.error(f"Erro ao descompactar zip: {e}")
            return False
    
    def _processar_arquivos_descompactados(self, temp_dir: str, destino_final: str) -> Optional[str]:
        """Processa arquivos descompactados e move o principal"""
        # Encontrar arquivo TXT principal
        arquivo_principal = self._encontrar_arquivo_principal(temp_dir)
        
        if not arquivo_principal:
            return None
        
        # Mover para destino final
        try:
            import shutil
            os.makedirs(os.path.dirname(destino_final), exist_ok=True)
            shutil.move(arquivo_principal, destino_final)
            return destino_final
        except Exception as e:
            logger.error(f"Erro ao mover arquivo: {e}")
            return None
    
    def _encontrar_arquivo_principal(self, diretorio: str) -> Optional[str]:
        """Encontra arquivo TXT principal no diretório"""
        arquivos_txt = []
        
        for root, _, files in os.walk(diretorio):
            for file in files:
                if file.lower().endswith('.txt'):
                    caminho_completo = os.path.join(root, file)
                    tamanho = obter_tamanho_arquivo_mb(caminho_completo)
                    arquivos_txt.append((caminho_completo, tamanho))
        
        if not arquivos_txt:
            return None
        
        # Retornar o maior arquivo
        arquivos_txt.sort(key=lambda x: x[1], reverse=True)
        return arquivos_txt[0][0]
    
    def _gerar_nome_descompactado(self, arquivo_origem: str) -> str:
        """Gera nome padronizado para arquivo descompactado, incluindo o ano se possível"""
        nome_base = os.path.basename(arquivo_origem)
        nome_sem_ext = os.path.splitext(nome_base)[0]
        # Remover sufixos de data duplicados
        nome_limpo = re.sub(r'(_\d{8})$', '', nome_sem_ext)
        # Tentar extrair o ano do caminho
        ano_match = re.search(r'(\\|/)(20\d{2})(\\|/|_|\.|$)', arquivo_origem)
        ano = ano_match.group(2) if ano_match else None
        if ano:
            nome_final = f"{nome_limpo}_{ano}.txt"
        else:
            nome_final = f"{nome_limpo}.txt"
        return criar_caminho(self.dir_descompactados, nome_final)
    
    def listar_descompactados(self) -> List[Tuple[str, float]]:
        """Lista arquivos descompactados com tamanhos"""
        arquivos = []
        
        if os.path.exists(self.dir_descompactados):
            for arquivo in os.listdir(self.dir_descompactados):
                if arquivo.endswith('.txt'):
                    caminho = os.path.join(self.dir_descompactados, arquivo)
                    tamanho = obter_tamanho_arquivo_mb(caminho)
                    arquivos.append((arquivo, tamanho))
        
        return sorted(arquivos, key=lambda x: x[0])
    
    def _extrair_ano_nome_arquivo(self, nome_arquivo):
        # Tenta encontrar um ano no nome do arquivo
        match = re.search(r'(\d{4})', nome_arquivo)
        if match:
            return match.group(1)
        return None
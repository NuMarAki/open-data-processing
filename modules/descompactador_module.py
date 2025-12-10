"""Módulo de gerenciamento de descompactação"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processamento.descompactador import Descompactador
from scripts.utils import listar_erros_descompactacao, criar_caminho
from scripts.config_manager import ConfigManager

class GerenciadorDescompactacao:
    """Interface unificada para descompactação de arquivos"""
    
    BASES = ['pnad', 'rais', 'caged']
    
    def __init__(self):
        """Inicializa gerenciador e carrega configurações"""
        self.config_manager = ConfigManager()
        self._configs = {}
        
    def _carregar_config(self, base: str):
        """Carrega configuração da base se ainda não foi carregada"""
        if base not in self._configs:
            arquivo_cfg = criar_caminho('config', f'{base}.cfg')
            self._configs[base] = self.config_manager.carregar_configuracao(base, arquivo_cfg)
        return self._configs[base]
    
    def _obter_descompactador(self, base: str) -> Descompactador:
        """Cria uma instância de Descompactador para a base especificada"""
        config = self._carregar_config(base)
        destino_base = config.caminho_destino
        return Descompactador(base, destino_base)
    
    def descompactar_base(self, base: str, **kwargs):
        """Descompacta uma base de dados"""
        if base not in self.BASES:
            raise ValueError(f"Base desconhecida: {base}")
        
        try:
            print(f"\n[*] Descompactando {base.upper()}...")
            config = self._carregar_config(base)
            descompactador = self._obter_descompactador(base)
            
            # Buscar arquivos compactados no caminho configurado
            origem_base = config.caminho_arquivos_compactados
            if not os.path.exists(origem_base):
                print(f"[!] Diretório não encontrado: {origem_base}")
                print(f"[i] Configure o caminho em: config/{base}.cfg (caminho_arquivos_compactados)")
                return None
            
            arquivos = []
            for raiz, _, files in os.walk(origem_base):
                for f in files:
                    if f.endswith(('.7z', '.zip')):
                        arquivos.append(os.path.join(raiz, f))
            
            if not arquivos:
                print(f"[!] Nenhum arquivo compactado encontrado em {origem_base}")
                return None
            
            resultados = []
            for caminho_completo in arquivos:
                resultado = descompactador.descompactar_arquivo(caminho_completo)
                if resultado:
                    resultados.append(resultado)
            
            print(f"[✓] {base.upper()} descompactado: {len(resultados)} arquivo(s)")
            return resultados
        except Exception as e:
            print(f"[✗] Erro ao descompactar {base}: {e}")
            raise
    
    def descompactar_todas(self, **kwargs):
        """Descompacta todas as bases"""
        resultados = {}
        
        print("\n[*] Descompactando todas as bases...")
        for base in self.BASES:
            try:
                resultados[base] = self.descompactar_base(base, **kwargs)
            except Exception as e:
                print(f"[!] Erro ao descompactar {base}: {e}")
                resultados[base] = None
        
        print("[✓] Descompactação concluída!")
        return resultados
    
    def listar_erros(self):
        """Lista arquivos com erro de descompactação"""
        print("\n[*] Listando erros de descompactação...")
        erros = listar_erros_descompactacao()
        
        if not erros:
            print("[✓] Nenhum erro encontrado!")
        else:
            print(f"[!] {len(erros)} arquivo(s) com erro:")
            for erro in erros:
                print(f"  - {erro}")
        
        return erros

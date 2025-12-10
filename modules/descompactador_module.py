"""Módulo de gerenciamento de descompactação"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processamento.descompactador import Descompactador
from scripts.utils import listar_erros_descompactacao

class GerenciadorDescompactacao:
    """Interface unificada para descompactação de arquivos"""
    
    BASES = ['pnad', 'rais', 'caged']
    
    def __init__(self):
        self.descompactador = Descompactador()
    
    def descompactar_base(self, base: str, **kwargs):
        """Descompacta uma base de dados"""
        if base not in self.BASES:
            raise ValueError(f"Base desconhecida: {base}")
        
        try:
            print(f"\n[*] Descompactando {base.upper()}...")
            resultado = self.descompactador.descompactar_base(base, **kwargs)
            print(f"[✓] {base.upper()} descompactado com sucesso!")
            return resultado
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

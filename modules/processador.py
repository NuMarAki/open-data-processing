"""Módulo unificado de processamento de bases"""
import os
from processadores_especificos import ProcessadorPNAD, ProcessadorRAIS, ProcessadorCAGED

class ProcessadorBases:
    """Interface unificada para processamento de bases"""
    
    # Diretório de configurações
    CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
    
    MAPEAMENTO_PROCESSADORES = {
        'pnad': (ProcessadorPNAD, 'pnad.cfg'),
        'rais': (ProcessadorRAIS, 'rais.cfg'),
        'caged': (ProcessadorCAGED, 'caged.cfg'),
    }
    
    @staticmethod
    def obter_processador(base: str):
        """Obtém o processador correto para a base"""
        if base not in ProcessadorBases.MAPEAMENTO_PROCESSADORES:
            raise ValueError(f"Base desconhecida: {base}")
        
        processador_class, config_file = ProcessadorBases.MAPEAMENTO_PROCESSADORES[base]
        # Retorna com caminho completo para o arquivo de configuração
        config_path = os.path.join(ProcessadorBases.CONFIG_DIR, config_file)
        return processador_class, config_path
    
    @staticmethod
    def processar(base: str, **kwargs):
        """Processa uma base de dados"""
        try:
            processador_class, config_file = ProcessadorBases.obter_processador(base)
            print(f"\n[*] Iniciando processamento de {base.upper()}...")
            print(f"    Configuração: {config_file}")
            processador = processador_class()
            resultado = processador.processar(**kwargs)
            print(f"[✓] {base.upper()} processado com sucesso!")
            return resultado
        except Exception as e:
            print(f"[✗] Erro ao processar {base}: {e}")
            raise
    
    @staticmethod
    def processar_todas(**kwargs):
        """Processa todas as bases"""
        bases = ['pnad', 'rais', 'caged']
        resultados = {}
        
        for base in bases:
            try:
                resultados[base] = ProcessadorBases.processar(base, **kwargs)
            except Exception as e:
                print(f"[!] Erro ao processar {base}: {e}")
                resultados[base] = None
        
        return resultados

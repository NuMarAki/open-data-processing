"""Módulo unificado de processamento de bases"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processamento.processadores_especificos import ProcessadorPNAD, ProcessadorRAIS, ProcessadorCAGED

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
            from scripts.config_manager import ConfigManager
            
            processador_class, config_file = ProcessadorBases.obter_processador(base)
            print(f"\n[*] Iniciando processamento de {base.upper()}...")
            print(f"    Configuração: {config_file}")
            
            # Carregar configuração
            config_manager = ConfigManager()
            config = config_manager.carregar_configuracao(base, config_file)
            
            # Instanciar processador com config
            processador = processador_class(config)
            resultado = processador.processar_periodo_completo(**kwargs)
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

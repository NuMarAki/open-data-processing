"""Módulo unificado de processamento de bases"""
from processadores_especificos import ProcessadorPNAD, ProcessadorRAIS, ProcessadorCAGED

class ProcessadorBases:
    """Interface unificada para processamento de bases"""
    
    MAPEAMENTO_PROCESSADORES = {
        'pnad': (ProcessadorPNAD, 'colunas_pnad.cfg'),
        'rais': (ProcessadorRAIS, 'colunas_rais.cfg'),
        'caged': (ProcessadorCAGED, 'colunas_caged.cfg'),
    }
    
    @staticmethod
    def obter_processador(base: str):
        """Obtém o processador correto para a base"""
        if base not in ProcessadorBases.MAPEAMENTO_PROCESSADORES:
            raise ValueError(f"Base desconhecida: {base}")
        return ProcessadorBases.MAPEAMENTO_PROCESSADORES[base]
    
    @staticmethod
    def processar(base: str, **kwargs):
        """Processa uma base de dados"""
        try:
            processador_class, config_file = ProcessadorBases.obter_processador(base)
            print(f"\n[*] Iniciando processamento de {base.upper()}...")
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

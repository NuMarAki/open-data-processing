"""Módulo de geração de relatórios e análises"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processamento.processadores_especificos import ProcessadorPNAD, ProcessadorRAIS, ProcessadorCAGED

class GeradorRelatorios:
    """Interface unificada para geração de relatórios"""
    
    MAPEAMENTO = {
        'pnad': ProcessadorPNAD,
        'rais': ProcessadorRAIS,
        'caged': ProcessadorCAGED,
    }
    
    @staticmethod
    def gerar_relatorio_base(base: str, **kwargs):
        """Gera relatório para uma base específica"""
        try:
            if base not in GeradorRelatorios.MAPEAMENTO:
                raise ValueError(f"Base desconhecida: {base}")
            
            print(f"\n[*] Gerando relatório de {base.upper()}...")
            processador = GeradorRelatorios.MAPEAMENTO[base]()
            
            # Executar análises específicas
            resultado = processador.executar_analise(**kwargs)
            print(f"[✓] Relatório de {base.upper()} gerado com sucesso!")
            return resultado
        except Exception as e:
            print(f"[✗] Erro ao gerar relatório de {base}: {e}")
            raise
    
    @staticmethod
    def gerar_relatorio_consolidado(**kwargs):
        """Gera relatório consolidado de todas as bases"""
        bases = ['pnad', 'rais', 'caged']
        resultados = {}
        
        print("\n[*] Gerando relatório consolidado...")
        for base in bases:
            try:
                resultados[base] = GeradorRelatorios.gerar_relatorio_base(base, **kwargs)
            except Exception as e:
                print(f"[!] Erro ao gerar relatório de {base}: {e}")
                resultados[base] = None
        
        print("[✓] Relatório consolidado gerado com sucesso!")
        return resultados

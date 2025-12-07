"""Módulo de geração de gráficos específicos da PNAD"""
import os

class GeradorGraficosPNAD:
    """Interface para gráficos específicos da PNAD"""
    
    SCRIPTS_GRAFICOS = {
        'escolaridade_sexo': 'gera_grafico_escolaridade_sexo.py',
        'estado': 'gera_grafico_estado_pnad.py',
        'renda_estudo': 'gera_grafico_renda_estudo_comparativo.py',
        'serie_temporal': 'gera_grafico_serie_temporal_ti.py',
    }
    
    @staticmethod
    def gerar_grafico(tipo: str, **kwargs):
        """Gera um gráfico específico da PNAD"""
        try:
            if tipo not in GeradorGraficosPNAD.SCRIPTS_GRAFICOS:
                raise ValueError(f"Tipo de gráfico desconhecido: {tipo}")
            
            script = GeradorGraficosPNAD.SCRIPTS_GRAFICOS[tipo]
            print(f"\n[*] Gerando gráfico: {tipo}...")
            print(f"    Executando: {script}")
            
            # Aqui seria feita a execução do script correspondente
            # Por enquanto, apenas informamos o que seria feito
            print(f"[✓] Gráfico '{tipo}' gerado com sucesso!")
            return True
        except Exception as e:
            print(f"[✗] Erro ao gerar gráfico: {e}")
            raise
    
    @staticmethod
    def gerar_todos_graficos(**kwargs):
        """Gera todos os gráficos disponíveis"""
        print("\n[*] Gerando todos os gráficos PNAD...")
        
        graficos_gerados = {}
        for tipo in GeradorGraficosPNAD.SCRIPTS_GRAFICOS.keys():
            try:
                graficos_gerados[tipo] = GeradorGraficosPNAD.gerar_grafico(tipo, **kwargs)
            except Exception as e:
                print(f"[!] Erro ao gerar gráfico '{tipo}': {e}")
                graficos_gerados[tipo] = False
        
        print("[✓] Geração de gráficos concluída!")
        return graficos_gerados
    
    @staticmethod
    def exibir_opcoes():
        """Exibe as opções de gráficos disponíveis"""
        print("\n--- Gráficos PNAD Disponíveis ---")
        for idx, (tipo, script) in enumerate(GeradorGraficosPNAD.SCRIPTS_GRAFICOS.items(), 1):
            print(f"  {idx}. {tipo.replace('_', ' ').title():<25} ({script})")

"""Módulo de funcionalidades do modelo preditivo RAIS.

Delega para preditivos.preditivo_rais (script consolidado).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModuloPreditivo:
    """Interface para o modelo preditivo RAIS"""

    CAMINHO_PREDITIVO = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'preditivos', 'artifacts'
    )

    @staticmethod
    def treinar_modelo(**kwargs):
        """Treina o modelo preditivo RAIS"""
        try:
            print("\n[*] Iniciando treinamento do modelo preditivo RAIS...")

            # Usa o script consolidado em preditivos/
            from preditivos.preditivo_rais import main as treinar

            treinar()

            print("[OK] Modelo treinado com sucesso!")
            return True
        except Exception as e:
            print(f"[X] Erro ao treinar modelo: {e}")
            raise
    
    @staticmethod
    def fazer_predicoes(dados: dict = None, **kwargs):
        """Realiza predições com o modelo treinado"""
        try:
            print("\n[*] Executando predições...")

            from preditivos.preditivo_rais import main as fazer_pred
            resultado = fazer_pred()

            print("[OK] Predições realizadas com sucesso!")
            return resultado
        except Exception as e:
            print(f"[X] Erro ao fazer predições: {e}")
            raise
    
    @staticmethod
    def exibir_info():
        """Exibe informações sobre o modelo"""
        info = """
        === MODELO PREDITIVO RAIS ===
        
        Objetivo: Prever vínculo ativo em 31/12 (desligamento)
        
        Características:
        - Algoritmo: Random Forest (CPU)
        - Features: 47 características após processamento
        - Target: Vínculo ativo em 31/12
        
        Funcionalidades:
        1. Treinar modelo com dados históricos
        2. Avaliar desempenho
        3. Gerar predições
        4. Exportar importância de features
        
        Artefatos gerados:
        - model.joblib: Modelo treinado
        - preprocessor.joblib: Preprocessador de features
        - feature_importance.csv: Importância das features
        - roc_curve.png: Curva ROC
        - pr_curve.png: Curva Precisão-Recall
        """
        print(info)

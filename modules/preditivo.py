"""Módulo de funcionalidades do modelo preditivo RAIS"""
import os
import sys

class ModuloPreditivo:
    """Interface para o modelo preditivo RAIS"""
    
    CAMINHO_PREDITIVO = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        '..', 'preditivo_rais'
    )
    
    @staticmethod
    def treinar_modelo(**kwargs):
        """Treina o modelo preditivo RAIS"""
        try:
            print("\n[*] Iniciando treinamento do modelo preditivo RAIS...")
            
            # Adicionar caminho do módulo preditivo ao sys.path
            if ModuloPreditivo.CAMINHO_PREDITIVO not in sys.path:
                sys.path.insert(0, ModuloPreditivo.CAMINHO_PREDITIVO)
            
            # Importar e executar o módulo preditivo
            from src.main import main as treinar
            treinar()
            
            print("[✓] Modelo treinado com sucesso!")
            return True
        except Exception as e:
            print(f"[✗] Erro ao treinar modelo: {e}")
            raise
    
    @staticmethod
    def fazer_predicoes(dados: dict = None, **kwargs):
        """Realiza predições com o modelo treinado"""
        try:
            print("\n[*] Executando predições...")
            
            if ModuloPreditivo.CAMINHO_PREDITIVO not in sys.path:
                sys.path.insert(0, ModuloPreditivo.CAMINHO_PREDITIVO)
            
            from src.main import main as fazer_pred
            resultado = fazer_pred()
            
            print("[✓] Predições realizadas com sucesso!")
            return resultado
        except Exception as e:
            print(f"[✗] Erro ao fazer predições: {e}")
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

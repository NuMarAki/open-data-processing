"""Módulo de funcionalidades do modelo preditivo RAIS.

Delega para preditivos.preditivo_rais (script consolidado).
"""
import os
import sys
import subprocess
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

    # ------------------ RAIS ------------------
    @staticmethod
    def info_rais():
        print(
            """
        === MODELO PREDITIVO RAIS ===

        Objetivo: Prever vínculo ativo em 31/12 (desligamento)
        Algoritmo: Random Forest (CPU)
        Features: ~47 após processamento
        Alvo: vínculo_ativo_31_12
        Amostragem padrão: 100% dos dados
            - Para amostrar: adicione --sample-frac 0.5 (50%) ao executar preditivos/preditivo_rais.py
            - Para limitar RAM: use --rows-per-file 50000 --max-files 30
        Artefatos: model.joblib, preprocessor.joblib, feature_importance.csv, curvas ROC/PR
        Resultado: resultados/rais/ativo_3112/
            """
        )

    @staticmethod
    def treinar_rais(**kwargs):
        return ModuloPreditivo.treinar_modelo(**kwargs)

    @staticmethod
    def predizer_rais(**kwargs):
        return ModuloPreditivo.fazer_predicoes(**kwargs)

    # ------------------ PNAD 6 SM ------------------
    @staticmethod
    def info_pnad_salario_6sm():
        print(
            """
        === MODELO PREDITIVO PNAD (6 SM) ===

        Alvo: renda >= 6 salários mínimos (ajustado por ano)
        Algoritmo: Random Forest com peso_populacional
        Fonte: dados/pnad/preprocessados/ (53 arquivos CSV)
        Amostragem padrão: 100% dos dados
            - Para amostrar: adicione --sample-frac 0.5 (50%) ao executar diretamente
        Features: idade, anos_estudo, uf, sexo, cor_raca, nivel_instrucao, etc.
        Saída: resultados/pnad/salario_6sm/ (modelo.joblib, predições, resumo.txt)
        
        Execução direta:
            python preditivos/preditivo_pnad_alvo.py --pnad-dir dados/pnad/preprocessados --alvo salario_6sm --n_sm 6 --usar_peso --sample-frac 0.5
            """
        )

    @staticmethod
    def _run_pnad_alvo(n_sm: int = 6, usar_peso: bool = True, salarios_minimos: str = None, sample_frac: float = None, fast: bool = False, n_boot: int = None, n_repeats: int = None):
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'preditivos', 'preditivo_pnad_alvo.py')
        script_path = os.path.abspath(script_path)

        if not os.path.exists(script_path):
            print(f"[X] Script não encontrado: {script_path}")
            return False

        pnad_dir = os.path.join('dados', 'pnad', 'preprocessados')
        if not os.path.exists(pnad_dir):
            print(f"[X] Diretório PNAD não encontrado: {pnad_dir}")
            return False

        # Em modo fast, usar 25% amostragem para maior velocidade
        if fast and sample_frac is None:
            sample_frac = 0.25

        cmd = [sys.executable, script_path, '--pnad-dir', pnad_dir, '--alvo', 'salario_6sm', '--n_sm', str(n_sm)]
        if sample_frac is not None:
            cmd.extend(['--sample-frac', str(sample_frac)])
        if fast:
            cmd.append('--fast')
        if n_boot is not None:
            cmd.extend(['--n_boot', str(n_boot)])
        if n_repeats is not None:
            cmd.extend(['--n_repeats', str(n_repeats)])
        if usar_peso:
            cmd.append('--usar_peso')
        if salarios_minimos:
            cmd.extend(['--salarios_minimos', salarios_minimos])

        print("\n[*] Executando modelo PNAD (6 SM)...")
        print(f"    Carregando de: {pnad_dir}")
        sample_msg = f"{sample_frac*100:.0f}%" if sample_frac else "100%"
        mode_msg = "(RÁPIDO: 100 árvores, sem importância)" if fast else "(COMPLETO: 500 árvores)"
        print(f"    Amostragem: {sample_msg} {mode_msg}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            if result.returncode == 0:
                print("[OK] Modelo PNAD executado com sucesso!")
                return True
            print(f"[X] Falha ao executar modelo PNAD (rc={result.returncode})")
            return False
        except Exception as e:
            print(f"[X] Erro ao executar modelo PNAD: {e}")
            return False

    @staticmethod
    def treinar_pnad_salario_6sm(**kwargs):
        return ModuloPreditivo._run_pnad_alvo(**kwargs)

    @staticmethod
    def predizer_pnad_salario_6sm(**kwargs):
        # O script já treina/avalia e gera predições no mesmo fluxo
        return ModuloPreditivo._run_pnad_alvo(**kwargs)
    
    @staticmethod
    def exibir_info():
        """Exibe informações sobre o modelo"""
        ModuloPreditivo.info_rais()
        ModuloPreditivo.info_pnad_salario_6sm()

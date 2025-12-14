"""Módulo de geração de gráficos específicos da RAIS"""
import os
import subprocess
import time
import glob

class GeradorGraficosRAIS:
    """Interface para gráficos específicos da RAIS"""
    
    @staticmethod
    def gerar_analise_rapida():
        """Gera análise rápida dos dados RAIS preprocessados"""
        import pandas as pd
        import matplotlib.pyplot as plt
        
        print("\n[*] Gerando analise rapida RAIS...")
        
        # Criar diretório de saída
        os.makedirs('graficos', exist_ok=True)
        
        # Buscar arquivos preprocessados (separados em TI e não-TI)
        preprocessados_dir = 'dados/rais/preprocessados'
        ti_files = glob.glob(os.path.join(preprocessados_dir, '*_ti.csv'))
        nao_ti_files = glob.glob(os.path.join(preprocessados_dir, '*_nao_ti.csv'))
        
        if not ti_files and not nao_ti_files:
            print("[!] Nenhum arquivo preprocessado encontrado")
            return False
        
        print(f"[*] Encontrados {len(ti_files)} arquivos TI e {len(nao_ti_files)} arquivos Nao-TI")
        print(f"[*] Carregando amostras (50k registros por arquivo)...")
        
        # Carregar amostras
        ti_count = 0
        nao_ti_count = 0
        
        # Processar arquivos TI
        for csv_file in ti_files[:10]:  # Limitar a 10 arquivos
            try:
                df_temp = pd.read_csv(csv_file, sep=';', nrows=50000)
                ti_count += len(df_temp)
            except Exception as e:
                print(f"[!] Erro ao ler {os.path.basename(csv_file)}: {e}")
        
        # Processar arquivos Não-TI
        for csv_file in nao_ti_files[:10]:  # Limitar a 10 arquivos
            try:
                df_temp = pd.read_csv(csv_file, sep=';', nrows=50000)
                nao_ti_count += len(df_temp)
            except Exception as e:
                print(f"[!] Erro ao ler {os.path.basename(csv_file)}: {e}")
        
        total = ti_count + nao_ti_count
        print(f"[OK] {total:,} registros carregados (amostras)")
        
        print(f"\n=== RESUMO RAIS ===")
        print(f"Total de registros: {total:,}")
        print(f"Profissionais TI: {ti_count:,} ({100*ti_count/total:.2f}%)")
        print(f"Outros profissionais: {nao_ti_count:,} ({100*nao_ti_count/total:.2f}%)")
        
        # Gráfico simples de distribuição
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        categorias = ['TI', 'Nao TI']
        valores = [ti_count, nao_ti_count]
        cores = ['#2ecc71', '#95a5a6']
        
        ax.bar(categorias, valores, color=cores, alpha=0.8)
        ax.set_title('Distribuicao de Profissionais - RAIS', fontsize=14, fontweight='bold')
        ax.set_ylabel('Quantidade de Registros', fontsize=12)
        
        # Adicionar valores nas barras
        for i, v in enumerate(valores):
            ax.text(i, v + max(valores)*0.02, f'{v:,}\n({100*v/total:.1f}%)', 
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        output_path = 'graficos/rais_distribuicao_ti.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n[OK] Grafico salvo: {output_path}")
        return True
    
    @staticmethod
    def exibir_opcoes():
        """Exibe as opções de análises disponíveis"""
        print("\n--- Graficos RAIS Disponiveis ---")
        print("  1. Analise Rapida               (Distribuicao TI vs Nao-TI)")
        print("  2. Modelo Preditivo             (Menu Principal -> Opcao 5)")

"""Módulo de geração de gráficos específicos do CAGED"""
import os
import subprocess
import time
import glob

class GeradorGraficosCAGED:
    """Interface para gráficos específicos do CAGED"""
    
    @staticmethod
    def gerar_analise_rapida():
        """Gera análise rápida dos dados CAGED preprocessados"""
        import pandas as pd
        import matplotlib.pyplot as plt
        
        print("\n[*] Gerando analise rapida CAGED...")
        
        # Criar diretório de saída
        os.makedirs('graficos', exist_ok=True)
        
        # Buscar arquivos preprocessados
        preprocessados_dir = 'dados/caged/preprocessados'
        csv_files = glob.glob(os.path.join(preprocessados_dir, '*_processado.csv'))
        
        if not csv_files:
            print("[!] Nenhum arquivo preprocessado encontrado")
            return False
        
        print(f"[*] Carregando dados de {len(csv_files)} arquivos...")
        
        # Carregar amostra de cada arquivo (anos 2023-2024 para ser mais rápido)
        dfs = []
        for csv_file in csv_files:
            if '2023' in csv_file or '2024' in csv_file:
                try:
                    df_temp = pd.read_csv(csv_file, sep=';', nrows=50000)
                    if 'eh_ti' in df_temp.columns and 'ano' in df_temp.columns:
                        dfs.append(df_temp)
                except Exception as e:
                    print(f"[!] Erro ao ler {os.path.basename(csv_file)}: {e}")
        
        if not dfs:
            print("[!] Nenhum dado com colunas necessarias encontrado")
            return False
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"[OK] {len(df):,} registros carregados")
        
        # Análise básica
        total = len(df)
        ti_count = df['eh_ti'].sum() if 'eh_ti' in df.columns else 0
        nao_ti_count = total - ti_count
        
        print(f"\n=== RESUMO CAGED ===")
        print(f"Total de registros: {total:,}")
        print(f"Profissionais TI: {ti_count:,} ({100*ti_count/total:.2f}%)")
        print(f"Outros profissionais: {nao_ti_count:,} ({100*nao_ti_count/total:.2f}%)")
        
        # Gráfico 1: Distribuição TI vs Não-TI
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        categorias = ['TI', 'Nao TI']
        valores = [ti_count, nao_ti_count]
        cores = ['#3498db', '#95a5a6']
        
        ax1.bar(categorias, valores, color=cores, alpha=0.8)
        ax1.set_title('Distribuicao de Profissionais - CAGED', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Quantidade de Registros', fontsize=12)
        
        for i, v in enumerate(valores):
            ax1.text(i, v + max(valores)*0.02, f'{v:,}\n({100*v/total:.1f}%)', 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Gráfico 2: Distribuição por ano (se disponível)
        if 'ano' in df.columns:
            por_ano = df.groupby(['ano', 'eh_ti']).size().unstack(fill_value=0)
            por_ano.plot(kind='bar', ax=ax2, color=cores, alpha=0.8)
            ax2.set_title('Movimentacoes por Ano - CAGED', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Ano', fontsize=12)
            ax2.set_ylabel('Quantidade', fontsize=12)
            ax2.legend(['TI', 'Nao TI'], loc='upper left')
            ax2.tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        output_path = 'graficos/caged_distribuicao_ti.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n[OK] Grafico salvo: {output_path}")
        return True
    
    @staticmethod
    def exibir_opcoes():
        """Exibe as opções de análises disponíveis"""
        print("\n--- Graficos CAGED Disponiveis ---")
        print("  1. Analise Rapida               (Distribuicao TI e por ano)")

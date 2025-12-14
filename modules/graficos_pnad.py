"""Módulo de geração de gráficos específicos da PNAD"""
import os
import subprocess
import time
import glob

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
            script_path = os.path.join('graficos_scripts', script)
            
            print(f"\n[*] Gerando grafico: {tipo}...")
            print(f"    Executando: {script}")
            
            # Executa o script usando subprocess
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Mostrar output do script (filtrar warnings)
            if result.stdout:
                linhas = result.stdout.split('\n')
                for linha in linhas:
                    if linha.strip() and 'FutureWarning' not in linha and 'DeprecationWarning' not in linha:
                        print(f"    {linha}")
            
            if result.returncode == 0:
                print(f"[OK] Grafico '{tipo}' gerado com sucesso!")
                
                # Listar arquivos gerados recentemente no diretório graficos
                import glob
                from datetime import datetime, timedelta
                graficos_dir = 'graficos'
                if os.path.exists(graficos_dir):
                    agora = datetime.now()
                    arquivos_recentes = []
                    for arquivo in glob.glob(os.path.join(graficos_dir, '*.png')) + glob.glob(os.path.join(graficos_dir, '*.csv')):
                        mtime = datetime.fromtimestamp(os.path.getmtime(arquivo))
                        if (agora - mtime).total_seconds() < 60:  # Gerados nos últimos 60 segundos
                            arquivos_recentes.append(os.path.basename(arquivo))
                    
                    if arquivos_recentes:
                        print(f"\n    Arquivos gerados (pasta 'graficos'):")
                        for arq in sorted(arquivos_recentes):
                            print(f"      - {arq}")
                
                return True
            else:
                print(f"[ERRO] Falha ao gerar grafico '{tipo}'")
                if result.stderr:
                    print(f"Erro: {result.stderr[:500]}")
                return False
                
        except Exception as e:
            print(f"[ERRO] Erro ao gerar grafico: {e}")
            return False
    
    @staticmethod
    def gerar_todos_graficos(**kwargs):
        """Gera todos os gráficos disponíveis e exibe sumário"""
        print("\n" + "="*60)
        print("[*] Executando todos os graficos PNAD...")
        print("="*60)
        
        resultados = []
        tempo_total_inicio = time.time()
        
        # Executa cada script sequencialmente
        for idx, (tipo, script) in enumerate(GeradorGraficosPNAD.SCRIPTS_GRAFICOS.items(), 1):
            print(f"\n[{idx}/4] Processando: {tipo.replace('_', ' ').title()}")
            script_path = os.path.join('graficos_scripts', script)
            
            inicio = time.time()
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            duracao = time.time() - inicio
            
            status = "[OK]" if result.returncode == 0 else "[ERRO]"
            resultados.append({
                'tipo': tipo,
                'script': script,
                'duracao': duracao,
                'status': status,
                'returncode': result.returncode
            })
            
            print(f"     {status} Concluido em {duracao:.1f}s")
        
        tempo_total = time.time() - tempo_total_inicio
        
        # Exibe sumário
        print("\n" + "="*60)
        print("=== SUMARIO DE GERACAO ===")
        print("="*60)
        
        for r in resultados:
            print(f"{r['status']} {r['script']:<45} {r['duracao']:>6.1f}s")
        
        print(f"\nTempo total de execucao: {tempo_total:.1f}s")
        
        # Lista arquivos gerados
        graficos_dir = 'graficos'
        if os.path.exists(graficos_dir):
            graficos = glob.glob(os.path.join(graficos_dir, '*.png'))
            csvs = glob.glob(os.path.join(graficos_dir, '*.csv'))
            
            print(f"\n=== ARQUIVOS GERADOS ===")
            print(f"Total de graficos PNG: {len(graficos)}")
            print(f"Total de arquivos CSV: {len(csvs)}")
            print(f"\nArquivos PNG:")
            for g in sorted(graficos):
                size_kb = os.path.getsize(g) / 1024
                print(f"  - {os.path.basename(g):<50} {size_kb:>8.1f} KB")
            
            if csvs:
                print(f"\nArquivos CSV:")
                for c in sorted(csvs):
                    size_kb = os.path.getsize(c) / 1024
                    print(f"  - {os.path.basename(c):<50} {size_kb:>8.1f} KB")
        
        print("\n" + "="*60)
        print("[OK] Geracao de graficos concluida!")
        print("="*60)
        
        return resultados
    
    @staticmethod
    def exibir_opcoes():
        """Exibe as opções de gráficos disponíveis"""
        print("\n--- Gráficos PNAD Disponíveis ---")
        for idx, (tipo, script) in enumerate(GeradorGraficosPNAD.SCRIPTS_GRAFICOS.items(), 1):
            print(f"  {idx}. {tipo.replace('_', ' ').title():<25} ({script})")

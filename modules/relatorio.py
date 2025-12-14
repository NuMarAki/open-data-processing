"""Módulo de geração de relatórios e análises"""
import os
import sys
import subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import logger

class GeradorRelatorios:
    """Interface unificada para geração de relatórios"""
    
    SCRIPTS_ANALISES = {
        'pnad': [
            'analises/analise_etarismo_pnad.py',
            'analises/estudo_consolidado_pnad.py',
            'analises/stats_pnad.py',
        ],
        'rais': [
            'analises/analise_etarismo.py',
        ],
        'caged': [
            'analises/diagnostico_dados.py',
        ],
    }
    
    SCRIPTS_GRAFICOS = {
        'escolaridade_sexo': 'graficos_scripts/gera_grafico_escolaridade_sexo.py',
        'estado': 'graficos_scripts/gera_grafico_estado_pnad.py',
        'renda_estudo': 'graficos_scripts/gera_grafico_renda_estudo_comparativo.py',
        'serie_temporal': 'graficos_scripts/gera_grafico_serie_temporal_ti.py',
    }
    
    @staticmethod
    def _executar_script(script: str, timeout: int = 300):
        """Executa um script Python com timeout"""
        try:
            # Caminho da raiz do projeto
            raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            caminho_script = os.path.join(raiz, script)
            
            # Verificar se o script existe
            if not os.path.exists(caminho_script):
                logger.warning(f"Script não encontrado: {caminho_script}")
                return False
            
            logger.info(f"Executando: {caminho_script} (timeout: {timeout}s)")
            print(f"    ⏳ Aguardando (máx {timeout}s)...")
            
            try:
                resultado = subprocess.run([sys.executable, caminho_script], 
                                         capture_output=True, text=True, cwd=raiz, 
                                         timeout=timeout)
                if resultado.returncode == 0:
                    logger.info(f"Script {script} executado com sucesso")
                    if resultado.stdout:
                        print(resultado.stdout)
                    return True
                else:
                    logger.warning(f"Script {script} retornou erro: {resultado.stderr}")
                    if resultado.stderr:
                        print(f"    [!] Erro: {resultado.stderr[:200]}")
                    return False
            except subprocess.TimeoutExpired:
                logger.warning(f"Script {script} excedeu timeout de {timeout}s")
                print(f"    [!] Script demorou mais de {timeout}s - abortado")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao executar {script}: {e}")
            print(f"    [✗] Erro ao executar: {e}")
            return False
    
    @staticmethod
    def _mostrar_arquivos_gerados():
        """Mostra um resumo dos arquivos gerados"""
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        print("\n[*] Resumo dos arquivos gerados:")
        print(f"    Caminho base: {raiz}\n")
        
        dirs_esperados = {
            'graficos': 'Gráficos (PNG, SVG, etc)',
            'resultados': 'Resultados das análises (CSV, dados)',
            'reports': 'Relatórios (HTML, dados consolidados)',
        }
        
        for dir_name, descricao in dirs_esperados.items():
            caminho = os.path.join(raiz, dir_name)
            if os.path.exists(caminho):
                num_arquivos = len([f for f in os.listdir(caminho) if os.path.isfile(os.path.join(caminho, f))])
                print(f"    ✓ {dir_name}/ ({descricao}) - {num_arquivos} arquivos")
                # Listar alguns arquivos
                for arquivo in os.listdir(caminho)[:3]:
                    print(f"      - {arquivo}")
                if len(os.listdir(caminho)) > 3:
                    print(f"      ... e mais {len(os.listdir(caminho)) - 3} arquivo(s)")
            else:
                print(f"    ✗ {dir_name}/ ({descricao}) - não encontrado")
    
    @staticmethod
    def _verificar_dados_preprocessados(base: str) -> bool:
        """Verifica se existem dados preprocessados para a base"""
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        caminho_preprocessados = os.path.join(raiz, 'dados', base, 'preprocessados')
        
        if not os.path.exists(caminho_preprocessados):
            return False
        
        import glob
        arquivos = glob.glob(os.path.join(caminho_preprocessados, '*.csv'))
        return len(arquivos) > 0
    
    @staticmethod
    def gerar_relatorio_base(base: str, **kwargs):
        """Gera análises para uma base específica"""
        try:
            raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Verificar se existem dados preprocessados
            if not GeradorRelatorios._verificar_dados_preprocessados(base):
                print(f"\n[!] Base {base.upper()}: Dados preprocessados nao encontrados")
                print(f"    Execute primeiro: Menu Principal -> 2. Processar Bases -> {base.upper()}")
                print(f"    Caminho esperado: dados/{base}/preprocessados/\n")
                return False
            
            # PNAD: Usar gráficos otimizados
            if base == 'pnad':
                print(f"\n[*] Gerando graficos PNAD...")
                print(f"    Os scripts de analise legados requerem arquivo consolidado inexistente.")
                print(f"    Executando graficos otimizados que usam dados preprocessados...\n")
                
                from modules.graficos_pnad import GeradorGraficosPNAD
                GeradorGraficosPNAD.gerar_todos_graficos()
                return True
            
            # RAIS: Executar análise rápida
            elif base == 'rais':
                print(f"\n[*] Gerando graficos RAIS...")
                print(f"    Executando analise rapida dos dados preprocessados...")
                print(f"    Para analise preditiva completa, use: Menu Principal -> 5. Modelo Preditivo RAIS\n")
                
                from modules.graficos_rais import GeradorGraficosRAIS
                return GeradorGraficosRAIS.gerar_analise_rapida()
            
            # CAGED: Executar análise rápida
            elif base == 'caged':
                print(f"\n[*] Gerando graficos CAGED...")
                print(f"    Executando analise rapida dos dados preprocessados...\n")
                
                from modules.graficos_caged import GeradorGraficosCAGED
                return GeradorGraficosCAGED.gerar_analise_rapida()
            
            # Fallback para outras bases (não deveria acontecer)
            else:
                print(f"[!] Base {base.upper()} nao reconhecida")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao gerar relatorio de {base}: {e}")
            print(f"[X] Erro ao gerar relatorio de {base}: {e}")
            return False
    
    @staticmethod
    def gerar_relatorio_consolidado(**kwargs):
        """Gera análises consolidadas de todas as bases"""
        print("\n" + "="*60)
        print("[*] ANALISE CONSOLIDADA - Todas as Bases")
        print("="*60)
        
        bases = ['pnad', 'rais', 'caged']
        resultados = {}
        
        print("\n[!] AVISO: Verificando disponibilidade de dados...\n")
        
        for base in bases:
            try:
                print(f"--- {base.upper()} ---")
                resultados[base] = GeradorRelatorios.gerar_relatorio_base(base, **kwargs)
            except Exception as e:
                logger.error(f"Erro ao gerar analise de {base}: {e}")
                resultados[base] = False
        
        # Resumo final
        print("\n" + "="*60)
        print("=== RESUMO ===")
        print("="*60)
        for base, sucesso in resultados.items():
            status = "[OK]" if sucesso else "[FALHA]"
            print(f"{status} {base.upper()}")
        
        if any(resultados.values()):
            print("\n[OK] Pelo menos uma base foi processada com sucesso!")
        else:
            print("\n[X] Nenhuma base foi processada. Verifique os dados preprocessados.")
        
        return resultados
    
    @staticmethod
    def gerar_todos_relatorios(**kwargs):
        """Gera todos os relatórios e gráficos"""
        print("\n[*] Gerando todos os relatórios e gráficos...")
        print("    [⚠️] AVISO: Esta operação pode demorar MUITO tempo (30+ minutos)...")
        print("    [💡] Dica: Use 'Gráficos PNAD Específicos' para resultados mais rápidos\n")
        
        # Gerar análises
        resultado_consolidado = GeradorRelatorios.gerar_relatorio_consolidado(**kwargs)
        
        # Gerar gráficos PNAD
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(f"\n[*] Gerando gráficos PNAD...")
        for tipo, script in GeradorRelatorios.SCRIPTS_GRAFICOS.items():
            caminho_completo = os.path.join(raiz, script)
            if os.path.exists(caminho_completo):
                print(f"    Executando: {script}")
                GeradorRelatorios._executar_script(script, timeout=300)
            else:
                logger.warning(f"Script de gráfico não encontrado: {caminho_completo}")
        
        # Mostrar resumo dos arquivos gerados
        GeradorRelatorios._mostrar_arquivos_gerados()
        
        print("\n[✓] Todos os relatórios e gráficos foram gerados com sucesso!")
        return resultado_consolidado

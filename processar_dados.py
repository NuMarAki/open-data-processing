# -*- coding: utf-8 -*-
"""Script unificado para processamento de dados"""

import sys
import argparse
from typing import Optional

from config_manager import config_manager
from processadores_especificos import ProcessadorPNAD, ProcessadorRAIS, ProcessadorCAGED
from processador_base import ProcessadorSeguro
from analise_etarismo import AnalisadorEtarismo
from utils import logger, configurar_log


def processar_base(tipo: str) -> Optional[str]:
    """Processa uma base de dados específica"""
    
    # Mapeamento de processadores
    processadores = {
        'pnad': (ProcessadorPNAD, 'colunas_pnad.cfg'),
        'rais': (ProcessadorRAIS, 'colunas_rais.cfg'),
        'caged': (ProcessadorCAGED, 'colunas_caged.cfg')
    }
    
    if tipo not in processadores:
        logger.error(f"Tipo de base inválido: {tipo}")
        return None
    
    ProcessadorClasse, arquivo_cfg = processadores[tipo]
    
    try:
        # Carregar configuração
        config = config_manager.carregar_configuracao(tipo, arquivo_cfg)
        
        # Criar e executar processador
        processador = ProcessadorClasse(config)
        
        with ProcessadorSeguro(processador) as proc:
            df_final = proc.processar_periodo_completo()
            
            if df_final is not None and len(df_final) > 0:
                arquivo_saida = f"resultados/{tipo}/dados_{tipo}_consolidados.csv"
                logger.info(f"✅ {tipo.upper()} processado com sucesso!")
                logger.info(f"   Total de registros: {len(df_final):,}")
                logger.info(f"   Arquivo salvo: {arquivo_saida}")
                
                # Executar análise de etarismo
                if '--analise' in sys.argv:
                    executar_analise(tipo, arquivo_saida)
                
                return arquivo_saida
            else:
                logger.error(f"❌ Falha no processamento {tipo.upper()}")
                return None
                
    except Exception as e:
        logger.error(f"Erro ao processar {tipo}: {e}")
        return None


def executar_analise(tipo: str, arquivo_csv: str):
    """Executa análise de etarismo nos dados processados"""
    logger.info(f"\n📊 Executando análise de etarismo para {tipo.upper()}")
    
    try:
        analisador = AnalisadorEtarismo(f'resultados/{tipo}')
        analisador.executar_analise_completa(arquivo_csv)
        logger.info(f"✅ Análise {tipo.upper()} concluída!")
    except Exception as e:
        logger.error(f"Erro na análise {tipo}: {e}")


def processar_todas_bases():
    """Processa todas as bases sequencialmente"""
    logger.info("🚀 Processamento completo de todas as bases")
    
    bases = ['pnad', 'rais', 'caged']
    resultados = {}
    
    for base in bases:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processando {base.upper()}")
        logger.info(f"{'='*60}")
        
        arquivo = processar_base(base)
        resultados[base] = arquivo
    
    # Resumo final
    logger.info(f"\n{'='*60}")
    logger.info("RESUMO DO PROCESSAMENTO")
    logger.info(f"{'='*60}")
    
    for base, arquivo in resultados.items():
        status = "✅ Sucesso" if arquivo else "❌ Falha"
        logger.info(f"{base.upper()}: {status}")
        if arquivo:
            logger.info(f"   Arquivo: {arquivo}")


def main():
    """Função principal com argumentos de linha de comando"""
    parser = argparse.ArgumentParser(
        description='Processamento de dados para análise de etarismo em TI'
    )
    
    parser.add_argument(
        'base',
        nargs='?',
        choices=['pnad', 'rais', 'caged', 'todas'],
        default='todas',
        help='Base de dados a processar (padrão: todas)'
    )
    
    parser.add_argument(
        '--analise',
        action='store_true',
        help='Executar análise de etarismo após processamento'
    )
    
    parser.add_argument(
        '--log',
        default='processamento.log',
        help='Arquivo de log (padrão: processamento.log)'
    )
    
    args = parser.parse_args()
    
    # Configurar log
    configurar_log(args.log)
    
    # Executar processamento
    if args.base == 'todas':
        processar_todas_bases()
    else:
        processar_base(args.base)


if __name__ == '__main__':
    main()
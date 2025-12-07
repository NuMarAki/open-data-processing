# Open Data Processing

Sistema de processamento e análise de dados do IBGE (PNAD, RAIS, CAGED) com modelo preditivo.

##  Inicio Rápido

\\\ash
python app_main.py
\\\

##  Estrutura

- **app_main.py** - Ponto de entrada principal
- **config/** -  Arquivos de configuração (.cfg)
  - pnad.cfg - Configuração PNAD
  - rais.cfg - Configuração RAIS  
  - caged.cfg - Configuração CAGED
- **menu/** - Classes de menu
- **modules/** - Módulos funcionais
- **preditivo_rais/** - Modelo preditivo

##  Configuração

Todos os parâmetros estão centralizados em **config/*.cfg**:

- Períodos de análise (ano_inicio, ano_fim)
- Caminhos de dados e resultados
- Parâmetros de processamento (paralelo, workers, batch)
- Limites de memória
- Códigos CBO para TI
- Faixas etárias

##  Funcionalidades

- Processamento: PNAD, RAIS, CAGED
- Análises e diagnósticos
- 4 tipos de gráficos PNAD
- Modelo preditivo (treinar/prever)
- Descompactação automática

##  Requisitos

- Python 3.8+

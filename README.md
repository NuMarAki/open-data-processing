# Open Data Processing

Ferramenta para operar três bases públicas (PNAD, RAIS e CAGED) em um só lugar: descompactar, processar, inspecionar, gerar gráficos e treinar modelos preditivos.

## O que o projeto faz

- **Descompactação**: descompacta os arquivos brutos e organiza em pastas de trabalho.
- **Processamento**: normaliza e consolida PNAD, RAIS e CAGED para uso em análises e modelos.
- **Diagnóstico**: verifica cobertura de colunas, tamanho dos dados e possíveis faltas/inconsistências.
- **Gráficos PNAD**: escolaridade x sexo, renda x estudo, comparativo por UF e série temporal completa.
- **Modelos preditivos**:
	- RAIS: prevê vínculo ativo em 31/12.
	- PNAD: classifica renda >= 6 salários mínimos (ajustado por ano), com pesos amostrais.

## Como rodar

1) Pré-requisitos: Python 3.8+ e dados nas pastas esperadas (dados/pnad, dados/rais, dados/caged).
2) Execute o menu interativo:

```bash
python app_main.py
```

No menu você encontra:
- Descompactação das três bases.
- Processamento completo por base ou todas.
- Relatórios e gráficos PNAD.
- Módulo preditivo: RAIS (opções 1–3) e PNAD 6 SM (opções 10, 20, 21, 30). Use a opção 21 para modo rápido (amostra reduzida).

## Organização do projeto

- app_main.py — ponto de entrada com menus.
- config/*.cfg — caminhos e parâmetros de cada base.
- menu/ — navegação de console.
- modules/ — processadores, relatórios, diagnósticos, preditivo.
- preditivo_rais/ — pacote do modelo RAIS (artefatos e código).

## Configuração (CFG)

Arquivos em `config/` controlam caminhos e parâmetros:

- `pnad.cfg`: pastas `dados/pnad/raw` e `dados/pnad/preprocessados`, anos de início/fim, separador (`;`), chunk/tamanho de lote.
- `rais.cfg`: pastas `dados/rais/raw` e `dados/rais/preprocessados`, limites de memória/arquivos, mapeamento de colunas e CBO.
- `caged.cfg`: pastas de entrada/saída do CAGED e opções de parsing.

Para mudar onde estão os compactados (brutos), edite a seção de caminhos no cfg correspondente, por exemplo em `pnad.cfg`:

```
[paths]
raw_dir = D:/meus_dados/pnad/zip     ; onde ficam os .zip/.7z/.rar
out_dir = D:/meus_dados/pnad/        ; saída dos descompactados e preprocessados
```

Depois rode o menu normalmente; os módulos de descompactação e processamento usarão os novos caminhos.

## Notas rápidas

- Saídas ficam em resultados/<base>/...
- PNAD preprocessados esperados em dados/pnad/preprocessados.
- Para reduzir tempo no PNAD preditivo, use o modo rápido (opção 21) ou passe `--sample-frac` ao script.

## Requisitos

- Python 3.8 ou superior.

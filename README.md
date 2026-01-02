# Open Data Processing

Ferramenta para operar três bases públicas (PNAD, RAIS e CAGED) em um só lugar: descompactar, processar, inspecionar, gerar gráficos e treinar modelos preditivos.

## O que o projeto faz

- **Descompactação**: descompacta os arquivos brutos e organiza em pastas de trabalho.
- **Processamento**: normaliza e consolida PNAD, RAIS e CAGED para uso em análises e modelos.
- **Diagnóstico**: verifica cobertura de colunas, tamanho dos dados e possíveis faltas/inconsistências.
- **Gráficos PNAD**: escolaridade x sexo, renda x estudo, comparativo por UF e série temporal completa.
- **Modelos preditivos**:
	- RAIS: prevê vínculo ativo em 31/12 usando Random Forest com validação hold-out (75/25).
	- PNAD: classifica renda >= 6 salários mínimos (ajustado por ano), com pesos amostrais, usando Regressão Logística com validação cruzada estratificada (StratifiedKFold, k=5).

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

## Validação dos Modelos Preditivos

### PNAD - Classificação de Renda >= 6 SM

**Variável Target**: `salario_6sm` (binária)
- **1**: Renda >= 6 salários mínimos (ajustado por ano)
- **0**: Renda < 6 salários mínimos
- **Construção**: `renda >= 6 * salario_minimo_ano` onde `salario_minimo_ano` é mapeado de tabela histórica

**Método**: Validação cruzada estratificada (Stratified K-Fold Cross-Validation)
- **Algoritmo**: Regressão Logística
- **Estratégia**: StratifiedKFold com k=5 folds
- **Métrica principal**: ROC-AUC
- **Configuração**: `shuffle=True, random_state=42`
- **Implementação**: Ver [preditivos/preditivo_pnad.py](preditivos/preditivo_pnad.py#L112-L116)

**Divisão treino/teste (avaliação hold-out do relatório)**
- **Proporção**: 70% treino / 30% teste (`test_size=0.3`)
- **Critério**: divisão aleatória dos registros, sem sobreposição entre treino e teste
- **Estratificação**: `stratify=y` para preservar a proporção das classes em ambos os conjuntos
- **Reprodutibilidade**: `random_state=42`
- **Implementação**: Ver [preditivos/preditivo_pnad.py](preditivos/preditivo_pnad.py#L77-L79)

**Hiperparâmetros e justificativas (PNAD)**
- **`class_weight="balanced"`**: compensa desbalanceamento entre as classes (renda < 6 SM vs. >= 6 SM), reduzindo viés do classificador para a classe majoritária.
- **`max_iter=500`**: aumenta o limite de iterações para garantir convergência do otimizador, especialmente após padronização e com possível colinearidade.
- **`random_state=42`**: garante reprodutibilidade nos resultados (split e treinamento).
- **Padronização (`StandardScaler`)**: coloca variáveis numéricas na mesma escala, estabilizando a estimação dos coeficientes na regressão logística.

**Correção de desbalanceamento (PNAD)**
- No fluxo principal de modelagem, o desbalanceamento é tratado por **ponderação no algoritmo** (`class_weight="balanced"`) e por **estratificação** no split (`stratify=y`), sem reamostragem explícita.
- Existe também um fluxo alternativo com pesos amostrais (`sample_weight`) no script [preditivos/preditivo_pnad_alvo.py](preditivos/preditivo_pnad_alvo.py) (quando aplicável), que repassa os pesos para o estimador (`clf__sample_weight`).

A validação cruzada estratificada foi escolhida para:
1. Garantir que cada fold tenha proporção balanceada das classes (renda < 6 SM e >= 6 SM)
2. Reduzir viés da divisão única de treino/teste
3. Fornecer estimativa robusta da performance (média ± desvio padrão)

### RAIS - Predição de Vínculo Ativo 31/12

**Variável Target**: `vinculo_ativo_3112` (binária)
- **1**: Trabalhador com vínculo ativo em 31 de dezembro do ano de referência
- **0**: Vínculo encerrado antes de 31/12 (demitido, aposentado, afastado definitivamente)
- **Variações aceitas**: `vinculo_ativo_31_12`, `ind_vinculo_ativo_3112`, `ativo_3112`

**Método**: Validação hold-out (train-test split)
- **Algoritmo**: Random Forest (500 árvores)
- **Divisão**: 75% treino / 25% teste
- **Estratificação**: Baseada na variável alvo
- **Configuração**: `test_size=0.25, random_state=42, stratify=y`
- **Implementação**: Ver [preditivos/preditivo_rais.py](preditivos/preditivo_rais.py#L383-L385)

**Hiperparâmetros e justificativas (RAIS)**
- **`n_estimators=500`**: melhora a estabilidade da predição por média de muitas árvores (reduz variância), mantendo custo computacional aceitável.
- **`max_depth=None`**: permite capturar não-linearidades e interações entre atributos (ex.: idade × ocupação × setor).
- **`min_samples_leaf=5`**: regulariza as árvores ao impor um tamanho mínimo de folha, reduzindo overfitting em padrões raros.
- **`class_weight="balanced"`**: lida com desbalanceamento do alvo (ativos vs. inativos), equilibrando o custo de erro entre classes.
- **`n_jobs=-1`**: paraleliza o treino para viabilizar o processamento em bases grandes.
- **`random_state=42`**: garante reprodutibilidade (amostragem opcional, split e modelo).

Observação: a estratificação é aplicada quando existem as duas classes no conjunto (`y.nunique()>1`); caso contrário, o script usa `stratify=None` para evitar erro de split.

A validação hold-out foi adotada devido ao:
1. Alto volume de dados da RAIS (milhões de registros)
2. Custo computacional elevado para k-fold cross-validation com Random Forest
3. Representatividade suficiente com 25% dos dados para teste

**Métricas reportadas**: Acurácia, Precisão, Recall, F1-Score, ROC-AUC, Average Precision

## Relação com Empregabilidade e Etarismo

### Como as variáveis target capturam fenômenos de interesse

#### PNAD - Renda >= 6 SM como indicador de empregabilidade qualificada

A variável `salario_6sm` representa um limiar de **empregabilidade qualificada** no mercado de trabalho brasileiro:

**Indicador de posicionamento no mercado**:
- 6 salários mínimos (~R$ 8.472 em 2024) representa aproximadamente 3× a renda média brasileira
- Identifica trabalhadores em posições técnicas/gerenciais de maior qualificação
- Reflete o retorno econômico de investimentos em educação e experiência

**Relação com etarismo**:
- Permite analisar se a **idade** (feature do modelo) impacta positiva ou negativamente a probabilidade de alcançar essa faixa salarial
- Coeficientes do modelo logístico revelam se trabalhadores mais velhos têm menor probabilidade de renda alta (evidência de discriminação etária)
- Controle de variáveis confundidoras (escolaridade, sexo, região) isola o efeito da idade sobre renda

**Exemplo de análise**: Se o coeficiente de `idade` for negativo e significativo, indica que, controlando educação e outros fatores, trabalhadores mais velhos têm menor probabilidade de alta remuneração — evidência potencial de etarismo no mercado.

#### RAIS - Vínculo Ativo 31/12 como indicador de estabilidade e permanência

A variável `vinculo_ativo_3112` representa **estabilidade e permanência no emprego formal**:

**Indicador de continuidade no mercado formal**:
- Captura se o trabalhador manteve seu emprego ao longo do ano inteiro
- Distingue entre trabalhadores estáveis (vínculo ativo) e desligados (demissão, aposentadoria forçada, fim de contrato)
- Reflete vulnerabilidade a rupturas de vínculo empregatício

**Relação com etarismo**:
- Permite identificar se trabalhadores mais velhos têm maior probabilidade de **desligamento** ao longo do ano
- Features como `idade`, `tempo_emprego`, `cnae_classe` e `vl_remun_media` ajudam a explicar padrões de permanência
- Importâncias de permutação revelam se idade é fator determinante para perda de vínculo (indicativo de discriminação etária estrutural)

**Exemplo de análise**: Se a feature `idade` tiver alta importância na predição de vínculo **inativo**, e trabalhadores acima de 50 anos sistematicamente apresentarem maior risco de desligamento (controlando setor, salário e função), isso sugere **etarismo sistêmico** no mercado formal.

### Por que essas métricas são relevantes para o estudo de etarismo

1. **Mensuração objetiva**: Usam dados oficiais (IBGE e MTE) com milhões de registros, evitando viés de autorrelato
2. **Controle estatístico**: Modelos multivariados isolam o efeito da idade de outros fatores socioeconômicos
3. **Dimensões complementares**: 
   - PNAD → retorno econômico e valorização pelo mercado
   - RAIS → estabilidade e risco de exclusão do mercado formal
4. **Evidência quantitativa**: Coeficientes e importâncias fornecem magnitude do impacto da idade, não apenas correlações

## Notas rápidas

- Saídas ficam em resultados/<base>/...
- PNAD preprocessados esperados em dados/pnad/preprocessados.
- Para reduzir tempo no PNAD preditivo, use o modo rápido (opção 21) ou passe `--sample-frac` ao script.

## Requisitos

- Python 3.8 ou superior.

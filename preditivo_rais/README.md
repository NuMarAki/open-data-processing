# Preditivo RAIS (CPU)

Versão incorporada ao `open-data-processing` do pipeline `preditivo_rais_radeon`, agora sem dependências ou uso de GPU. Mantém a lógica de carregamento, pré-processamento e treinamento de Random Forest, operando somente com scikit-learn/CPU.

## Estrutura
```
preditivo_rais_radeon/
├── config/params.yaml
├── requirements.txt
├── environment.yml
├── setup.py
└── src/
    ├── main.py
    ├── data/
    │   ├── loader.py
    │   ├── preprocessing.py
    │   └── feature_processing.py
    ├── models/random_forest.py
    └── utils/gpu_utils.py  # stubs (GPU removido)
```

## Uso
```bash
# criar env
conda env create -f environment.yml
conda activate preditivo_rais_cpu

# executar treinamento/avaliação
python -m src.main
```

Parâmetros em `config/params.yaml` (caminhos absolutos de dados/artefatos mantidos). Nenhum recurso GPU é utilizado.

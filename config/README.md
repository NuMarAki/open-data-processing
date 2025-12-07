# Configurações das Bases de Dados

Esta pasta contém os arquivos de configuração centralizados para cada base de dados.

## 📋 Arquivos

- **pnad.cfg** - PNAD Contínua (Pesquisa Nacional por Amostra de Domicílios)
- **rais.cfg** - RAIS (Relação Anual de Informações Sociais)
- **caged.cfg** - Novo CAGED (Cadastro Geral de Empregados e Desempregados)

## 🔧 Estrutura dos Arquivos

Cada arquivo `.cfg` contém as seguintes seções:

### [geral]
Informações básicas da base (nome, tipo, versão)

### [periodo_analise]
Anos/períodos a serem processados

### [caminhos]
- Localização dos arquivos compactados
- Diretórios de descompactação
- Diretórios de resultados
- Padrões de nome dos arquivos

### [parametros_processamento]
- Processamento paralelo (sim/não)
- Número de workers
- Tamanho de lote
- Amostragem de dados
- Força processamento sequencial

### [parametros_leitura]
- Delimitador de colunas
- Encoding
- Chunking para arquivos grandes
- Tamanho do chunk

### [colunas_*] ou [layout]
Mapeamento de colunas (nome no arquivo → nome padronizado)

### [faixas_etarias]
Definição das faixas etárias para análise

### [cbo_ti]
Códigos CBO que identificam profissionais de TI

### [filtros_analise]
Filtros automáticos aplicados durante o processamento

### [memoria]
Limites e controles de memória

## 💡 Como Usar

Os arquivos são carregados automaticamente pelo `ConfigManager`:

```python
from config_manager import config_manager

# Carregar configuração
config = config_manager.carregar_configuracao('pnad', 'config/pnad.cfg')

# Acessar parâmetros
print(config.ano_inicio)
print(config.ano_fim)
print(config.usar_paralelo)
```

## ⚙️ Personalização

Para ajustar o processamento, edite os valores nos arquivos `.cfg`:

1. **Período**: Ajuste `ano_inicio` e `ano_fim`
2. **Performance**: Ajuste `max_workers` e `batch_size`
3. **Memória**: Ajuste `limite_memoria_mb` e `forcar_limpeza_memoria`
4. **Caminhos**: Ajuste os diretórios conforme sua estrutura

## 🎯 Campos Importantes

### Processamento
- `usar_paralelo` - Ativar processamento paralelo
- `max_workers` - Número de workers (0 = automático)
- `batch_size` - Arquivos processados por vez
- `forcar_sequencial` - Forçar sequencial (RAIS)

### Memória
- `limite_memoria_mb` - Limite de memória em MB
- `forcar_limpeza_memoria` - Limpar após cada arquivo
- `percentual_max_memoria` - % máximo de uso

### Dados
- `amostra_registros` - Amostra para testes (0 = todos)
- `usar_chunking` - Processar em chunks
- `tamanho_chunk` - Linhas por chunk

## 📍 Localização

Todos os arquivos `.cfg` devem estar em:
```
open-data-processing/config/
```

O sistema busca automaticamente nesta pasta.

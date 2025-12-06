import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# ------------------------
# Configuração do ambiente
# ------------------------
def configurar_ambiente():
    pasta = Path("graficos")
    pasta.mkdir(exist_ok=True)
    print("Diretório 'graficos' pronto.")
    return pasta

# ------------------------
# Carregamento dos dados
# ------------------------
def carregar_e_preparar_dados():
    """
    Carrega os dados da PNAD e prepara as variáveis para regressão logística.
    """
    # Primeiro tenta carregar base consolidada, senão usa amostra
    try:
        df = pd.read_csv(Path(r"C:\TCC\dados\pnad\dados_pnad_consolidados.csv"), sep=";")
        print("Dados completos carregados com sucesso.")
    except FileNotFoundError:
        print("Arquivo principal não encontrado. Carregando 'amostra_pnad.csv'.")
        df = pd.read_csv("amostra_pnad.csv", sep=";")

    # Salário mínimo: carregue de um CSV/JSON oficial (ajustar caminho)
    try:
        salarios_minimos = pd.read_csv("salarios_minimos.csv")  # colunas: ano, salario
        mapa_salarios = dict(zip(salarios_minimos["ano"], salarios_minimos["salario"]))
    except FileNotFoundError:
        print("⚠️ Arquivo de salários mínimos não encontrado. Usando fallback limitado.")
        mapa_salarios = {2012: 622, 2018: 954, 2024: 1412}

    df["salario_minimo_ano"] = df["ano"].map(mapa_salarios)

    # Conversão de tipos numéricos
    for col in ["rendimento_bruto_mensal", "anos_estudo", "idade", "sexo"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["salario_minimo_ano"], inplace=True)

    # Variável alvo
    df["salario_alto"] = (df["rendimento_bruto_mensal"] > 10 * df["salario_minimo_ano"]).astype(int)

    # Seleção de variáveis
    features = ["anos_estudo", "idade", "sexo"]
    target = "salario_alto"
    df_modelo = df[features + [target]].copy()
    df_modelo.dropna(inplace=True)

    # Dummies para sexo (1=homem, 2=mulher)
    df_modelo = pd.get_dummies(df_modelo, columns=["sexo"], drop_first=True, prefix="sexo")
    df_modelo.rename(columns={"sexo_2.0": "sexo_mulher"}, inplace=True)

    X = df_modelo.drop(target, axis=1)
    y = df_modelo[target]

    # Normalização
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[["anos_estudo", "idade"]] = scaler.fit_transform(X_scaled[["anos_estudo", "idade"]])

    return X_scaled, y, scaler

# ------------------------
# Treinamento e avaliação
# ------------------------
def treinar_avaliar_modelo(X, y, pasta):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    modelo = LogisticRegression(random_state=42, class_weight="balanced", max_iter=500)
    modelo.fit(X_train, y_train)

    # Predição
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    # Avaliação básica
    print("\n--- AVALIAÇÃO NO TESTE ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
    print(f"F1-score: {f1_score(y_test, y_pred):.2%}")
    print(f"Precisão: {precision_score(y_test, y_pred):.2%}")
    print(f"Recall:   {recall_score(y_test, y_pred):.2%}")
    print(f"ROC AUC:  {roc_auc_score(y_test, y_prob):.2%}")
    print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(pasta / "matriz_confusao.png", dpi=300)

    # Curva ROC
    RocCurveDisplay.from_estimator(modelo, X_test, y_test)
    plt.title("Curva ROC")
    plt.savefig(pasta / "curva_roc.png", dpi=300)

    # Cross-validation
    print("\n--- CROSS-VALIDATION ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(modelo, X, y, cv=cv, scoring="roc_auc")
    print(f"ROC AUC médio (CV=5): {scores.mean():.2%} ± {scores.std():.2%}")

    return modelo

# ------------------------
# Interpretação
# ------------------------
def interpretar_modelo(modelo, X):
    print("\n--- INTERPRETAÇÃO DO MODELO ---")
    coefs = pd.Series(modelo.coef_[0], index=X.columns)
    coefs_ordenados = coefs.sort_values(ascending=False)
    print(coefs_ordenados)

    plt.figure(figsize=(8, 5))
    coefs_ordenados.plot(kind="barh", color="darkgreen")
    plt.title("Impacto das Variáveis na Regressão Logística")
    plt.xlabel("Coeficiente (log-odds)")
    plt.tight_layout()
    plt.savefig("graficos/impacto_variaveis.png", dpi=300)

# ------------------------
# Main
# ------------------------
def main():
    pasta = configurar_ambiente()
    X, y, scaler = carregar_e_preparar_dados()

    if X.empty:
        print("Não há dados suficientes para treinar o modelo após a limpeza.")
        return

    modelo = treinar_avaliar_modelo(X, y, pasta)
    interpretar_modelo(modelo, X)

if __name__ == "__main__":
    main()

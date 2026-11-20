import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

os.makedirs("results/figures", exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Carrega dataset tratado na etapa 1
df = pd.read_csv("data/processed/olympics_clean.csv")

X = df[["Sex", "Age", "Height", "Weight", "Sport"]]
y = df["Target"]

# Divisão estratificada (80% treino, 20% teste) mantendo a proporção de 14.64% medalhistas
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pré-processamento: Padronização nos dados biométricos + OneHot nas categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Age", "Height", "Weight"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Sex", "Sport"]),
    ]
)

# Definição dos 3 modelos do projeto (class_weight='balanced' lida com o desbalanceamento)
modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Árvore de Decisão": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42),
    "SVM (Linear)": LinearSVC(class_weight="balanced", random_state=42, max_iter=2000),
}

resultados = []
matrizes = {}
pipe_logistica = None

print("Treinando e avaliando os modelos...\n")

for nome, clf in modelos.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    if nome == "Regressão Logística":
        pipe_logistica = pipe

    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    resultados.append({
        "Modelo": nome,
        "Precisão": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
    })
    matrizes[nome] = confusion_matrix(y_test, y_pred)

df_resultados = pd.DataFrame(resultados)

# 2. Ajuste de Limiar (Threshold) na Regressão Logística
y_probs = pipe_logistica.predict_proba(X_test)[:, 1]
limiares = [0.35, 0.45, 0.50, 0.55, 0.65]
resultados_limiar = []

for thresh in limiares:
    y_pred_thresh = (y_probs >= thresh).astype(int)
    resultados_limiar.append({
        "Limiar (Threshold)": thresh,
        "Precisão": round(precision_score(y_test, y_pred_thresh, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred_thresh), 4),
        "F1-Score": round(f1_score(y_test, y_pred_thresh), 4),
    })

df_limiares = pd.DataFrame(resultados_limiar)

# 3. Gráfico 4: Comparação das Métricas entre os 3 Modelos
df_melt = df_resultados.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")
plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df_melt, x="Modelo", y="Valor", hue="Métrica", palette="viridis")
plt.title("Comparação de Desempenho dos Modelos (Precisão, Recall e F1-Score)")
plt.ylim(0, 0.85)
plt.ylabel("Pontuação")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/04_comparacao_modelos.png", dpi=300)
plt.close()

# 4. Gráfico 5: Matrizes de Confusão lado a lado
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (nome, cm) in zip(axes, matrizes.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(nome)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_xticklabels(["Sem (0)", "Com (1)"])
    ax.set_yticklabels(["Sem (0)", "Com (1)"])

plt.tight_layout()
plt.savefig("results/figures/05_matrizes_confusao.png", dpi=300)
plt.close()

# 5. Exporta resultados em texto/Markdown para a API da NVIDIA gerar o relatório
resumo_modelos = f"""=== RESULTADOS DA MODELAGEM (CONJUNTO DE TESTE: {len(y_test)} REGISTROS) ===

1. Comparação Geral entre os Modelos (Limiar padrão):
{df_resultados.to_markdown(index=False)}

2. Ajuste de Limiar de Decisão (Regressão Logística):
{df_limiares.to_markdown(index=False)}
"""

with open("results/resumo_modelos.txt", "w", encoding="utf-8") as f:
    f.write(resumo_modelos)

print(resumo_modelos)
print("Gráficos salvos em: results/figures/04_comparacao_modelos.png e 05_matrizes_confusao.png")
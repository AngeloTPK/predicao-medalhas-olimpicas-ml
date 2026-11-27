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
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

os.makedirs("results/figures", exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Carrega dataset tratado na etapa 1
df = pd.read_csv("data/processed/olympics_clean.csv")

X = df[["Sex", "Age", "Height", "Weight", "Sport"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Age", "Height", "Weight"]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Sex", "Sport"]),
    ]
)

modelos = {
    "Regressão Logística": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Árvore de Decisão": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=42),
    "SVM (Linear)": LinearSVC(class_weight="balanced", random_state=42, max_iter=2000),
}

resultados = []
matrizes = {}
pipes_treinados = {}

print("Treinando e avaliando os modelos...\n")

for nome, clf in modelos.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    pipes_treinados[nome] = pipe

    resultados.append({
        "Modelo": nome,
        "Precisão": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
    })
    matrizes[nome] = confusion_matrix(y_test, y_pred)

df_resultados = pd.DataFrame(resultados)

# 2. Ajuste de Limiar (Threshold) na Regressão Logística
pipe_logistica = pipes_treinados["Regressão Logística"]
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

# 3. NOVO: Simulação de Fichas de Atletas (Estudo de Caso)
fichas_teste = pd.DataFrame([
    {"Perfil": "Atleta A (Basquete - Biotipo Alto)", "Sex": "M", "Age": 26, "Height": 205, "Weight": 102, "Sport": "Basketball"},
    {"Perfil": "Atleta B (Basquete - Baixa Estatura)", "Sex": "M", "Age": 26, "Height": 168, "Weight": 68, "Sport": "Basketball"},
    {"Perfil": "Atleta C (Ginástica - Biotipo Leve)", "Sex": "F", "Age": 19, "Height": 154, "Weight": 46, "Sport": "Gymnastics"},
    {"Perfil": "Atleta D (Ginástica - Alta/Pesada)", "Sex": "F", "Age": 19, "Height": 182, "Weight": 78, "Sport": "Gymnastics"},
    {"Perfil": "Atleta E (Natação - Envergadura/Alto)", "Sex": "M", "Age": 23, "Height": 196, "Weight": 88, "Sport": "Swimming"},
    {"Perfil": "Atleta F (Halterofilismo - Força/Peso)", "Sex": "M", "Age": 27, "Height": 175, "Weight": 115, "Sport": "Weightlifting"},
])

probs_fichas = pipe_logistica.predict_proba(fichas_teste[["Sex", "Age", "Height", "Weight", "Sport"]])[:, 1]
fichas_teste["Prob. Medalha (%)"] = (probs_fichas * 100).round(2)
fichas_teste["Classificação (Limiar 0.50)"] = np.where(probs_fichas >= 0.50, "Medalhista (1)", "Não Medalhista (0)")

# 4. NOVO: Importância das Variáveis (Árvore de Decisão)
pipe_arvore = pipes_treinados["Árvore de Decisão"]
nomes_features = pipe_arvore.named_steps["prep"].get_feature_names_out()
importancias = pipe_arvore.named_steps["model"].feature_importances_

df_imp = pd.DataFrame({"Atributo": nomes_features, "Importância": importancias})
df_imp["Atributo"] = df_imp["Atributo"].str.replace("num__", "").str.replace("cat__", "")
df_imp_top10 = df_imp.sort_values(by="Importância", ascending=False).head(10)

plt.figure(figsize=(8, 4.5))
sns.barplot(data=df_imp_top10, x="Importância", y="Atributo", palette="magma")
plt.title("Top 10 Atributos Mais Relevantes (Árvore de Decisão)")
plt.xlabel("Importância Relativa (Gini)")
plt.ylabel("Atributo")
plt.tight_layout()
plt.savefig("results/figures/06_importancia_variaveis.png", dpi=300)
plt.close()

# Gráficos anteriores (04 e 05)
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

# 5. Exporta tudo para o arquivo de resumo
resumo_modelos = f"""=== RESULTADOS DA MODELAGEM (CONJUNTO DE TESTE: {len(y_test)} REGISTROS) ===

1. Comparação Geral entre os Modelos (Limiar padrão):
{df_resultados.to_markdown(index=False)}

2. Ajuste de Limiar de Decisão (Regressão Logística):
{df_limiares.to_markdown(index=False)}

3. Simulação de Fichas de Atletas (Regressão Logística):
{fichas_teste.to_markdown(index=False)}

4. Top 10 Atributos Mais Importantes (Árvore de Decisão):
{df_imp_top10.to_markdown(index=False)}
"""

with open("results/resumo_modelos.txt", "w", encoding="utf-8") as f:
    f.write(resumo_modelos)

print(resumo_modelos)
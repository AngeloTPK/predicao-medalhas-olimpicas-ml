import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Garante que as pastas de saída existam
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# Configuração visual dos gráficos
sns.set_theme(style="whitegrid")

# 1. Carregamento dos dados brutos
caminho_csv = "data/raw/athlete_events.csv"
print(f"Carregando {caminho_csv}...")
df = pd.read_csv(caminho_csv)

total_bruto = len(df)
nulos_antes = df[["Age", "Height", "Weight"]].isna().sum().to_dict()

# 2. Criação da variável-alvo binária (1 = Conquistou medalha, 0 = Não conquistou)
df["Target"] = df["Medal"].notna().astype(int)

# 3. Seleção das colunas relevantes para análise biométrica e temporal
colunas = ["Sex", "Age", "Height", "Weight", "Sport", "Year", "Season", "Target"]
df_clean = df[colunas].copy()

# 4. Tratamento de valores ausentes (NaN) nos atributos biométricos
# Imputação pela mediana agrupada por Modalidade (Sport) e Sexo (Sex)
for col in ["Age", "Height", "Weight"]:
    df_clean[col] = df_clean.groupby(["Sport", "Sex"])[col].transform(
        lambda x: x.fillna(x.median())
    )

# Remove registros residuais de modalidades que nunca tiveram medição biométrica
df_clean.dropna(subset=["Age", "Height", "Weight"], inplace=True)

total_limpo = len(df_clean)

# Salva o dataset tratado para a etapa de modelagem
df_clean.to_csv("data/processed/olympics_clean.csv", index=False)
print("Dataset limpo salvo em: data/processed/olympics_clean.csv")

# 5. Geração de Gráficos para o Artigo Científico

# Gráfico 1: Desbalanceamento de Classes (Medalhistas vs Não Medalhistas)
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df_clean, x="Target", palette=["#7f8c8d", "#f1c40f"])
ax.set_xticklabels(["Sem Medalha (0)", "Com Medalha (1)"])
plt.title("Distribuição da Variável-Alvo (Target)")
plt.xlabel("Resultado Competitivo")
plt.ylabel("Quantidade de Registros")
for container in ax.containers:
    ax.bar_label(container, fmt="%d")
plt.tight_layout()
plt.savefig("results/figures/01_distribuicao_target.png", dpi=300)
plt.close()

# Gráfico 2: Comparação de Altura e Peso em modalidades contrastantes
esportes_destaque = ["Basketball", "Gymnastics", "Weightlifting", "Swimming", "Athletics"]
df_destaque = df_clean[df_clean["Sport"].isin(esportes_destaque)]

plt.figure(figsize=(9, 5))
sns.boxplot(data=df_destaque, x="Sport", y="Height", hue="Target", palette=["#95a5a6", "#2ecc71"])
plt.title("Distribuição de Altura (cm) por Modalidade e Conquista de Medalha")
plt.xlabel("Modalidade Esportiva")
plt.ylabel("Altura (cm)")
plt.legend(title="Medalha", labels=["Não (0)", "Sim (1)"])
plt.tight_layout()
plt.savefig("results/figures/02_altura_modalidades.png", dpi=300)
plt.close()

# Gráfico 3: Evolução da Altura e Peso médios dos medalhistas ao longo das décadas
df_medalhistas = df_clean[df_clean["Target"] == 1].copy()
df_medalhistas["Decada"] = (df_medalhistas["Year"] // 10) * 10

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.lineplot(data=df_medalhistas, x="Decada", y="Height", hue="Sex", marker="o", ax=axes[0])
axes[0].set_title("Evolução da Altura Média dos Medalhistas")
axes[0].set_xlabel("Década")
axes[0].set_ylabel("Altura (cm)")

sns.lineplot(data=df_medalhistas, x="Decada", y="Weight", hue="Sex", marker="o", ax=axes[1])
axes[1].set_title("Evolução do Peso Médio dos Medalhistas")
axes[1].set_xlabel("Década")
axes[1].set_ylabel("Peso (kg)")

plt.tight_layout()
plt.savefig("results/figures/03_evolucao_biometria.png", dpi=300)
plt.close()

# 6. Exporta resumo numérico para alimentar o script de relatório da IA
dist_target = df_clean["Target"].value_counts(normalize=True) * 100
medias_grupo = df_clean.groupby("Target")[["Age", "Height", "Weight"]].mean().round(2)

resumo = f"""=== RESUMO DA ANÁLISE EXPLORATÓRIA (EDA) ===
Registros originais: {total_bruto}
Valores nulos antes da imputação: {nulos_antes}
Registros após limpeza e imputação por (Sport, Sex): {total_limpo}
Modalidades distintas: {df_clean['Sport'].nunique()}

Distribuição da variável-alvo (Target):
- Sem Medalha (0): {dist_target[0]:.2f}%
- Com Medalha (1): {dist_target[1]:.2f}%

Médias biométricas por classe:
{medias_grupo.to_markdown()}
"""

with open("results/resumo_eda.txt", "w", encoding="utf-8") as f:
    f.write(resumo)

print("\n" + resumo)
print("Gráficos salvos em: results/figures/")
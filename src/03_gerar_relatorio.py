import os
from dotenv import load_dotenv
from openai import OpenAI

# Carrega NVIDIA_API_KEY e MODELO_AULA do arquivo .env
load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY"),
)

modelo = os.environ.get("MODELO_AULA", "nvidia/nemotron-3.5-lightning-30b-a3b")

# Lê os resultados reais gerados pelas etapas 1 e 2
with open("results/resumo_eda.txt", "r", encoding="utf-8") as f:
    resumo_eda = f.read()

with open("results/resumo_modelos.txt", "r", encoding="utf-8") as f:
    resumo_modelos = f.read()

prompt = f"""
Você é um pesquisador sênior de Ciência de Dados redigindo um artigo científico no padrão da Sociedade Brasileira de Computação (SBC).

Contexto do Projeto:
- Objetivo: Analisar como características físicas (peso, altura, idade e gênero) se correlacionam com a conquista de medalhas em diferentes modalidades olímpicas, classificando a probabilidade de sucesso competitivo de um atleta com base nesses atributos e na modalidade disputada.
- Base de dados: 120 years of Olympic history: athletes and results (Kaggle).
- Abordagem: Aprendizado Supervisionado — Classificação Binária (Target: 1 = Conquistou medalha, 0 = Não conquistou).
- Modelos avaliados: Regressão Logística (baseline com ajuste de limiar), Árvore de Decisão e SVM Linear.

Abaixo estão os resultados reais obtidos nos experimentos:

{resumo_eda}

{resumo_modelos}

Com base ESTRITAMENTE nos dados e métricas acima (não invente números), escreva em português formal acadêmico:
1. **Resumo (Abstract)** estruturado em um parágrafo objetivo.
2. **Metodologia e Pré-processamento**, explicando o tratamento de nulos por modalidade/sexo, o desbalanceamento natural da base (~14,6% medalhistas) e a escolha de Precisão, Recall e F1-Score em vez de acurácia simples.
3. **Resultados e Discussão**, comparando o desempenho dos três modelos, o impacto do ajuste de limiar na Regressão Logística e como os atributos biométricos variam conforme a modalidade esportiva.
4. **Conclusão e Limitações**, citando fatores externos à biometria (como esportes coletivos, país/investimento e técnica) e trabalhos futuros.
"""

print(f"Enviando métricas reais para o modelo {modelo}...")

completion = client.chat.completions.create(
    model=modelo,
    messages=[
        {
            "role": "system",
            "content": "Você redige artigos científicos técnicos, claros e fiéis aos dados experimentais fornecidos.",
        },
        {"role": "user", "content": prompt},
    ],
    temperature=0.2,
    max_tokens=4096,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False,
)

relatorio = completion.choices[0].message.content

os.makedirs("results", exist_ok=True)
caminho_saida = "results/rascunho_artigo_sbc.md"
with open(caminho_saida, "w", encoding="utf-8") as f:
    f.write(relatorio)

print(f"\nRelatório gerado com sucesso em: {caminho_saida}")
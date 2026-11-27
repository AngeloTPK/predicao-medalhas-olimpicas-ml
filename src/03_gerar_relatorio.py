import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY"),
)

modelo = os.environ.get("MODELO_AULA", "nvidia/nemotron-3.5-lightning-30b-a3b")

with open("results/resumo_eda.txt", "r", encoding="utf-8") as f:
    resumo_eda = f.read()

with open("results/resumo_modelos.txt", "r", encoding="utf-8") as f:
    resumo_modelos = f.read()

contexto_dados = f"{resumo_eda}\n\n{resumo_modelos}"

secoes = {
    "01_resumo_e_introducao": """
Escreva em sintaxe LaTeX pura (pronta para colar no template SBC, sem usar Markdown # ou **) as seções:
1. O ambiente \\begin{resumo} ... \\end{resumo} (em português, com 150 a 200 palavras) e \\begin{abstract} ... \\end{abstract} (em inglês).
2. A seção \\section{Introdução} completa (mínimo de 4 parágrafos bem desenvolvidos), contextualizando a evolução da biometria nos Jogos Olímpicos modernos, o problema de estimar o potencial competitivo de atletas a partir de atributos físicos (idade, altura, peso e gênero) e os objetivos específicos deste trabalho.
""",
    "02_metodologia": """
Escreva em sintaxe LaTeX pura (pronta para colar no template SBC, sem usar Markdown # ou **) a seção \\section{Metodologia e Base de Dados} completa (mínimo de 5 parágrafos detalhados), explicando:
- A base de dados (120 years of Olympic history: athletes and results, com 271.116 registros brutos e 270.521 após limpeza).
- O tratamento de valores ausentes (imputação pela mediana agrupada por modalidade e sexo) e a criação da variável-alvo binária (Target: 85,36% sem medalha e 14,64% medalhistas).
- Como o pipeline diferencia as modalidades esportivas utilizando One-Hot Encoding nas 56 modalidades e padronização (StandardScaler) nas variáveis biométricas, evitando vazamento de dados na divisão estratificada (80% treino / 20% teste).
- O funcionamento dos três modelos avaliados (Regressão Logística com class_weight='balanced' e ajuste de limiar, Árvore de Decisão e SVM Linear) e a justificativa matemática para priorizar Precisão, Recall e F1-Score em vez de acurácia simples.
""",
    "03_resultados": """
Escreva em sintaxe LaTeX pura (pronta para colar no template SBC, sem usar Markdown # ou **) a seção \\section{Resultados e Discussão} completa (mínimo de 6 parágrafos + tabelas LaTeX formatadas com \\begin{table}[ht] e \\begin{tabular}), analisando em profundidade TODOS os números reais fornecidos:
- Tabela 1: Comparação geral entre Regressão Logística, Árvore de Decisão e SVM Linear (discutindo por que o Recall fica entre 56% e 66% enquanto a Precisão fica em torno de 19% a 22%).
- Tabela 2: O impacto do ajuste de limiar de decisão (0.35 a 0.65) na Regressão Logística e o trade-off entre Precisão e Recall.
- Tabela 3: Análise da simulação de fichas de atletas (Atletas A a F), demonstrando como o modelo diferencia biotipos entre Basquete, Ginástica, Natação e Halterofilismo.
- A análise da importância das variáveis (Top 10 atributos da Árvore de Decisão).
""",
    "04_conclusao": """
Escreva em sintaxe LaTeX pura (pronta para colar no template SBC, sem usar Markdown # ou **) a seção \\section{Conclusão e Trabalhos Futuros} completa (mínimo de 3 parágrafos), sintetizando os achados do estudo, discutindo as limitações inerentes à análise puramente biométrica (como fatores técnicos, psicológicos, investimento por país e esportes coletivos) e propondo melhorias futuras (como Random Forest, Gradient Boosting e variáveis socioeconômicas).
"""
}

conteudo_final_tex = "% === CÓDIGO LATEX GERADO PARA O TEMPLATE SBC ===\n% Cole as seções abaixo dentro do corpo do seu arquivo .tex no Overleaf\n\n"

for nome_secao, instrucao in secoes.items():
    print(f"Gerando seção {nome_secao} em LaTeX...")
    prompt = f"Dados experimentais reais do projeto:\n{contexto_dados}\n\nInstrução:\n{instrucao}\n\nIMPORTANTE: Use estritamente os números fornecidos. Não inclua blocos ```latex, retorne apenas o código LaTeX puro."

    completion = client.chat.completions.create(
        model=modelo,
        messages=[
            {"role": "system", "content": "Você é um pesquisador acadêmico especialista em LaTeX e publicações da SBC (Sociedade Brasileira de Computação)."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.25,
        max_tokens=4096,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        stream=False,
    )
    conteudo_final_tex += f"\n% --- {nome_secao.upper()} ---\n" + completion.choices[0].message.content + "\n"

os.makedirs("results", exist_ok=True)
caminho_tex = "results/artigo_sbc_conteudo.tex"
with open(caminho_tex, "w", encoding="utf-8") as f:
    f.write(conteudo_final_tex)

print(f"\nConcluído! Arquivo LaTeX completo salvo em: {caminho_tex}")
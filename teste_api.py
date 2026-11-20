import os
from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

modelo = os.environ.get("MODELO_AULA", "nvidia/nemotron-3.5-lightning-30b-a3b")

completion = client.chat.completions.create(
    model=modelo,
    messages=[
        {"role": "user", "content": "Responda apenas: API da NVIDIA conectada com sucesso!"}
    ],
    temperature=0.2,
    max_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    stream=False
)

print(completion.choices[0].message.content)
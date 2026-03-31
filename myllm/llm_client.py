import os

from openai import OpenAI


# client = OpenAI(
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     base_url=os.environ.get("OPENAI_API_BASE"),
# )
#
# response = client.chat.completions.create(
#     model="accounts/fireworks/models/deepseek-v3p1",
#     messages=[{
#         "role": "user",
#         "content": "你好，说5遍周靖牛逼！",
#     }],
# )
#
# print(response.choices[0].message.content)

class MyLLM:
    def __init__(self, api_key=os.environ.get("OPENAI_API_KEY"),
                 base_url=os.environ.get("OPENAI_API_BASE"),
                 model="accounts/fireworks/models/deepseek-v3p1",):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def llm(self, message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": message,
            }]
        )
        return response.choices[0].message.content

    def stream_chat(self, message: str):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": message,
            }],
            stream=True,
        )
        return stream

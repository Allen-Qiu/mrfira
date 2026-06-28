"""
调用大语言模型生成答案
"""
from http import HTTPStatus
import dashscope
from openai import OpenAI

class QwenLLM():
    def __init__(self):
        dashscope.api_key = "XXX"
        self.model = dashscope.Generation.Models.qwen_max
        # self.model = "qwen_plus"

    def call_with_prompt(self, prompt):
        messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt}]

        response = dashscope.Generation.call(
            model = self.model,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
        )
        if response.status_code == HTTPStatus.OK:
            return (response.output["choices"][0]['message']['content'])
        else:
            return ('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))

    def call_with_messages(self, messages):

        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
            temperature = 0.1
        )
        if response.status_code == HTTPStatus.OK:
            return (response.output["choices"][0]['message']['content'])
        else:
            return ('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))

    def call_with_multiruns(self, messages):
        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_turbo,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
        )
        if response.status_code == HTTPStatus.OK:
            return (response.output["choices"][0]['message']['content'])
        else:
            return ('Request id: %s, Status code: %s, error code: %s, error message: %s \n请重新开始对话！' % (
                response.request_id, response.status_code,
                response.code, response.message
            ))


class DeepseekLLM():
    def __init__(self):
        self.key = "XXX"

    def call_with_prompt(self, prompt, temp=0.2):
        client = OpenAI(api_key=self.key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            temperature=temp,
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return (response.choices[0].message.content)

    # 多轮流式对话
    def call_with_messages(self, messages, temp=0.2):
        client = OpenAI(api_key=self.key, base_url="https://api.deepseek.com")
        try:
            response = client.chat.completions.create(
                temperature=temp,
                # top_p=topp,
                model="deepseek-chat",
                messages=messages,
                # stream=True
            )
            s = response.choices[0].message.content
            return s
        except Exception as e:
            return e

if __name__ == '__main__':
    llm = QwenLLM()
    query = "如何做西红柿炒鸡蛋？"
    llm.call_with_prompt(query)


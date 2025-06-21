import json
import time

import requests
from openai import OpenAI

client = OpenAI(api_key="sk-34a84406a6f14acdb01d7f6ebca893a4", base_url="https://api.deepseek.com")


def web_search(query):
    url = "https://api.bochaai.com/v1/web-search"

    payload = json.dumps({
        "query": query,
        "freshness": "oneWeek",
        "summary": True,
        "count": 20,
        "page": 1
    })

    headers = {
        'Authorization': 'Bearer sk-001972f22db8484b8cf25581640f8db6',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()


class DeepSeekChat:
    def __init__(self, model="deepseek-chat", max_tokens=4096, temperature=0.7):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.history = []
        self.api_url = "https://api.deepseek.com/v1/chat/completions"

        # 添加系统提示，让AI了解自己的角色
        self.add_system_message("你是一名炒股10年的炒股高手，非常善于量化选股")

    def add_system_message(self, content):
        """添加系统消息到对话历史"""
        self.history.append({"role": "system", "content": content})

    def chat(self, user_input):
        return self.chat_ai(user_input, 0)

    def chat_online(self, user_input):
        return self.chat_ai(user_input, 1)

    def chat_ai(self, user_input, is_web_search: False):
        """发送消息并获取AI回复"""
        # 联网查询
        if is_web_search:
            search_content = web_search(user_input)
            prompt = f"以下是从互联网上获取的最新信息：\n\n{search_content}\n\n请根据以上信息，回答用户的问题：\n用户：{user_input}"
        else:
            prompt = user_input

        # 添加用户消息到历史
        self.history.append({"role": "user", "content": prompt})

        try:
            # 请求API
            response = client.chat.completions.create(
                model=self.model,  # deepseek-reasoner
                messages=self.history,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            # AI答复
            ai_reply = response.choices[0].message.content

            # 添加AI回复到历史
            self.history.append({"role": "assistant", "content": ai_reply})
            return ai_reply

        except requests.exceptions.RequestException as e:
            return f"请求出错: {str(e)}"
        except KeyError:
            return "解析响应出错，请检查API返回格式"

    def reset_history(self):
        """重置对话历史但保留系统消息"""
        system_msg = self.history[0] if self.history and self.history[0]["role"] == "system" else None
        self.history = [system_msg] if system_msg else []

    def print_history(self):
        """打印完整对话历史"""
        print("\n" + "=" * 50 + "\n对话历史:")
        for msg in self.history:
            if msg["role"] == "system":
                continue  # 跳过系统消息
            role = "你" if msg["role"] == "user" else "AI"
            print(f"{role}: {msg['content']}")
        print("=" * 50 + "\n")


def main():
    # 创建聊天实例
    chat_bot = DeepSeekChat()

    print("=" * 50)
    print("DeepSeek 对话助手")
    print("输入你的问题开始对话")
    print("输入 '退出' 结束对话")

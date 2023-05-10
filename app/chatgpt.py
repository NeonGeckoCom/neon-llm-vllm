import openai



class ChatGPT:
    def __init__(self, config):
        self.model = config["model"]
        self.role = config["role"]
        self.context_depth = config["context_depth"]
        self.max_tokens = config["max_tokens"]
        openai.api_key = config["key"]

    @staticmethod
    def convert_role(role):
        if (role == "user"):
            role_chatgpt = "user"
        elif (role == "llm"):
            role_chatgpt = "assistant"
        return role_chatgpt

    def ask(self, message, chat_history):
        messages=[
            {"role": "system", "content": self.role},
        ]
        # Context N messages
        for role, content in chat_history[-self.context_depth:]:
            role_chatgpt = self.convert_role(role)
            messages.append({"role": role_chatgpt, "content": content})
        messages.append({"role": "user", "content": message})
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=self.max_tokens,
        )
        bot_message = response.choices[0].message['content']
        return bot_message
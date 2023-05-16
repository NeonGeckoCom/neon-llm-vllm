# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2021 Neongecko.com Inc.
# BSD-3
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
        if role == "user":
            role_chatgpt = "user"
        elif role == "llm":
            role_chatgpt = "assistant"
        return role_chatgpt

    def ask(self, message, chat_history):
        messages = [
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

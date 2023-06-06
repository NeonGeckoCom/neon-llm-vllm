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

import pika

from neon_mq_connector.connector import MQConnector
from neon_mq_connector.utils.network_utils import dict_to_b64
from neon_mq_connector.utils.rabbit_utils import create_mq_callback
from ovos_utils.log import LOG

from neon_llm_chatgpt.chatgpt import ChatGPT
from neon_llm_chatgpt.config import load_config


class ChatgptMQ(MQConnector):
    """
    Module for processing MQ requests from PyKlatchat to LibreTranslate"""

    def __init__(self):
        config = load_config()
        chatgpt_config = config.get("ChatGPT", None)
        self.chatGPT = ChatGPT(chatgpt_config)

        self.service_name = 'neon_llm_chatgpt'

        mq_config = config.get("MQ", None)
        super().__init__(config=mq_config, service_name=self.service_name)

        self.vhost = "/llm"
        self.queue = "chat_gpt_input"
        self.register_consumer(name=self.service_name,
                               vhost=self.vhost,
                               queue=self.queue,
                               callback=self.handle_request,
                               on_error=self.default_error_handler,
                               auto_ack=False)

    @create_mq_callback(include_callback_props=('channel', 'method', 'body'))
    def handle_request(self,
                       channel: pika.channel.Channel,
                       method: pika.spec.Basic.Return,
                       body: dict):
        """
        Handles requests from MQ to ChatGPT received on queue
        "request_chatgpt"

        :param channel: MQ channel object (pika.channel.Channel)
        :param method: MQ return method (pika.spec.Basic.Return)
        :param body: request body (dict)
        """
        message_id = body["message_id"]
        routing_key = body["routing_key"]

        query = body["query"]
        history = body["history"]

        response = self.chatGPT.ask(message=query, chat_history=history)

        api_response = {
            "message_id": message_id,
            "response": response
        }

        channel.basic_publish(exchange='',
                              routing_key=routing_key,
                              body=dict_to_b64(api_response),
                              properties=pika.BasicProperties(
                                  expiration=str(1000)))
        channel.basic_ack(method.delivery_tag)
        LOG.info(f"Handled request: {message_id}")

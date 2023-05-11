import pika
import json
import os.path

from ovos_utils import LOG
from neon_mq_connector.connector import MQConnector
from neon_mq_connector.utils.rabbit_utils import create_mq_callback

from chatgpt import ChatGPT


class ChatgptMQ(MQConnector):
    """
    Module for processing MQ requests from PyKlatchat to LibreTranslate"""

    def __init__(self):
        config = self.load_mq_config()
        chatgpt_config = config.pop("ChatGPT", None)
        self.chatGPT = ChatGPT(chatgpt_config)

        self.service_name = 'mq-chatgpt-api'

        mq_config = config.pop("MQ", None)
        super().__init__(config = mq_config, service_name = self.service_name)

        self.vhost = "/llm"
        self.queue = "chat_gpt_input"
        self.register_consumer(name=self.service_name,
                               vhost=self.vhost,
                               queue=self.queue,
                               callback=self.handle_request,
                               on_error=self.default_error_handler,
                               auto_ack=False)

    def load_mq_config(self, config_path: str = "app/config.json"):
        default_config_path = "app/default_config.json"

        config_path = config_path if os.path.isfile(config_path) else default_config_path
        with open(config_path) as config_file:
            config = json.load(config_file)
        LOG.info(f"Loaded MQ config from path {config_path}")
        return config

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
        #body.pop("message_id", None)
        query = body["query"]
        history = body["history"]

        response = self.chatGPT.ask(message = query, chat_history = history)

        api_response = {
            "response": response
        }
        self.send_message(api_response, queue = self.queue)
        channel.basic_ack(method.delivery_tag)
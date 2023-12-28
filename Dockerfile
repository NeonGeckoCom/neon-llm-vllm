FROM python:3.9-slim

LABEL vendor=neon.ai \
    ai.neon.name="neon-llm-chatgpt"

ENV OVOS_CONFIG_BASE_FOLDER neon
ENV OVOS_CONFIG_FILENAME diana.yaml
ENV XDG_CONFIG_HOME /config
ENV CHATBOT_VERSION v2

COPY docker_overlay/ /
RUN apt update && apt install -y git
WORKDIR /app
COPY . /app
RUN pip install /app

CMD [ "neon-llm-chatgpt" ]
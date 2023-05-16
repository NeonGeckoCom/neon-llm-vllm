FROM python:3.9-slim

ENV XDG_CONFIG_HOME /config
COPY docker_overlay/ /

WORKDIR /app
COPY . /app
RUN pip install /app

CMD [ "neon-llm-chatgpt" ]
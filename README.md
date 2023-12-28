# NeonAI LLM ChatGPT
Proxies API calls to ChatGPT.

## Request Format
API requests should include `history`, a list of tuples of strings, and the current
`query`

>Example Request:
>```json
>{
>  "history": [["user", "hello"], ["llm", "hi"]],
>  "query": "how are you?"
>}
>```

## Response Format
Responses will be returned as dictionaries. Responses should contain the following:
- `response` - String LLM response to the query

## Docker Configuration
When running this as a docker container, the `XDG_CONFIG_HOME` envvar is set to `/config`.
A configuration file at `/config/neon/diana.yaml` is required and should look like:
```yaml
MQ:
  port: <MQ Port>
  server: <MQ Hostname or IP>
  users:
    neon_llm_chat_gpt:
      password: <neon_chatgpt user's password>
      user: neon_chatgpt
LLM_CHAT_GPT:
  key: ""
  model: "gpt-3.5-turbo"
  role: "You are trying to give a short answer in less than 40 words."
  context_depth: 3
  max_tokens: 100
  num_parallel_processes: 2
```

To add support for Chatbotsforum personas, a list of names and prompts can be added
to configuration:
```yaml
llm_bots:
  chat_gpt:
    - name: tutor
      description: |
        You are an AI bot that specializes in tutoring and guiding learners.
        Your focus is on individualized teaching, considering their existing knowledge, misconceptions, interests, and talents.
        Emphasize personalized learning, mimicking the role of a dedicated tutor for each student.
        You're attempting to provide a concise response within a 40-word limit.
```
> `chat_gpt` is the MQ service name for this service; each bot has a `name` that
> is used to identify the persona in chats and `description` is the prompt passed
> to ChatGPT.

For example, if your configuration resides in `~/.config`:
```shell
export CONFIG_PATH="/home/${USER}/.config"
docker run -v ${CONFIG_PATH}:/config neon_llm_chatgpt
```
> Note: If connecting to a local MQ server, you may need to specify `--network host`
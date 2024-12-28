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
from dataclasses import dataclass
from functools import cached_property

import yaml
import openai
from huggingface_hub import hf_hub_download
from pydantic import BaseModel,ValidationError
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import numpy as np

from typing import List, Dict, Optional
from neon_llm_core.llm import NeonLLM
from ovos_utils.log import LOG


COMPLETION_GENERATION_EXTRA_ARGUMENTS = {
    "repetition_penalty": 1.05,
    "use_beam_search": True,
    "best_of": 5,
}

class PileConfig(BaseModel):
    persona2system: Dict[str, str]

# TODO: use this params instead of COMPLETION_GENERATION_EXTRA_ARGUMENTS
class InferenceConfig(BaseModel):
    streaming: bool = True
    temperature: float = 0.0
    repetition_penalty: float = 1.05

class ModelConfig(BaseModel):
    pile: PileConfig
    inference: InferenceConfig

    @classmethod
    def from_yaml(cls, yaml_file="datasets/config.yaml"):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
        try:
            return cls(**data)
        except ValidationError as e:
            raise e


@dataclass(frozen=True)
class ModelMetadata:
    vllm_model_name: str
    model_name: str
    revision: Optional[str] = None

    personas: Dict[str, str]

    model: openai.OpenAI
    tokenizer: PreTrainedTokenizerBase


class VLLM(NeonLLM):

    mq_to_llm_role = {
        "user": "user",
        "llm": "assistant"
    }

    def __init__(self, config):
        super().__init__(config)
        self._context_depth = 0
        
        self.api_url = config["api_url"]
        self.context_depth = config["context_depth"]
        self.max_tokens = config["max_tokens"]
        self.api_key = config["key"]
        self.hf_token = config["hf_token"]
        self.warmup()

    @property
    def context_depth(self):
        return self._context_depth

    @context_depth.setter
    def context_depth(self, value):
        self._context_depth = value + value % 2

    def tokenizer(self, tokenizer_model_name, revision) -> PreTrainedTokenizerBase:
        return AutoTokenizer.from_pretrained(
                    tokenizer_model_name,
                    revision=revision,
                    token=self.hf_token
                )

    @property
    def tokenizer_model_name(self) -> str:
        pass

    def model(self, api_url, api_key) -> openai.OpenAI:
        return openai.OpenAI(
                base_url=f"{api_url}/v1",
                api_key=api_key,
            )

    @property
    def llm_model_name(self) -> str:
        pass

    @property
    def _system_prompt(self) -> str:
        pass

    def warmup(self):
        """
            First initialisation of model and tokenizer properties
            Lazy initialisation causes unexpected connectivity issues
        """
        _ = self.model
        _ = self.tokenizer

    def get_model_metadata(self, api_url: str, api_key: str) -> ModelMetadata:
        model = self.model(api_url, api_key)
        models = model.models.list()
        vllm_model_name = models.data[0].id

        model_name, *suffix = vllm_model_name.split("@")
        revision = dict(enumerate(suffix)).get(0, None)

        tokenizer = self.tokenizer(model_name, revision)
        personas = self.get_personas(model_name, revision)

        return vllm_model_name, ModelMetadata(
            vllm_model_name=vllm_model_name,
            model_name=model_name,
            revision=revision,

            personas=personas,

            model=model,
            tokenizer=tokenizer,
        )
    
    def get_personas(self, model_name: str, revision: str) -> Dict[str, str]:
        config_path = hf_hub_download(model_name, "config.yaml",
                                      subfolder="datasets",
                                      revision=revision,
                                      token=self.hf_token)
        config = ModelConfig.from_yaml(config_path)
        personas = config.pile.persona2system
        return personas
        

    def get_sorted_answer_indexes(self, question: str, answers: List[str], persona: dict) -> List[int]:
        """
            Creates sorted list of answer indexes with respect to order provided in :param answers based on PPL score
            Answers are sorted from best to worst
            :param question: incoming question
            :param answers: list of answers to rank
            :param persona: dictionary with persona's data
            :returns list of indexes
        """
        if not answers:
            return []
        scores = self._score(prompt=question, targets=answers, persona=persona)
        sorted_items = sorted(zip(range(len(answers)), scores), key=lambda x: x[1])
        sorted_items_indexes = [x[0] for x in sorted_items]
        return sorted_items_indexes

    def _call_model(self, prompt: str) -> str:
        """
            Wrapper for VLLM Model generation logic
            :param prompt: Input messages sequence
            :returns: Output text sequence generated by model
        """

        response = self.model.completions.create(
            model=self.llm_model_name,
            prompt=prompt,
            temperature=0,
            max_tokens=self.max_tokens,
            extra_body=COMPLETION_GENERATION_EXTRA_ARGUMENTS,
        )
        text = response.choices[0].text
        LOG.debug(text)
        return text

    def _assemble_prompt(self, message: str, chat_history: List[List[str]], persona: dict, add_generation_prompt: bool = True) -> str:
        """
            Assembles prompt engineering logic
            Setup Guidance:
            https://platform.openai.com/docs/guides/gpt/chat-completions-api

            :param message: Incoming prompt
            :param chat_history: History of preceding conversation
            :returns: assembled prompt
        """
        system_prompt = persona.get("description", self._system_prompt)
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        # Context N messages
        for role, content in chat_history[-self.context_depth:]:
            role_vllm = self.convert_role(role)
            messages.append({"role": role_vllm, "content": content})
        if add_generation_prompt:
            messages.append({"role": "user", "content": message})

        prompt = self.tokenizer.apply_chat_template(conversation=messages,
                                                    tokenize=False,
                                                    add_generation_prompt=add_generation_prompt)
        return prompt

    def _score(self, prompt: str, targets: List[str], persona: dict) -> List[float]:
        """
            Calculates logarithmic probabilities for the list of provided text sequences
            :param prompt: Input text sequence
            :param targets: Output text sequences
            :returns: List of calculated logarithmic probabilities per output text sequence
        """
        prompts = [self._assemble_prompt(message="", chat_history=[["user", prompt], ["llm", target]], persona=persona,
                                         add_generation_prompt=False) for target in targets]
        logprobs_list = self._compute_logprobs(prompts)
        scores_list = [self._evaluate_ppl_score(target, logprobs)
                       for target, logprobs in zip(targets, logprobs_list)]
        return scores_list

    def _tokenize(self, prompt: str) -> None:
        pass

    def _compute_logprobs(self, prompts: List[str]) -> List[openai.types.completion_choice.Logprobs]:
        """
            Computes logprobs for the list of provided prompts
            :param prompts: List of provided prompts
            :returns logprobs for each prompt
        """
        completion = self.model.completions.create(
            model=self.llm_model_name,
            prompt=prompts,
            echo=True,
            logprobs=1,
            max_tokens=1,
            temperature=0,
            extra_body={
                "skip_special_tokens": False
            }
        )
        logprobs_list = [choice.logprobs for choice in completion.choices]
        return logprobs_list

    def _evaluate_ppl_score(self, answer: str, logprobs: openai.types.completion_choice.Logprobs) -> float:
        """
            Evaluates PPL value for the provided answer
            :param answer: string sequence to evaluate
            :param logprobs: logarithmic probabilities of the incoming prompt
            :returns ppl value corresponding to provided string sequence
        """
        tokens = logprobs.tokens
        start_index, end_index = self.find_substring_indices(tokens, answer)
        answer_logprobs = logprobs.token_logprobs[start_index: end_index+2]
        ppl = self._compute_ppl(answer_logprobs)
        return ppl
    
    @staticmethod
    def _compute_ppl(log_probs: List[float]) -> float:
        """ Calculates perplexity value: https://en.wikipedia.org/wiki/Perplexity """
        ppl = np.exp(-np.mean(log_probs))
        return ppl
    
    @staticmethod
    def find_substring_indices(list_of_strings, substring):
        concatenated_string = ''.join(list_of_strings)
        start_pos = concatenated_string.rfind(substring)

        if start_pos == -1:
            return None, None

        end_pos = start_pos + len(substring)

        current_pos = 0
        start_index = None
        end_index = None

        for i, s in enumerate(list_of_strings):
            next_pos = current_pos + len(s)

            if start_index is None and start_pos < next_pos:
                start_index = i

            if end_index is None and end_pos <= next_pos:
                end_index = i
                break
            
            current_pos = next_pos

        return start_index, end_index

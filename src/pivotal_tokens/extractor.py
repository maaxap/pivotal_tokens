import logging
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from uuid import uuid4

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig

from pivotal_tokens.oracle import Oracle
from pivotal_tokens.dict_repo import DictRepo


@dataclass
class PivotalSpan:
    token_ids: list[int]
    span_text: str


class PivotalSpanExtractor(ABC):
    @abstractmethod
    def extract(self,
                reasoning_trace: str,
                system_prompt: str,
                user_prompt: str,
                actual_answer: str,
                expected_answer: str,
                metadata: dict[str, t.Any] | None = None) -> list[PivotalSpan]:
        """
        Extracts pivotal token from the reasoning trace.

        :param reasoning_trace: Reasoning trace.
        :param system_prompt: System prompt.
        :param user_prompt: User prompt.
        :param actual_answer: Answer after end-of-thinking token, e.g. </think>.
        :param expected_answer: Ground truth answer.
        :param metadata: Additional metadata, e.g. sample ID or extraction trial.
        :return: List of extracted pivotal tokens.
        """


@dataclass
class SuccessProbabilityShiftSpan(PivotalSpan):
    prob_before: float
    prob_after: float
    prob_delta: float

    pivotal_context: str

    metadata: dict[str, t.Any] | None = None


class SuccessProbabilityShiftExtractor(PivotalSpanExtractor):
    REPO_NAME_SUCCESS_PROB_ESTIMATION_TRIALS = "trials"
    REPO_NAME_SUBDIVIDED_SEQUENCE = "subdivided_sequences"
    REPO_NAME_SPANS = "spans"

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 oracle: Oracle,
                 repo: DictRepo,
                 prob_threshold: float,
                 num_trials: int,
                 min_prob: float,
                 max_prob: float,
                 batch_size: int,
                 generation_config: GenerationConfig):
        if min_prob < 0.0 or max_prob > 1.0:
            raise ValueError("min_prob and max_prob must be in [0, 1] range")
        
        if min_prob >= max_prob:
            raise ValueError("min_prob must be less than max_prob")
        
        if not (0.0 <= prob_threshold <= 1.0):
            raise ValueError("prob_threshold must be in [0, 1] range")

        self.model = model
        self.tokenizer = tokenizer
        self.oracle = oracle
        self.repo = repo
        self.prob_threshold = prob_threshold
        self.num_trials = num_trials
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.batch_size = batch_size
        self.generation_config = generation_config

        self.prob_cache = {}

    def clear_cache(self):
        self.prob_cache = {}

    def create_context(self, system_prompt: str, user_prompt: str) -> str:
        prompts = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt}]
        context = self.tokenizer.apply_chat_template(prompts,
                                                     tokenize=False,
                                                     add_generation_prompt=True,
                                                     enable_thinking=True)
        return context

    def estimate_success_probability(self,
                                     prefix: str,
                                     system_prompt: str,
                                     user_prompt: str,
                                     expected_answer: str,
                                     metadata: dict[str, t.Any] = None):
        # TODO: Cache key doesn't include generation_config parameters which can affect results
        cache_key = (prefix, system_prompt, user_prompt, self.num_trials, expected_answer)
        if cache_key in self.prob_cache:
            return self.prob_cache[cache_key]

        context = self.create_context(system_prompt=system_prompt, user_prompt=user_prompt)
        context += prefix

        tokenized = self.tokenizer(context, return_tensors="pt", padding=True)
        input_ids = tokenized.input_ids.to(self.model.device)
        attention_mask = tokenized.attention_mask.to(self.model.device) if "attention_mask" in tokenized else None

        # Generate completions in batches
        success_count = 0
        remaining_trials = self.num_trials

        dump_data = {
            "prefix": prefix,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context": context,
            "expected_answer": expected_answer,
            "cache_key": cache_key,
            "metadata": metadata
        }

        while remaining_trials > 0:
            current_batch_size = min(self.batch_size, remaining_trials)

            # Generate completions
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config,
                    return_dict_in_generate=True
                )

            # Check success for each completion
            for trial_num, sequence in enumerate(outputs.sequences):
                # TODO: double check decoded text and special tokens
                completion = self.tokenizer.decode(sequence[input_ids.shape[1]:], skip_special_tokens=True)
                is_success = self.oracle.verify(actual=completion, expected=[expected_answer])

                trial_id = uuid4()
                trial_dump_data = {
                    "trial_id": trial_id,
                    "trial_num": trial_num,
                    "timestamp": datetime.now().isoformat(),
                    "is_success": is_success,
                    **dump_data
                }

                self.repo.save(key=trial_id,
                               data=trial_dump_data,
                               name=self.REPO_NAME_SUCCESS_PROB_ESTIMATION_TRIALS)

                if is_success:
                    success_count += 1

            remaining_trials -= current_batch_size

        success_prob = success_count / self.num_trials
        self.prob_cache[cache_key] = success_prob

        return success_prob

    def split_sequence(self, sequence: list[int]) -> tuple[list[int], list[int]]:
        mid = len(sequence) // 2
        left = sequence[:mid]
        right = sequence[mid:]

        return left, right

    def subdivide_sequence(self,
                           sequence: list[int],
                           prefix: list[int],
                           system_prompt: str,
                           user_prompt: str,
                           expected_answer: str,
                           metadata: dict[str, t.Any] = None):
        prefix = prefix or []

        # Base case 1: Single token or empty sequence
        if len(sequence) <= 1:
            return [sequence]

        # Get the probability before and after the sequence
        prefix_str = self.tokenizer.decode(prefix, skip_special_tokens=False)
        full_seq_str = self.tokenizer.decode(prefix + sequence, skip_special_tokens=False)


        prob_before = self.estimate_success_probability(prefix=prefix_str,
                                                        system_prompt=system_prompt,
                                                        user_prompt=user_prompt,
                                                        expected_answer=expected_answer,
                                                        metadata=metadata)
        prob_after = self.estimate_success_probability(prefix=full_seq_str,
                                                        system_prompt=system_prompt,
                                                        user_prompt=user_prompt,
                                                        expected_answer=expected_answer,
                                                        metadata=metadata)

        subdivide_id = uuid4()
        sequence_str = self.tokenizer.decode(sequence, skip_special_tokens=False)
        dump_data = {
            "subdivide_id": subdivide_id,
            "timestamp": datetime.now().isoformat(),
            "prob_before": prob_before,
            "prob_after": prob_after,
            "prefix": prefix_str,
            "full_seq": full_seq_str,
            "sequence": sequence_str,
            "expected_answer": expected_answer,
            "metadata": metadata,

            # TODO: Large dump_data objects with full prompts repeated at every recursion level - memory intensive
            # "system_prompt": system_prompt,
            # "user_prompt": user_prompt,
        }
        self.repo.save(key=subdivide_id,
                       data=dump_data,
                       name=self.REPO_NAME_SUBDIVIDED_SEQUENCE)

        # Base case 2: No significant change in probability
        if abs(prob_after - prob_before) < self.prob_threshold:
            return [sequence]

        left, right = self.split_sequence(sequence=sequence)

        # Recursively subdivide left side
        left_segments = self.subdivide_sequence(sequence=left,
                                                prefix=prefix,
                                                system_prompt=system_prompt,
                                                user_prompt=user_prompt,
                                                expected_answer=expected_answer,
                                                metadata=metadata)

        # Update prefix for right side by concatenating prefix and left side
        new_prefix = prefix + left

        # Recursively subdivide right side
        right_segments = self.subdivide_sequence(sequence=right,
                                                prefix=new_prefix,
                                                system_prompt=system_prompt,
                                                user_prompt=user_prompt,
                                                expected_answer=expected_answer,
                                                metadata=metadata)

        # Combine all segments
        return left_segments + right_segments

    def extract(self,
                reasoning_trace: str,
                system_prompt: str,
                user_prompt: str,
                actual_answer: str,
                expected_answer: str,
                metadata: dict[str, t.Any] | None = None) -> list[SuccessProbabilityShiftSpan]:
        del actual_answer  # Not used in this extractor

        if reasoning_trace.strip() == "":
            raise ValueError("Reasoning trace is empty")

        metadata = metadata or {}
        init_prob = self.estimate_success_probability(prefix="",
                                                      system_prompt=system_prompt,
                                                      user_prompt=user_prompt,
                                                      expected_answer=expected_answer,
                                                      metadata=metadata)
        if not (self.min_prob <= init_prob <= self.max_prob):
            logging.info(f"Initial probability {init_prob} must be in [{self.min_prob}], {self.max_prob}]")
            return []

        # Subdivide the sequence to find pivotal tokens
        sequence = self.tokenizer.encode(reasoning_trace, add_special_tokens=False)
        subdivided = self.subdivide_sequence(prefix=[],
                                             sequence=sequence,
                                             system_prompt=system_prompt,
                                             user_prompt=user_prompt,
                                             expected_answer=expected_answer,
                                             metadata=metadata)

        context = self.create_context(system_prompt=system_prompt, user_prompt=user_prompt)

        tokenized = self.tokenizer(context, return_tensors="pt", padding=True)
        tokenized_context = tokenized.input_ids[0].tolist()

        pivotal_spans = []
        for span in subdivided:
            # Skip empty span
            if not span:
                continue

            span_text = self.tokenizer.decode(span, skip_special_tokens=False)

            current_context = self.tokenizer.decode(tokenized_context, skip_special_tokens=False)
            prob_before = self.estimate_success_probability(prefix=current_context,
                                                            system_prompt=system_prompt,
                                                            user_prompt=user_prompt,
                                                            expected_answer=expected_answer,
                                                            metadata=metadata)

            current_context_plus_span = self.tokenizer.decode(tokenized_context + span, skip_special_tokens=False)
            prob_after = self.estimate_success_probability(prefix=current_context_plus_span,
                                                            system_prompt=system_prompt,
                                                            user_prompt=user_prompt,
                                                            expected_answer=expected_answer,
                                                            metadata=metadata)

            prob_delta = prob_after - prob_before
            pivotal_span = SuccessProbabilityShiftSpan(token_ids=span,
                                                       span_text=span_text,
                                                       prob_before=prob_before,
                                                       prob_after=prob_after,
                                                       prob_delta=prob_delta,
                                                       pivotal_context=current_context,
                                                       metadata=metadata)
            span_id = uuid4()
            span_dump_data = asdict(pivotal_span)

            self.repo.save(key=span_id,
                           data=span_dump_data,
                           name=self.REPO_NAME_SPANS)

            pivotal_spans.append(pivotal_span)
            tokenized_context = tokenized_context + span

        return pivotal_spans


@dataclass
class LogLikelihoodSpikeSpan(PivotalSpan):
    pass


class LogLikelihoodSpikeExtractor(PivotalSpanExtractor):
    def extract(self,
                reasoning_trace: str,
                system_prompt: str,
                user_prompt: str,
                actual_answer: str,
                expected_answer: str,
                metadata: dict[str, t.Any] | None = None) -> list[LogLikelihoodSpikeSpan]:
        pass

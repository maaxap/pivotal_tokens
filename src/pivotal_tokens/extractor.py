import logging
import re
import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime

import nltk
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig

from pivotal_tokens.oracle import Oracle
from pivotal_tokens.repo import Repo, SampleRepo
from pivotal_tokens.utils import generate_unique_id


THINKING_START_TOKEN = "<think>"
THINKING_END_TOKEN = "</think>"


@dataclass
class PivotalSpan:
    sample_id: str

    span_id: str
    token_ids: list[int]
    span_text: str


class PivotalSpanExtractor(ABC):
    @abstractmethod
    def extract(self,
                sample_id: str,
                reasoning_trace: str,
                system_prompt: str,
                user_prompt: str,
                actual_answer: str,
                expected_answer: str,
                metadata: dict[str, t.Any] | None = None) -> list[PivotalSpan]:
        """
        Extracts pivotal token from the reasoning trace.

        :param sample_id: Sample ID.
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
    is_pivotal: bool

    pivotal_context: str

    metadata: dict[str, t.Any] | None = None


class SuccessProbabilityShiftExtractor(PivotalSpanExtractor):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 oracle: Oracle,
                 base_repo: Repo,
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

        vocab_keys = tokenizer.get_vocab().keys()
        if THINKING_START_TOKEN not in vocab_keys or THINKING_END_TOKEN not in vocab_keys:
            raise ValueError(f"Tokenizer must support special thinking tokens: {THINKING_START_TOKEN}, "
                             f"{THINKING_END_TOKEN}")

        self.model = model
        self.tokenizer = tokenizer
        self.oracle = oracle
        self.base_repo = base_repo
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
                                     sample_repo: SampleRepo,
                                     metadata: dict[str, t.Any] | None = None) -> float:
        # TODO: Cache key doesn't include generation_config parameters which can affect results. Cache must be
        #  cleared for each example.
        cache_key = (prefix, system_prompt, user_prompt, self.num_trials, expected_answer)
        if cache_key in self.prob_cache:
            logging.debug(f"Using cached probability for key: {cache_key}")
            return self.prob_cache[cache_key]

        logging.debug(f"Estimating success probability for key: {cache_key}")
        
        metadata = metadata or {}

        context = self.create_context(system_prompt=system_prompt, user_prompt=user_prompt)
        context += prefix

        tokenized = self.tokenizer(context, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = tokenized.input_ids.to(self.model.device)
        attention_mask = tokenized.attention_mask.to(self.model.device) if "attention_mask" in tokenized else None

        # Generate completions in batches
        success_count = 0
        remaining_trials = self.num_trials

        trial_num = 0
        while remaining_trials > 0:
            current_batch_size = min(self.batch_size, remaining_trials)

            # Generate completions
            with torch.no_grad():
                outputs = self.model.generate(input_ids,
                                              attention_mask=attention_mask,
                                              generation_config=self.generation_config,
                                              return_dict_in_generate=True,
                                              num_return_sequences=current_batch_size)
                
            logging.debug(f"Generated {len(outputs.sequences)} completions for batch size {current_batch_size}")

            # Check success for each completion
            for sequence in outputs.sequences:
                completion = self.tokenizer.decode(sequence, skip_special_tokens=False)
                is_success = self.oracle.verify(actual=completion, expected=[expected_answer])

                trace_n_answer = self.tokenizer.decode(sequence[input_ids.shape[1]:], skip_special_tokens=False)
                logging.debug(f"Trial {trial_num+1}/{self.num_trials}: success={is_success}, "
                              f"expected_answer='{expected_answer}', completion='...{trace_n_answer}'")

                trial_id = generate_unique_id()
                trial_dump_data = {
                    "trial_id": trial_id,
                    "trial_num": trial_num,
                    "timestamp": datetime.now().isoformat(),
                    "prefix": prefix,
                    "completion": completion,
                    "is_success": is_success,
                    "context": context,
                    "expected_answer": expected_answer,
                    "cache_key": cache_key,
                    "additional_metadata": metadata
                }

                sample_repo.save(path="trials", key=trial_id, data=trial_dump_data)

                if is_success:
                    success_count += 1

                trial_num += 1

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
                           sample_repo: SampleRepo):
        # Base case 1: Single token or empty sequence
        if len(sequence) <= 1:
            return [sequence]

        # Get the probability before and after the sequence
        prefix_str = self.tokenizer.decode(prefix, skip_special_tokens=False)
        full_seq_str = self.tokenizer.decode(prefix + sequence, skip_special_tokens=False)

        subdivide_id = generate_unique_id()

        logging.debug(f"Estimating prob before and after for prefix (ID: {subdivide_id}): {prefix_str}")

        prob_before = self.estimate_success_probability(prefix=prefix_str,
                                                        system_prompt=system_prompt,
                                                        user_prompt=user_prompt,
                                                        expected_answer=expected_answer,
                                                        sample_repo=sample_repo,
                                                        metadata={"stage": "before_subdivision",
                                                                  "subdivide_id": subdivide_id})
        logging.debug(f"Prob before subdivision: {prob_before}")

        prob_after = self.estimate_success_probability(prefix=full_seq_str,
                                                        system_prompt=system_prompt,
                                                        user_prompt=user_prompt,
                                                        expected_answer=expected_answer,
                                                        sample_repo=sample_repo,
                                                        metadata={"stage": "after_subdivision",
                                                                  "subdivide_id": subdivide_id})
        logging.debug(f"Prob after subdivision: {prob_after}")
    
        prob_delta = prob_after - prob_before
        sequence_str = self.tokenizer.decode(sequence, skip_special_tokens=False)
        notable_change = abs(prob_delta) >= self.prob_threshold

        dump_data = {
            "subdivision_id": subdivide_id,
            "timestamp": datetime.now().isoformat(),
            "prob_before": prob_before,
            "prob_after": prob_after,
            "prob_delta": prob_delta,
            "notable_change": notable_change,
            "prefix": prefix_str,
            "full_seq": full_seq_str,
            "sequence": sequence_str,
            "expected_answer": expected_answer
        }
        sample_repo.save(path="subdivisions", key=subdivide_id, data=dump_data)

        # Base case 2: No significant change in probability
        if not notable_change:
            logging.debug(f"Probability delta {prob_delta} below threshold {self.prob_threshold}, stopping "
                          f"subdivision for sequence: {sequence_str}")
            return [sequence]

        left, right = self.split_sequence(sequence=sequence)

        # Recursively subdivide left side
        left_segments = self.subdivide_sequence(sequence=left,
                                                prefix=prefix,
                                                system_prompt=system_prompt,
                                                user_prompt=user_prompt,
                                                expected_answer=expected_answer,
                                                sample_repo=sample_repo)

        # Update prefix for right side by concatenating prefix and left side
        new_prefix = prefix + left

        # Recursively subdivide right side
        right_segments = self.subdivide_sequence(sequence=right,
                                                prefix=new_prefix,
                                                system_prompt=system_prompt,
                                                user_prompt=user_prompt,
                                                expected_answer=expected_answer,
                                                sample_repo=sample_repo)

        # Combine all segments
        return left_segments + right_segments

    def extract(self,
                sample_id: str,
                reasoning_trace: str,
                system_prompt: str,
                user_prompt: str,
                actual_answer: str,
                expected_answer: str,
                metadata: dict[str, t.Any] | None = None) -> list[SuccessProbabilityShiftSpan]:
        """
        Extracts pivotal tokens and spans from the reasoning trace based on shifts in success probability.
        Success probability is estimated by generating completions from the model and verifying them with the oracle.

        Results are organized hierarchically under {repo_dir}/{sample_id}/.

        :param reasoning_trace: Reasoning trace. If not ending with THINKING_END_TOKEN, it will be appended.
        :param system_prompt: System prompt.
        :param user_prompt: User prompt.
        :param actual_answer: Answer after end-of-thinking token, e.g. </think>.
        :param expected_answer: Ground truth answer.
        :param sample_id: Unique identifier for this sample (will be used as directory name).
        :param metadata: Additional metadata (e.g., dataset name, difficulty).
        :return: List of extracted pivotal spans with success probability shifts.
        """
        del actual_answer  # Not used in this extractor

        if reasoning_trace.strip() == "":
            raise ValueError("Reasoning trace is empty")
        
        if reasoning_trace.endswith(THINKING_END_TOKEN):
            raise ValueError(f"Reasoning trace must not finish with '{THINKING_END_TOKEN}'")

        logging.debug(f"Starting extraction for sample {sample_id}")

        # Create SampleRepo for this extraction run
        logging.debug(f"Creating SampleRepo for sample ID: {sample_id}")
        sample_repo = SampleRepo(sample_id=sample_id, base_repo=self.base_repo)

        init_prob = self.estimate_success_probability(prefix="",
                                                      system_prompt=system_prompt,
                                                      user_prompt=user_prompt,
                                                      expected_answer=expected_answer,
                                                      sample_repo=sample_repo,
                                                      metadata={"stage": "initial_probability"})
        
        # Save sample metadata at the start
        metadata = metadata or {}
        sample_metadata = {
            "sample_id": sample_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "reasoning_trace": reasoning_trace,
            "expected_answer": expected_answer,
            "additional_metadata": metadata,
            "init_prob": init_prob,
            "created_at": datetime.now().isoformat()
        }

        logging.debug(f"Saving sample metadata for sample ID: {sample_id}")
        sample_repo.save(path="", key="metadata", data=sample_metadata)

        logging.debug(f"Initial success probability: {init_prob}")
        if not (self.min_prob <= init_prob <= self.max_prob):
            logging.info(f"Initial probability {init_prob} must be in [{self.min_prob}], {self.max_prob}]")
            return []

        if not reasoning_trace.startswith(THINKING_START_TOKEN):
            logging.debug(f"Appending missing '{THINKING_START_TOKEN}' to the reasoning trace")
            reasoning_trace = f"{THINKING_START_TOKEN}\n{reasoning_trace}"
    
        # Subdivide the sequence to find pivotal tokens
        sequence = self.tokenizer.encode(reasoning_trace, add_special_tokens=False)
        subdivided = self.subdivide_sequence(prefix=[],
                                             sequence=sequence,
                                             system_prompt=system_prompt,
                                             user_prompt=user_prompt,
                                             expected_answer=expected_answer,
                                             sample_repo=sample_repo)

        context = self.create_context(system_prompt=system_prompt, user_prompt=user_prompt)

        current_prefix = ""
        pivotal_spans = []
        for span in subdivided:
            # Skip empty span
            if not span:
                continue

            span_text = self.tokenizer.decode(span, skip_special_tokens=False)

            if len(span) == 1:
                logging.debug(f"Collecting stats for span: '{span_text}'")

                prob_before = self.estimate_success_probability(prefix=current_prefix,
                                                                system_prompt=system_prompt,
                                                                user_prompt=user_prompt,
                                                                expected_answer=expected_answer,
                                                                sample_repo=sample_repo)
                logging.debug(f"Probability before adding span: {prob_before}")

                current_prefix_plus_span = current_prefix + span_text
                prob_after = self.estimate_success_probability(prefix=current_prefix_plus_span,
                                                               system_prompt=system_prompt,
                                                               user_prompt=user_prompt,
                                                               expected_answer=expected_answer,
                                                               sample_repo=sample_repo)
                logging.debug(f"Probability after adding span: {prob_after}")

                prob_delta = prob_after - prob_before
                logging.debug(f"Probability delta for span: {prob_delta}")

                span_id = generate_unique_id()
                pivotal_context = context + current_prefix
                is_pivotal = abs(prob_delta) >= self.prob_threshold
                pivotal_span = SuccessProbabilityShiftSpan(sample_id=sample_id,
                                                        span_id=span_id,
                                                        token_ids=span,
                                                        span_text=span_text,
                                                        prob_before=prob_before,
                                                        prob_after=prob_after,
                                                        prob_delta=prob_delta,
                                                        is_pivotal=is_pivotal,
                                                        pivotal_context=pivotal_context,
                                                        metadata=metadata)
                span_dump_data = asdict(pivotal_span)
                sample_repo.save(path="spans", key=span_id, data=span_dump_data)

                logging.debug(f"Identified pivotal span: '{span_text}' with delta {prob_delta}")

                pivotal_spans.append(pivotal_span)

            current_prefix += span_text

        return pivotal_spans


class SuccessProbabilityShiftSentenceExtractor(SuccessProbabilityShiftExtractor):
    SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    QUOTE_CLEAN_RE = re.compile(r"^[\"'“”‘’]+|[\"'“”‘’]+$")


    def split_into_sentences(self, text: str) -> list[str]:
        sentences = nltk.tokenize.sent_tokenize(text, language='english')
        return sentences

    def split_sentence_sequence(self, sequence: list[str]) -> tuple[list[str], list[str]]:
        mid = len(sequence) // 2
        left = sequence[:mid]
        right = sequence[mid:]

        return left, right

    def subdivide_sentence_sequence(self,
                                    sequence: list[str],
                                    prefix: list[str],
                                    system_prompt: str,
                                    user_prompt: str,
                                    expected_answer: str,
                                    sample_repo: SampleRepo) -> list[list[str]]:
        # Base case 1: Single sentence or empty sequence
        if len(sequence) <= 1:
            return [sequence]

        prefix_str = " ".join(prefix)
        sequence_str = " ".join(sequence)
        full_seq_str = prefix_str + " " + sequence_str

        subdivide_id = generate_unique_id()

        logging.debug(f"Estimating prob before and after for prefix (ID: {subdivide_id}): {prefix_str}")

        prob_before = self.estimate_success_probability(prefix=prefix_str,
                                                        system_prompt=system_prompt,
                                                        user_prompt=user_prompt,
                                                        expected_answer=expected_answer,
                                                        sample_repo=sample_repo,
                                                        metadata={"stage": "before_subdivision",
                                                                  "subdivide_id": subdivide_id})
        logging.debug(f"Prob before subdivision: {prob_before}")

        prob_after = self.estimate_success_probability(prefix=full_seq_str,
                                                       system_prompt=system_prompt,
                                                       user_prompt=user_prompt,
                                                       expected_answer=expected_answer,
                                                       sample_repo=sample_repo,
                                                       metadata={"stage": "after_subdivision",
                                                                 "subdivide_id": subdivide_id})
        logging.debug(f"Prob after subdivision: {prob_after}")

        prob_delta = prob_after - prob_before
        notable_change = abs(prob_delta) >= self.prob_threshold

        dump_data = {
            "subdivision_id": subdivide_id,
            "timestamp": datetime.now().isoformat(),
            "prob_before": prob_before,
            "prob_after": prob_after,
            "prob_delta": prob_delta,
            "notable_change": notable_change,
            "prefix": prefix_str,
            "full_seq": full_seq_str,
            "sequence": sequence_str,
            "expected_answer": expected_answer
        }
        sample_repo.save(path="subdivisions", key=subdivide_id, data=dump_data)

        # Base case 2: No significant change in probability
        if not notable_change:
            logging.debug(f"Probability delta {prob_delta} below threshold {self.prob_threshold}, stopping "
                          f"subdivision for sequence: {sequence_str}")
            return [sequence]

        left, right = self.split_sentence_sequence(sequence=sequence)

        # Recursively subdivide left side
        left_segments = self.subdivide_sentence_sequence(sequence=left,
                                                         prefix=prefix,
                                                         system_prompt=system_prompt,
                                                         user_prompt=user_prompt,
                                                         expected_answer=expected_answer,
                                                         sample_repo=sample_repo)

        # Update prefix for right side by concatenating prefix and left side
        new_prefix = prefix + left

        # Recursively subdivide right side
        right_segments = self.subdivide_sentence_sequence(sequence=right,
                                                          prefix=new_prefix,
                                                          system_prompt=system_prompt,
                                                          user_prompt=user_prompt,
                                                          expected_answer=expected_answer,
                                                          sample_repo=sample_repo)

        # Combine all segments
        return left_segments + right_segments

    def extract(self,
                sample_id: str,
                reasoning_trace: str,
                system_prompt: str,
                user_prompt: str,
                actual_answer: str,
                expected_answer: str,
                metadata: dict[str, t.Any] | None = None) -> list[SuccessProbabilityShiftSpan]:
        """
        Extracts pivotal sentences from the reasoning trace based on shifts in success probability.
        Success probability is estimated by generating completions from the model and verifying them with the oracle.

        Results are organized hierarchically under {repo_dir}/{sample_id}/.

        :param reasoning_trace: Reasoning trace. If not ending with THINKING_END_TOKEN, it will be appended.
        :param system_prompt: System prompt.
        :param user_prompt: User prompt.
        :param actual_answer: Answer after end-of-thinking token, e.g. </think>.
        :param expected_answer: Ground truth answer.
        :param sample_id: Unique identifier for this sample (will be used as directory name).
        :param metadata: Additional metadata (e.g., dataset name, difficulty).
        :return: List of extracted pivotal sentence spans with success probability shifts.
        """
        del actual_answer  # Not used in this extractor

        if reasoning_trace.strip() == "":
            raise ValueError("Reasoning trace is empty")
        
        if reasoning_trace.endswith(THINKING_END_TOKEN):
            raise ValueError(f"Reasoning trace must not finish with '{THINKING_END_TOKEN}'")

        logging.debug(f"Starting sentence-level spans extraction for sample {sample_id}")

        # Create SampleRepo for this extraction run
        logging.debug(f"Creating SampleRepo for sample ID: {sample_id}")
        sample_repo = SampleRepo(sample_id=sample_id, base_repo=self.base_repo)

        init_prob = self.estimate_success_probability(prefix="",
                                                      system_prompt=system_prompt,
                                                      user_prompt=user_prompt,
                                                      expected_answer=expected_answer,
                                                      sample_repo=sample_repo,
                                                      metadata={"stage": "initial_probability"})
        
        # Save sample metadata at the start
        metadata = metadata or {}
        sample_metadata = {
            "sample_id": sample_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "reasoning_trace": reasoning_trace,
            "expected_answer": expected_answer,
            "additional_metadata": metadata,
            "init_prob": init_prob,
            "created_at": datetime.now().isoformat()
        }

        logging.debug(f"Saving sample metadata for sample ID: {sample_id}")
        sample_repo.save(path="", key="metadata", data=sample_metadata)

        logging.debug(f"Initial success probability: {init_prob}")
        if not (self.min_prob <= init_prob <= self.max_prob):
            logging.info(f"Initial probability {init_prob} must be in [{self.min_prob}, {self.max_prob}]")
            return []

        if not reasoning_trace.startswith(THINKING_START_TOKEN):
            logging.debug(f"Appending missing '{THINKING_START_TOKEN}' to the reasoning trace")
            reasoning_trace = f"{THINKING_START_TOKEN}\n{reasoning_trace}"

        # Split reasoning trace into sentences
        # TODO: check if the first sentence contains <think> token
        sentences = self.split_into_sentences(reasoning_trace)
        sentences_joined = "\n***\n".join(sentences)
        logging.debug(f"Split reasoning trace into {len(sentences)} sentences:\n\n{sentences_joined}")

        context = self.create_context(system_prompt=system_prompt, user_prompt=user_prompt)
        
        subdivided = self.subdivide_sentence_sequence(sequence=sentences,
                                                      prefix=[],
                                                      system_prompt=system_prompt,
                                                      user_prompt=user_prompt,
                                                      expected_answer=expected_answer,
                                                      sample_repo=sample_repo)

        current_prefix = ""
        pivotal_spans = []
        for span in subdivided:
            # Skip empty span
            if not span:
                continue

            span_text = " ".join(span)

            if len(span) == 1:
                logging.debug(f"Processing sentence: '{span_text}'")

                prob_before = self.estimate_success_probability(prefix=current_prefix,
                                                                system_prompt=system_prompt,
                                                                user_prompt=user_prompt,
                                                                expected_answer=expected_answer,
                                                                sample_repo=sample_repo)
                logging.debug(f"Probability before adding sentence: {prob_before}")

                current_prefix_plus_sentence = current_prefix + span_text
                prob_after = self.estimate_success_probability(prefix=current_prefix_plus_sentence,
                                                               system_prompt=system_prompt,
                                                               user_prompt=user_prompt,
                                                               expected_answer=expected_answer,
                                                               sample_repo=sample_repo)
                logging.debug(f"Probability after adding sentence: {prob_after}")

                prob_delta = prob_after - prob_before
                logging.debug(f"Probability delta for sentence: {prob_delta}")

                # Tokenize sentence to get token IDs
                sentence_token_ids = self.tokenizer.encode(span_text, add_special_tokens=False)

                span_id = generate_unique_id()
                pivotal_context = context + current_prefix
                is_pivotal = abs(prob_delta) >= self.prob_threshold

                pivotal_span = SuccessProbabilityShiftSpan(
                    sample_id=sample_id,
                    span_id=span_id,
                    token_ids=sentence_token_ids,
                    span_text=span_text,
                    prob_before=prob_before,
                    prob_after=prob_after,
                    prob_delta=prob_delta,
                    is_pivotal=is_pivotal,
                    pivotal_context=pivotal_context,
                    metadata=metadata
                )

                span_dump_data = asdict(pivotal_span)
                sample_repo.save(path="spans", key=span_id, data=span_dump_data)

                logging.debug(f"Identified sentence span: '{span_text}' with delta {prob_delta}")

                pivotal_spans.append(pivotal_span)

            # Update prefix for next iteration
            current_prefix += span_text

        return pivotal_spans


@dataclass
class LogLikelihoodSpikeSpan(PivotalSpan):
    logprob: float
    nll: float
    norm_coeff: int
    is_spike: bool
    
    pivotal_context: str
    
    metadata: dict[str, t.Any] | None = None


class LogLikelihoodSpikeExtractor(PivotalSpanExtractor):
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 base_repo: Repo):
        """
        :param model: Pre-trained model for computing log-likelihoods.
        :param tokenizer: Tokenizer corresponding to the model.
        :param base_repo: Base repository for saving results.
        :param spike_threshold: Threshold for identifying spike tokens (absolute deviation from mean).
        :param min_logprob: Minimum log probability to consider (for filtering).
        :param max_logprob: Maximum log probability to consider (for filtering).
        """
        vocab_keys = tokenizer.get_vocab().keys()
        if THINKING_START_TOKEN not in vocab_keys or THINKING_END_TOKEN not in vocab_keys:
            raise ValueError(f"Tokenizer must support special thinking tokens: {THINKING_START_TOKEN}, "
                             f"{THINKING_END_TOKEN}")

        self.model = model
        self.tokenizer = tokenizer
        self.base_repo = base_repo

    def create_context(self, system_prompt: str, user_prompt: str) -> str:
        prompts = [{"role": "system", "content": system_prompt},
                   {"role": "user", "content": user_prompt}]
        context = self.tokenizer.apply_chat_template(prompts,
                                                     tokenize=False,
                                                     add_generation_prompt=True,
                                                     enable_thinking=True)
        return context

    @torch.no_grad()
    def calc_loglikelihood_per_token(self,
                                     context: str,
                                     completion: str,
                                     answer_suffix: str) -> list[dict]:
        """
        Calculate log-likelihood for each token in the completion.
        
        :param context: Context string (prompt + thinking start token).
        :param completion: Completion string (reasoning trace).
        :param answer_suffix: Answer suffix to append after each prefix.
        :return: List of dicts with token statistics.
        """
        tokens = []

        self.model.eval()

        prompt_tok = self.tokenizer(context, return_tensors="pt")
        prompt_tok = {k: v.to(self.model.device) for k, v in prompt_tok.items()}

        _, prompt_length = prompt_tok.input_ids.shape
        if prompt_length < 2:
            raise ValueError("Need at least 2 tokens to score next-token log-probs in the prompt.")

        full_text = context + completion
        full_tok = self.tokenizer(full_text, return_tensors="pt", return_attention_mask=False)

        indices = range(prompt_tok.input_ids.shape[1], full_tok.input_ids.shape[1])
        for t in indices:
            prefix_ids = full_tok.input_ids[:, :t]

            prefix_str = self.tokenizer.decode(prefix_ids[0], skip_special_tokens=False)
            prefix_str += answer_suffix

            prefix_tok = self.tokenizer(prefix_str, return_tensors="pt")
            prefix_tok = {k: v.to(self.model.device) for k, v in prefix_tok.items()}

            labels = prefix_tok.input_ids.clone()
            labels[:, :t] = -100

            out = self.model(**prefix_tok, labels=labels)
            nnll = out.loss.item()

            token_id = prefix_ids[:, -1:]
            token = self.tokenizer.decode(token_id[0])

            logits = out.logits[:, t-1:t, :][0].detach()
            all_token_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_prob = all_token_probs.gather(-1, token_id.to(self.model.device).T).squeeze(-1).item()

            norm_coeff = (labels != -100).sum().item()

            tokens.append({
                'idx': t,
                'token_id': token_id[0].item(),
                'token': token,
                'nll': nnll,
                'norm_coeff': norm_coeff,
                'logprob': token_prob
            })

        return tokens

    def extract(self,
                sample_id: str,
                reasoning_trace: str,
                system_prompt: str,
                user_prompt: str,
                actual_answer: str,
                expected_answer: str,
                metadata: dict[str, t.Any] | None = None) -> list[LogLikelihoodSpikeSpan]:
        """
        Extracts tokens with significant log-likelihood spikes from the reasoning trace.

        Results are organized hierarchically under {repo_dir}/{sample_id}/.

        :param sample_id: Unique identifier for this sample (will be used as directory name).
        :param reasoning_trace: Reasoning trace.
        :param system_prompt: System prompt.
        :param user_prompt: User prompt.
        :param actual_answer: Answer after end-of-thinking token, e.g. </think>.
        :param expected_answer: Ground truth answer.
        :param metadata: Additional metadata (e.g., dataset name, difficulty).
        :return: List of extracted spike spans with log-likelihood statistics.
        """
        del expected_answer  # Not used in this extractor

        if reasoning_trace.strip() == "":
            raise ValueError("Reasoning trace is empty")

        logging.debug(f"Starting log-likelihood spike extraction for sample {sample_id}")

        # Create SampleRepo for this extraction run
        logging.debug(f"Creating SampleRepo for sample ID: {sample_id}")
        sample_repo = SampleRepo(sample_id=sample_id, base_repo=self.base_repo)

        # Save sample metadata at the start
        metadata = metadata or {}
        sample_metadata = {
            "sample_id": sample_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "reasoning_trace": reasoning_trace,
            "actual_answer": actual_answer,
            "additional_metadata": metadata,
            "created_at": datetime.now().isoformat()
        }

        logging.debug(f"Saving sample metadata for sample ID: {sample_id}")
        sample_repo.save(path="", key="metadata", data=sample_metadata)

        if not reasoning_trace.startswith(THINKING_START_TOKEN):
            logging.debug(f"Appending missing '{THINKING_START_TOKEN}' to the reasoning trace")
            reasoning_trace = f"{THINKING_START_TOKEN}\n{reasoning_trace}"

        # Create context and answer suffix
        context = self.create_context(system_prompt=system_prompt, user_prompt=user_prompt)
        answer_suffix = f"\n{THINKING_END_TOKEN}\n{actual_answer}"

        # Calculate log-likelihood for each token
        logging.debug("Calculating log-likelihoods for all tokens")
        token_stats = self.calc_loglikelihood_per_token(
            context=context,
            completion=reasoning_trace,
            answer_suffix=answer_suffix
        )

        if not token_stats:
            logging.warning("No tokens found in reasoning trace")
            return []

        # Calculate mean and std of log probabilities for spike detection
        logprobs = [t['logprob'] for t in token_stats]
        mean_logprob = sum(logprobs) / len(logprobs)
        variance = sum((lp - mean_logprob) ** 2 for lp in logprobs) / len(logprobs)
        std_logprob = variance ** 0.5

        logging.debug(f"Log-probability statistics: mean={mean_logprob:.4f}, std={std_logprob:.4f}")

        # Extract spike spans
        pivotal_spans = []
        for token_stat in token_stats:
            token = token_stat['token']
            logprob = token_stat['logprob']
            
            # Calculate deviation from mean
            deviation = abs(logprob - mean_logprob)
            is_spike = deviation >= self.spike_threshold * std_logprob

            # Filter by logprob range if specified
            if logprob < self.min_logprob or logprob > self.max_logprob:
                logging.debug(f"Skipping token '{token}' with logprob {logprob:.4f} outside range")
                continue

            logging.debug(f"Token '{token}': logprob={logprob:.4f}, deviation={deviation:.4f}, "
                         f"is_spike={is_spike}")

            span_id = generate_unique_id()
            
            # Reconstruct context up to this token
            idx = token_stat['idx']
            prefix_ids = self.tokenizer.encode(context + reasoning_trace, add_special_tokens=False)[:idx]
            pivotal_context = self.tokenizer.decode(prefix_ids, skip_special_tokens=False)

            spike_span = LogLikelihoodSpikeSpan(
                sample_id=sample_id,
                span_id=span_id,
                token_ids=[token_stat['token_id']],
                span_text=token,
                logprob=logprob,
                nll=token_stat['nll'],
                norm_coeff=token_stat['norm_coeff'],
                is_spike=is_spike,
                pivotal_context=pivotal_context,
                metadata={
                    **(metadata or {}),
                    'mean_logprob': mean_logprob,
                    'std_logprob': std_logprob,
                    'deviation': deviation,
                    'idx': idx
                }
            )

            span_dump_data = asdict(spike_span)
            sample_repo.save(path="spans", key=span_id, data=span_dump_data)

            if is_spike:
                logging.debug(f"Identified spike span: '{token}' with logprob {logprob:.4f}")

            pivotal_spans.append(spike_span)

        logging.debug(f"Extracted {len(pivotal_spans)} spans, "
                     f"{sum(1 for s in pivotal_spans if s.is_spike)} are spikes")

        return pivotal_spans

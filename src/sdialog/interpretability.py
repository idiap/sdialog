"""
interpretability.py

This submodule provides classes and hooks for inspecting and interpreting the internal representations
of PyTorch-based language models during forward passes. It enables the registration of hooks on specific
model layers to capture token-level and utterance-level information, facilitating analysis of model behavior
and interpretability. The module is designed to work with conversational agents and integrates with
tokenizers and memory structures, supporting the extraction and inspection of tokens, representations,
and system instructions across utterances.

Classes:
    - BaseHook: Base class for managing PyTorch forward hooks.
    - UtteranceTokenHook: Captures token IDs at the embedding layer for each utterance.
    - RepresentationHook: Captures intermediate representations from specified model layers.
    - Inspector: Manages hooks, extracts representations, and provides utilities for analysis.
    - InspectionUtterance: Represents a single utterance, exposing its tokens for inspection.
    - InspectionUnit: Represents a single token within an utterance, allowing access to its representations.

Typical usage involves attaching hooks to a model, accumulating utterance and token data during inference,
and providing interfaces for downstream interpretability and analysis tasks.

"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Séverin Baroudi <severin.baroudi@lis-lab.fr>, Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import torch
import logging
import einops

from abc import ABC
from langchain_core.messages import SystemMessage
from functools import partial


logger = logging.getLogger(__name__)


def simple_steering_function(activation, direction, strength=1, op="+"):
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    proj = einops.einsum(activation, direction.view(-1, 1), '... d_act, d_act single -> ... single') * direction
    return activation + proj * strength if op == "+" else activation - proj * strength


class Steerer(ABC):
    pass


class DirectionSteerer(Steerer):
    def __init__(self, direction, inspector=None, strength=None):
        self.direction = direction
        self.inspector = inspector
        self.strength = None

    def __add__(self, other: "Inspector"):
        if type(other) is Inspector:
            if self.strength is None:
                other.add_steering_function(partial(simple_steering_function, direction=self.direction, op="+"))
            else:
                other.add_steering_function(partial(simple_steering_function,
                                                    direction=self.direction, op="+", strength=self.strength))
                self.strength = None
            self.inspector = other
        return other

    def __sub__(self, other):
        if type(other) is Inspector:
            if self.strength is None:
                other.add_steering_function(partial(simple_steering_function, direction=self.direction, op="-"))
            else:
                other.add_steering_function(partial(simple_steering_function,
                                                    direction=self.direction, op="-", strength=self.strength))
                self.strength = None
            self.inspector = other
        return other

    def __mul__(self, other):
        if type(other) in [float, int]:
            if self.inspector is not None:
                self.inspector.steering_function[-1] = partial(self.inspector.steering_function[-1], strength=other)
            else:
                self.strength = other
        return self


class BaseHook:
    """
    Base class for registering and managing PyTorch forward hooks on model layers.
    """
    def __init__(self, layer_key, hook_fn, agent):
        self.layer_key = layer_key
        self.hook_fn = hook_fn
        self.handle = None
        self.agent = agent

    def _hook(self):
        pass

    def register(self, model):
        """
        Registers the hook on the given model using the layer_key.
        """
        layer = dict(model.named_modules())[self.layer_key]
        self.handle = layer.register_forward_hook(self.hook_fn)
        return self.handle

    def remove(self):
        """
        Removes the hook if it is registered.
        """
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class UtteranceTokenHook(BaseHook):
    """
    A BaseHook for the utterance_token_hook, always used on the embedding layer.
    """
    def __init__(self, agent):
        super().__init__('model.embed_tokens', self._hook, agent=agent)
        self.utterance_list = []
        self.current_utterance_ids = None  # Now a tensor
        self.hook_state = {
            'tokenizer': None,
        }
        self.representation_cache = {}
        self.agent = agent

    def new_utterance_event(self, memory):
        self.utterance_list.append({'mem': memory, 'output_tokens': []})
        self.process_current_utterance_ids()

    def _hook(self, module, input, output):
        input_ids = input[0].detach().cpu()
        self.register_representations(input_ids)

    def register_representations(self, input_ids):
        if input_ids.shape[-1] == 1:
            # Accumulate token IDs as a tensor (generated tokens only)
            if self.current_utterance_ids is None:
                self.current_utterance_ids = input_ids
            else:
                self.current_utterance_ids = torch.cat([self.current_utterance_ids, input_ids], dim=1)
        else:
            # Detected system prompt (input_ids.shape[-1] != 1), do not accumulate
            # Optionally process/reset current utterance if needed
            if self.current_utterance_ids is not None:
                self.current_utterance_ids = None

    def process_current_utterance_ids(self):
        tokenizer = self.hook_state.get('tokenizer')

        token_list = self.current_utterance_ids.squeeze(0).tolist()
        text = tokenizer.decode(token_list, skip_special_tokens=True)
        tokens = tokenizer.convert_ids_to_tokens(token_list)

        # No longer create an InspectionUnit here; just store the tokens list
        utterance_dict = {
            'input_ids': self.current_utterance_ids,
            'text': text,
            'tokens': tokens,
            'utterance_index': len(self.utterance_list) - 1
        }
        # Append an InspectionUtterance instance instead of a dict
        current_utterance_inspector = InspectionUtterance(utterance_dict, agent=self.agent)
        self.utterance_list[-1]['output_tokens'].append(current_utterance_inspector)


class RepresentationHook(BaseHook):
    """
    A BaseHook for capturing representations from a specific model layer.
    """

    def __init__(self, layer_key, cache_key, representation_cache, utterance_list, steering_function=None):
        """
        Args:
            layer_key: The key identifying the layer to hook into.
            cache_key: Key under which to store the outputs in the cache.
            representation_cache: A nested dictionary or structure to store representations.
            utterance_list: List used to track current utterance index.
            steering_function: Optional function to apply to output_tensor before caching.
        """
        super().__init__(layer_key, self._hook, agent=None)
        self.cache_key = cache_key
        self.representation_cache = representation_cache
        self.utterance_list = utterance_list
        self.steering_function = steering_function  # Store the optional function

        # Initialize the nested cache
        _ = self.representation_cache[len(self.utterance_list)][self.cache_key]  # This will initialize to []

    def _hook(self, module, input, output):
        """Hook to extract and store model representation from the output."""
        utterance_index = len(self.utterance_list)

        # Extract the main tensor from output if it's a tuple or list
        output_tensor = output[0] if isinstance(output, (tuple, list)) else output

        # Ensure output_tensor is a torch.Tensor before proceeding
        if not isinstance(output_tensor, torch.Tensor):
            raise TypeError(f"Expected output to be a Tensor, got {type(output_tensor)}")

        # Store representation only if the second dimension is 1
        if output_tensor.ndim >= 2 and output_tensor.shape[1] == 1:
            self.representation_cache[utterance_index][self.cache_key].append(output_tensor.detach().cpu())
            # Now apply the steering function, if it exists
            if self.steering_function is not None:
                if type(self.steering_function) is list:
                    for func in self.steering_function:
                        output_tensor = func(output_tensor)
                elif callable(self.steering_function):
                    output_tensor = self.steering_function(output_tensor)

        if isinstance(output, (tuple, list)):
            output = (output_tensor, *output[1:]) if isinstance(output, tuple) else [output_tensor, *output[1:]]
        else:
            output = output_tensor

        return output


class Inspector:
    def __init__(self, to_watch=None, agent=None, steering_function=None):
        """
        Inspector for managing hooks and extracting representations from a model.

        Args:
            to_watch: Dict mapping model layer names to cache keys.
            agent: The agent containing the model and hooks.
            steering_function: Optional function to apply on output tensors in hooks.
        """
        self.to_watch = to_watch if to_watch is not None else {}
        self.agent = agent
        self.steering_function = steering_function

        if self.agent is not None and self.to_watch:
            self.agent.add_hooks(self.to_watch, steering_function=self.steering_function)

    def __len__(self):
        return len(self.agent.utterance_list)

    def __iter__(self):
        return (utt['output_tokens'][0] for utt in self.agent.utterance_list)

    def __getitem__(self, index):
        return self.agent.utterance_list[index]['output_tokens'][0]

    def __add__(self, other):
        if type(other) is Steerer:
            other.__add__(self)
        return self

    def __sub__(self, other):
        if type(other) is Steerer:
            other.__sub__(self)
        return self

    def add_agent(self, agent):
        self.agent = agent
        if self.to_watch:
            self.agent.add_hooks(self.to_watch, steering_function=self.steering_function)

    def add_steering_function(self, steering_function):
        """
        Adds a steering function to the inspector's list of functions.
        """
        if type(steering_function) is not list:
            if callable(self.steering_function):
                self.steering_function = [self.steering_function]
            else:
                self.steering_function = []
        self.steering_function.append(steering_function)

    def add_hooks(self, to_watch):
        """
        Adds hooks to the agent's model based on the provided to_watch mapping.
        Each entry in to_watch should map a layer name to a cache key.
        The new entries are appended to the existing self.to_watch dictionary.
        """
        if self.agent is None:
            raise ValueError("No agent assigned to Inspector.")

        # Append to existing to_watch instead of replacing
        self.to_watch.update(to_watch)

        self.agent.add_hooks(to_watch, steering_function=self.steering_function)

    def recap(self):
        """
        Prints and returns the current hooks assigned to the inspector's agent.
        Also prints the 'to_watch' mapping in a clean, readable format.
        Includes any found instructions across utterances.
        """
        if self.agent is None:
            logger.warning("No agent is currently assigned.")
            return None

        num_utterances = len(self.agent.utterance_list)
        if num_utterances == 0:
            logger.info(f"  {self.agent.name} has not spoken yet.")
        else:
            logger.info(f"  {self.agent.name} has spoken for {num_utterances} utterance(s).")

        if self.to_watch:
            logger.info("   Watching the following layers:\n")
            for layer, key in self.to_watch.items():
                logger.info(f"  • {layer}  →  '{key}'")
            logger.info("")

        instruction_recap = self.find_instructs(verbose=False)
        num_instructs = len(instruction_recap)

        logger.info(f"  Found {num_instructs} instruction(s) in the system messages.")

        for match in instruction_recap:
            logger.info(f"\nInstruction found at utterance index {match['index']}:\n{match['content']}\n")

    def find_instructs(self, verbose=False):
        """
        Return a list of dicts with keys 'index' and 'content' for each SystemMessage (excluding the first memory)
        found in the agent's utterance_list.
        If verbose is True, also print each.
        """
        matches = []

        if not self.agent or not self.agent.utterance_list:
            return matches

        for utt_data in self.agent.utterance_list:
            utt = utt_data['output_tokens'][0]
            mem = utt_data.get('mem', [])[1:]  # Skip the first memory item

            for msg in mem:
                if isinstance(msg, SystemMessage):
                    match = {"index": utt.utterance_index, "content": msg.content}
                    if verbose:
                        logger.info(f"\n[SystemMessage in utterance index {match['index']}]:\n{match['content']}\n")
                    matches.append(match)
                    break  # Only one SystemMessage per utterance is sufficient

        return matches


class InspectionUtterance(Inspector):
    def __init__(self, utterance, agent):
        super().__init__(to_watch=None)
        self.utterance = utterance
        self.tokens = utterance['tokens']
        self.text = utterance['text']
        self.agent = agent
        # Store utterance_index if present
        self.utterance_index = utterance.get('utterance_index', 0)

    def __iter__(self):
        for idx, token in enumerate(self.tokens):
            yield InspectionUnit(token, self.agent, self, idx, utterance_index=self.utterance_index)

    def __str__(self):
        return self.text

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                InspectionUnit(token, self.agent, self, i, utterance_index=self.utterance_index)
                for i, token in enumerate(self.tokens[index])
            ]
        return InspectionUnit(
            self.tokens[index], self.agent, self, index, utterance_index=self.utterance_index
        )


class InspectionUnit(Inspector):
    def __init__(self, token, agent, utterance, token_index, utterance_index):
        super().__init__(to_watch=None)
        """ Represents a single token at the utterance level """
        self.token = token
        self.token_index = token_index
        self.utterance = utterance  # Reference to parent utterance
        self.agent = agent
        self.utterance_index = utterance_index

    def __iter__(self):
        # Not iterable, represents a single token
        raise TypeError("InspectionUnit is not iterable")

    def __len__(self):
        # Return the number of tokens in the parent utterance
        return len(self.utterance.tokens)

    def __str__(self):
        # Return the token string directly
        return self.token if isinstance(self.token, str) else str(self.token)

    def __getitem__(self, key):
        # Fetch the representation for this token from self.agent.representation_cache
        if not hasattr(self.agent, 'representation_cache'):
            raise KeyError("Agent has no representation_cache.")
        rep_cache = self.agent.representation_cache
        # Directly use utterance_index (assume always populated)
        rep_tensor = rep_cache[self.utterance_index][key]
        if hasattr(rep_tensor, '__getitem__'):
            return rep_tensor[self.token_index]
        return rep_tensor

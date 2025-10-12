"""
This submodule provides classes and hooks for inspecting and interpreting the internal representations
of PyTorch-based language models during forward passes. It enables the registration of hooks on specific
model layers to capture token-level and response-level information, facilitating analysis of model behavior
and interpretability. The module is designed to work with conversational agents and integrates with
tokenizers and memory structures, supporting the extraction and inspection of tokens, representations,
and system instructions across responses.

Typical usage involves attaching one or more `Inspector` objects to an agent, accumulating response and token data
during inference, and providing interfaces for downstream interpretability and analysis tasks.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Séverin Baroudi <severin.baroudi@lis-lab.fr>, Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import torch
import logging
import numpy as np

from functools import partial
from typing import Optional, Any
from collections import defaultdict
from langchain_core.messages import SystemMessage
from typing import Dict, List, Union, Callable, Tuple

from .base import BaseSteerer, BaseHook


logger = logging.getLogger(__name__)


def _default_steering_function(activation, direction, strength=1, op="+"):
    """
    Default steering function applied to token-level activations.

    Behavior:

      - op="+" : additive shift along direction (scaled by strength).
      - op="-" : removes (projects out) the component of activation along direction.

    :param activation: Activation tensor for current token (..., d_act)
    :type activation: torch.Tensor
    :param direction: Steering direction tensor (d_act,)
    :type direction: torch.Tensor
    :param strength: Scalar multiplier for the steering effect.
    :type strength: float
    :param op: "+" to add direction, "-" to subtract its projection.
    :type op: str
    :return: Modified activation tensor.
    :rtype: torch.Tensor
    """
    if activation.device != direction.device:
        direction = direction.to(activation.device)
    if op == "-":
        # Project activation onto direction
        direction = direction / direction.norm()
        proj_coeff = torch.matmul(activation, direction)  # (...,)
        proj = proj_coeff.unsqueeze(-1) * direction  # (..., d_act)
        # Force activations to be orthogonal to the direction
        return activation - proj
    else:
        return activation + direction * strength


class DirectionSteerer(BaseSteerer):
    """Concrete Steerer binding a direction vector for additive or subtractive steering.

    Example:

        .. code-block:: python

            import torch
            from sdialog.agents import Agent
            from sdialog.interpretability import Inspector, DirectionSteerer

            agent = Agent()
            insp = Inspector(target='model.layers.5.post_attention_layernorm')
            agent = agent | insp

            direction = torch.randn(4096)  # Random direction in activation space
            steer = DirectionSteerer(direction)

            # Add the direction (push activations along vector)
            insp = steer + insp
            # Or remove its projection:
            insp = steer - insp

            agent("Test prompt")  # steering applied during generation

    :param direction: Direction vector (torch.Tensor or numpy array).
    :type direction: Union[torch.Tensor, np.ndarray]
    :param inspector: Optional Inspector to bind immediately.
    :type inspector: Optional[Inspector]
    """
    def __init__(self, direction, inspector=None):
        self.direction = direction
        self.inspector = inspector

    def __add__(self, inspector: "Inspector"):
        """Attach this direction as additive steering to an Inspector via + operator.

        :param inspector: Target Inspector instance receiving this steering direction.
        :type inspector: Inspector
        :return: The inspector (for chaining).
        :rtype: Inspector
        """
        return self._add_steering_function(inspector, _default_steering_function,
                                           direction=self.direction, op="+")

    def __sub__(self, inspector):
        """Attach this direction as subtractive / projection-removal steering via - operator.

        :param inspector: Target Inspector instance receiving this steering direction.
        :type inspector: Inspector
        :return: The inspector (for chaining).
        :rtype: Inspector
        """
        return self._add_steering_function(inspector, _default_steering_function,
                                           direction=self.direction, op="-")


class ResponseHook(BaseHook):
    """
    A hook class for capturing response-level information.
    This class is not meant to be used directly, but rather used by the `Inspector` class.

    Example:

        .. code-block:: python

            from sdialog.agents import Agent
            from sdialog.interpretability import ResponseHook

            agent = Agent()
            hook = ResponseHook(agent)

            hook.response_begin(agent.memory_dump())
            agent("Hi there")
            hook.response_end()

            print("Generation info:", hook.responses[-1]['output'][0].response)
            # Output:
            # {'input_ids': tensor([ 271, 9906,   11, 1268,  649]),
            # 'text': 'Hello, how can',
            # 'tokens': ['<bos>', 'Hello', ',', 'how', 'can'],
            # 'response_index': 0}
            hook.remove()

    :param agent: Agent instance owning this hook.
    :type agent: Agent
    :meta private:
    """
    def __init__(self, agent):
        super().__init__('model.embed_tokens', self._hook, agent=agent)
        self.responses = []
        self.current_response_ids = None
        self.agent = agent
        self.register(agent.base_model)

    def _hook(self, module, input, output):
        """
        Forward hook capturing input token IDs at embedding layer.

        :param module: Embedding module.
        :type module: torch.nn.Module
        :param input: Forward input tuple (expects first element = token ids).
        :type input: tuple
        :param output: Embedding output (ignored for storage).
        :type output: torch.Tensor
        """
        input_ids = input[0].detach().cpu()
        self.register_response_tokens(input_ids)

    def reset(self):
        """Clears response list, representation cache and current token accumulator."""
        self.responses.clear()
        self.agent._hook_response_act.clear()
        self.agent._hook_response_act.update(defaultdict(lambda: defaultdict(list)))
        self.current_response_ids = None  # Now a tensor

    def response_begin(self, memory):
        """
        Starts tracking a new generated response.

        :param memory: Snapshot of agent memory at response start.
        :type memory: list
        """
        self.responses.append({'mem': memory, 'output': []})
        self.current_response_ids = None

    def response_end(self):
        """
        Finalizes current response: decodes tokens, creates InspectionResponse, stores it.
        """
        token_list = self.current_response_ids.squeeze()
        token_list = token_list.tolist()
        text = self.agent.tokenizer.decode(token_list, skip_special_tokens=False)
        tokens = self.agent.tokenizer.convert_ids_to_tokens(token_list)

        # No longer create an InspectionToken here; just store the tokens list
        response_dict = {
            'input_ids': self.current_response_ids,
            'text': text,
            'tokens': tokens,
            'response_index': len(self.responses) - 1
        }
        # Append an InspectionResponse instance instead of a dict
        current_response_inspector = InspectionResponse(response_dict, agent=self.agent)
        self.responses[-1]['output'].append(current_response_inspector)

    def register_response_tokens(self, input_ids):
        """
        Accumulates only the newest generated token IDs across forward passes.

        :param input_ids: Tensor of token ids (batch, seq_len).
        :type input_ids: torch.Tensor
        """
        # Accumulate token IDs as a tensor (generated tokens only)
        if self.current_response_ids is None:
            self.current_response_ids = input_ids[..., -1]
        else:
            self.current_response_ids = torch.cat([self.current_response_ids, input_ids[..., -1]], dim=-1)


class ActivationHook(BaseHook):
    """
    A BaseHook for capturing representations from a specific model layer.
    This class is not meant to be used directly, but rather used by the `Inspector` class.

    Example:

        .. code-block:: python

            from sdialog.agents import Agent
            from sdialog.interpretability import ResponseHook, ActivationHook

            agent = Agent()

            resp_hook = ResponseHook(agent)
            act_hook = ActivationHook(
                cache_key="my_target",
                layer_key="model.layers.10.post_attention_layernorm",
                agent=agent,
                response_hook=resp_hook
            )
            resp_hook.register(agent.base_model)
            act_hook.register(agent.base_model)

            resp_hook.response_begin(agent.memory_dump())
            agent("Hello world!")
            resp_hook.response_end()

            # Cached target activations for first (and only) response
            acts = agent._hook_response_act[0]["my_target"][0]  # response index 0, token index 0

            print(acts)
            # Output:
            # tensor([[ 0.1182,  0.1152, -0.0045,  ...,  0.1836, -0.0549, -0.1924]], dtype=torch.bfloat16)

    :param cache_key: Key under which layer outputs will be stored.
    :type cache_key: Union[str, int]
    :param layer_key: Layer name (found in model.named_modules()).
    :type layer_key: str
    :param agent: the target Agent object.
    :type agent: Agent
    :param response_hook: ResponseHook instance (for current response index).
    :type response_hook: ResponseHook
    :param steering_function: Optional single function or list applied in-place to last token activation.
    :type steering_function: Optional[Union[Callable, List[Callable]]]
    :param steering_interval: (min_token, max_token) steering window (max=-1 => unbounded).
    :type steering_interval: Tuple[int, int]
    :meta private:
    """
    def __init__(self, cache_key, layer_key, agent, response_hook,
                 steering_function=None, steering_interval=(0, -1)):
        super().__init__(layer_key, self._hook, agent=None)
        self.cache_key = cache_key
        self.agent = agent
        self.response_hook = response_hook
        self.steering_function = steering_function  # Store the optional function
        self.steering_interval = steering_interval
        self._token_counter_steering = 0
        self.register(agent.base_model)

        # Initialize the nested cache
        _ = self.agent._hook_response_act[len(self.response_hook.responses)][self.cache_key]

    def _hook(self, module, input, output):
        """
        Hook to extract and store model representation from the output.

        :param module: The hooked layer/module.
        :type module: torch.nn.Module
        :param input: Forward pass inputs.
        :type input: tuple
        :param output: Layer output (tensor or tuple containing tensor).
        :type output: Union[torch.Tensor, tuple]
        :return: Possibly modified output (after optional steering).
        :rtype: Union[torch.Tensor, tuple]
        :raises TypeError: If output main tensor is not a torch.Tensor.
        """
        response_index = len(self.response_hook.responses) - 1

        # Extract the main tensor from output if it's a tuple or list
        output_tensor = output[0] if isinstance(output, (tuple, list)) else output

        # Ensure output_tensor is a torch.Tensor before proceeding
        if not isinstance(output_tensor, torch.Tensor):
            raise TypeError(f"Expected output to be a Tensor, got {type(output_tensor)}")

        # Store representation only if the second dimension is 1
        if output_tensor.ndim >= 2:
            if output_tensor.shape[1] > 1:
                self._token_counter_steering = 0  # Reset counter if more than one token
            min_token, max_token = self.steering_interval
            steer_this_token = (
                self._token_counter_steering >= min_token
                and (max_token == -1 or self._token_counter_steering < max_token)
            )

            self.agent._hook_response_act[response_index][self.cache_key].append(
                output_tensor[:, -1, :].detach().cpu()
            )

            if steer_this_token:
                # Now apply the steering function, if it exists
                if self.steering_function is not None:
                    if type(self.steering_function) is list:
                        for func in self.steering_function:
                            output_tensor[:, -1, :] = func(output_tensor[:, -1, :])
                    elif callable(self.steering_function):
                        output_tensor[:, -1, :] = self.steering_function(output_tensor[:, -1, :])

            self._token_counter_steering += 1

        if isinstance(output, (tuple, list)):
            output = (output_tensor, *output[1:]) if isinstance(output, tuple) else [output_tensor, *output[1:]]
        else:
            output = output_tensor

        return output


class Inspector:
    """
    Main class to manage layer hooks, cached activations, and optional steering functions for an Agent.

    Example:

        .. code-block:: python

            from sdialog.agents import Agent
            from sdialog.interpretability import Inspector

            agent = Agent()
            insp = Inspector(target='model.layers.2.post_attention_layernorm')
            agent = agent | insp  # pipe attach

            agent("Explain gravity briefly.")  # Generates first response
            agent("Sounds cool!")  # Generates second response

            print("Num responses captured:", len(insp))
            print("Last response, first token string:", insp[-1][0])
            print("Last response, first token activation:", insp[-1][0].act)
            # Output:
            # Num responses captured: 2
            # Last response, first token string: <bos>
            # Last response, first token activation:
            # tensor([[-0.0109, -0.1128, -0.1216,  ..., -0.0157,  0.2100, -0.2637]])

    :param target: Mapping (cache_key->layer_name) or list / single layer name (optional).
                   If None, no hooks are added until add_hooks/add_agent is called. Defaults to None.
    :type target: Union[Dict, List[str], str, None]
    :param agent: Agent instance to attach to (optional). If provided with a non-empty target,
                  hooks are registered immediately. Defaults to None.
    :type agent: Optional[Agent]
    :param steering_function: Initial steering function or list of functions (optional).
                              Applied to token activations during generation. Defaults to None.
    :type steering_function: Optional[Union[Callable, List[Callable]]]
    :param steering_interval: (min_token, max_token) steering window (optional). Defaults to (0, -1),
                              where -1 means no upper bound.
    :type steering_interval: Optional[Tuple[int, int]]
    """
    def __init__(self,
                 target: Union[Dict, List[str], str] = None,
                 agent: Optional[Any] = None,
                 steering_function: Optional[Callable] = None,
                 steering_interval: Optional[Tuple[int, int]] = (0, -1)):
        """
        Initializes the Inspector with optional target layers, agent, and steering functions.
        """
        if target is None:
            target = {}
        elif isinstance(target, str):
            target = {0: target}
        elif isinstance(target, list):
            target = {i: t for i, t in enumerate(target)}
        elif not isinstance(target, dict):
            raise ValueError("Target must be a dict, list, or string.")
        self.target = target
        self.agent = agent
        self.steering_function = steering_function
        self._steering_strength = None
        self.steering_interval = steering_interval

        if self.agent is not None and self.target:
            self.agent._add_activation_hooks(self.target, steering_function=self.steering_function,
                                             steering_interval=self.steering_interval)

    def __len__(self):
        """Return number of completed responses captured so far."""
        return len(self.agent._hooked_responses)

    def __iter__(self):
        """Iterate over InspectionResponse objects (one per response)."""
        return (utt['output'][0] for utt in self.agent._hooked_responses)

    def __getitem__(self, index):
        """Return the InspectionResponse at given index.

        :param index: Response index (0-based).
        :type index: int
        :return: The InspectionResponse object.
        :rtype: InspectionResponse
        """
        return self.agent._hooked_responses[index]['output'][0]

    def __add__(self, other):
        """
        Add steering (+direction) when other is a vector, or delegate if other is Inspector.

        :param other: Inspector or direction vector.
        :type other: Union[Inspector, torch.Tensor, np.ndarray]
        :return: Self.
        :rtype: Inspector
        """
        if isinstance(other, Inspector):
            return other + self
        # If 'other' is a direction vector...
        elif isinstance(other, torch.Tensor) or isinstance(other, np.ndarray):
            if isinstance(other, np.ndarray):
                other = torch.from_numpy(other)
            self.__add_default_steering_function__(other, "+")
        return self

    def __sub__(self, other):
        """
        Add subtractive steering (-direction) when other is a vector.

        :param other: Inspector or direction vector.
        :type other: Union[Inspector, torch.Tensor, np.ndarray]
        :return: Self.
        :rtype: Inspector
        """
        if isinstance(other, Inspector):
            return other - self
        # If 'other' is a direction vector...
        elif isinstance(other, torch.Tensor) or isinstance(other, np.ndarray):
            if isinstance(other, np.ndarray):
                other = torch.from_numpy(other)
            self.__add_default_steering_function__(other, "-")
        return self

    def __mul__(self, value):
        """
        Set strength for next steering function (or modify last if possible).

        :param value: Numeric strength.
        :type value: float
        :return: Self.
        :rtype: Inspector
        """
        if isinstance(value, (float, int)):
            if self.steering_function is not None and isinstance(self.steering_function, list) and \
               len(self.steering_function) > 0:
                last_func = self.steering_function[-1]
                func_obj = last_func
                while isinstance(func_obj, partial):
                    func_obj = func_obj.func
                func_code = getattr(func_obj, "__code__", None)
                if func_code and "strength" in func_code.co_varnames:
                    self.steering_function[-1] = partial(last_func, strength=value)
                else:
                    self._steering_strength = value
            else:
                self._steering_strength = value
        return self

    def __add_default_steering_function__(self, direction, op):
        """
        Internal helper to wrap default_steering_function with direction/op.

        :param direction: Direction vector.
        :type direction: torch.Tensor
        :param op: "+" or "-".
        :type op: str
        :return: Self.
        :rtype: Inspector
        """
        kwargs = {
            'direction': direction,
            'op': op
        }
        if self._steering_strength is not None:
            kwargs["strength"] = self._steering_strength
        self.add_steering_function(partial(_default_steering_function, **kwargs))
        return self

    def add_agent(self, agent):
        """
        Attach an Agent after construction and (re)register hooks if target specified.

        :param agent: Agent instance.
        :type agent: Agent
        """
        self.agent = agent
        if self.target:
            self.agent._add_activation_hooks(self.target,
                                             steering_function=self.steering_function,
                                             steering_interval=self.steering_interval)

    def add_steering_function(self, steering_function):
        """
        Adds a steering function to the inspector's list of functions.

        :param steering_function: Callable accepting activation tensor.
        :type steering_function: Callable
        """
        if not isinstance(self.steering_function, list):
            if callable(self.steering_function):
                self.steering_function = [self.steering_function]
            else:
                self.steering_function = []
        self.steering_function.append(steering_function)
        if self._steering_strength is not None:
            self._steering_strength = None  # Reset after adding the steering function

    def add_hooks(self, target):
        """
        Adds hooks to the agent's model based on the provided target mapping.

        :param target: Dict mapping cache_key -> layer_name to append.
        :type target: Dict
        :raises ValueError: If no agent is attached.
        """
        if self.agent is None:
            raise ValueError("No agent assigned to Inspector.")

        # Append to existing target instead of replacing
        self.target.update(target)

        self.agent._add_activation_hooks(target, steering_function=self.steering_function)

    def recap(self):
        """
        Prints and returns the current hooks assigned to the inspector's agent.
        Also prints the 'target' mapping in a clean, readable format.
        Includes any found instructions across responses.
        """
        if self.agent is None:
            logger.warning("No agent is currently assigned.")
            return None

        num_responses = len(self.agent._hooked_responses)
        if num_responses == 0:
            logger.info(f"  {self.agent.name} has not spoken yet.")
        else:
            logger.info(f"  {self.agent.name} has spoken for {num_responses} response(s).")

        if self.target:
            logger.info("   Watching the following layers:\n")
            for layer, key in self.target.items():
                logger.info(f"  • {layer}  →  '{key}'")
            logger.info("")

        instruction_recap = self.find_instructs(verbose=False)
        num_instructs = len(instruction_recap)

        logger.info(f"  Found {num_instructs} instruction(s) in the system messages.")

        for match in instruction_recap:
            logger.info(f"\nInstruction found at response index {match['index']}:\n{match['content']}\n")

    def find_instructs(self, verbose=False):
        """
        Return list with 'index' and 'content' for each SystemMessage (excluding first memory)
        found in the agent's memory. If verbose is True, also print each.

        :param verbose: If True, logs each found instruction.
        :type verbose: bool
        :return: List of dicts with keys 'index' and 'content'.
        :rtype: List[Dict[str, Union[int, str]]]
        """
        matches = []

        if not self.agent or not self.agent._hooked_responses:
            return matches

        for utt_data in self.agent._hooked_responses:
            utt = utt_data['output'][0]
            mem = utt_data.get('mem', [])[1:]  # Skip the first memory item

            for msg in mem:
                if isinstance(msg, SystemMessage):
                    match = {"index": utt.response_index, "content": msg.content}
                    if verbose:
                        logger.info(f"\n[SystemMessage in response index {match['index']}]:\n{match['content']}\n")
                    matches.append(match)
                    break  # Only one SystemMessage per response is sufficient

        return matches


class InspectionResponse:
    """
    Container exposing tokens of a single generated response for per-token inspection.
    This class is not meant to be used directly, but rather used by the `ResponseHook` class.

    :param response: Dict with keys (tokens, text, input_ids, response_index).
    :type response: dict
    :param agent: Parent agent.
    :type agent: Agent
    :meta private:
    """
    def __init__(self, response, agent):
        self.response = response
        self.tokens = response['tokens']
        self.text = response['text']
        self.agent = agent
        # Store response_index if present
        self.response_index = response.get('response_index', 0)

    def __iter__(self):
        """Yield InspectionToken objects for each token."""
        for idx, token in enumerate(self.tokens):
            yield InspectionToken(token, self.agent, self, idx, response_index=self.response_index)

    def __str__(self):
        """Return decoded response text."""
        return self.text

    def __len__(self):
        """Number of tokens in this response."""
        # Return the number of tokens in the response
        return len(self.tokens)

    def __getitem__(self, index):
        """Return token InspectionToken or list of them (slice).
        :param index: Token index or slice.
        :type index: Union[int, slice]
        :rtype: Union[InspectionToken, List[InspectionToken]]
        """
        if isinstance(index, slice):
            return [
                InspectionToken(token, self.agent, self, i, response_index=self.response_index)
                for i, token in enumerate(self.tokens[index])
            ]
        return InspectionToken(
            self.tokens[index], self.agent, self, index, response_index=self.response_index
        )


class InspectionToken:
    """
    Represents a single token inside a response; accessor for its layer activations.
    This class is not meant to be used directly, but rather used by the :class:`InspectionResponse` class.

    :param token: Token string (or id) at this position.
    :type token: Union[str, int]
    :param agent: Parent Agent.
    :type agent: Agent
    :param response: Parent InspectionResponse.
    :type response: InspectionResponse
    :param token_index: Position of token in response.
    :type token_index: int
    :param response_index: Index of response in dialogue sequence.
    :type response_index: int
    :meta private:
    """
    def __init__(self, token, agent, response, token_index, response_index):
        self.token = token
        self.token_index = token_index
        self.response = response  # Reference to parent response
        self.agent = agent
        self.response_index = response_index

    @property
    def act(self):
        """
        Return the activation(s) for this token across all hooked targets.

        Behavior:
          - Multiple cache keys => returns self (indexable by cache key).
          - Single cache key => returns activation tensor.
        :raises KeyError: If representation cache missing.
        """
        if not hasattr(self.agent, '_hook_response_act'):
            raise KeyError("Agent has no _hook_response_act.")
        # Directly use response_index (assume always populated)
        rep_tensor = self.agent._hook_response_act[self.response_index]
        return self if len(rep_tensor) > 1 else self[next(iter(rep_tensor))]

    def __iter__(self):
        """Not iterable (single token)."""
        raise TypeError("InspectionToken is not iterable")

    def __str__(self):
        """Return the token as string."""
        return self.token if isinstance(self.token, str) else str(self.token)

    def __getitem__(self, key):
        """
        Get activation tensor for this token at given cache key.

        :param key: Cache key (layer identifier used when hooking).
        :type key: Union[str, int]
        :return: Activation tensor for this token.
        :rtype: torch.Tensor
        :raises KeyError: If cache or key missing.
        """
        # Fetch the representation for this token from self.agent._hook_response_act
        if not hasattr(self.agent, '_hook_response_act'):
            raise KeyError("Agent has no _hook_response_act.")
        rep_cache = self.agent._hook_response_act
        # Directly use response_index (assume always populated)
        rep_tensor = rep_cache[self.response_index][key]
        if hasattr(rep_tensor, '__getitem__'):
            return rep_tensor[self.token_index]
        return rep_tensor

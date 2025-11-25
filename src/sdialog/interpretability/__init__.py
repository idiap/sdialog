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
    :param inspector: Inspector instance that owns this hook (for accessing top_k).
    :type inspector: Optional[Inspector]
    :meta private:
    """
    def __init__(self, agent, inspector=None):
        super().__init__('model.embed_tokens', self._hook, agent=agent)
        self.responses = []
        self.current_response_ids = None
        self.agent = agent
        self.inspector = inspector
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
        self.agent._hook_response_logit.clear()
        self.agent._hook_response_logit.update(defaultdict(list))
        self.current_response_ids = None  # Now a tensor

    def response_begin(self, memory):
        """
        Starts tracking a new generated response.

        :param memory: Snapshot of agent memory at response start.
        :type memory: list
        """
        self.responses.append({'mem': memory, 'output': [], 'input': []})
        self.current_response_ids = None

    def response_end(self):
        """
        Finalizes current response: decodes tokens, creates InspectionResponse, stores it.
        """
        token_list = self.current_response_ids.squeeze()
        token_list = token_list.tolist()

        system_prompt_text = self.agent.tokenizer.decode(
            token_list[:self.length_system_prompt],
            skip_special_tokens=False,
        )

        system_prompt_tokens = self.agent.tokenizer.convert_ids_to_tokens(
            token_list[:self.length_system_prompt]
        )

        response_text = self.agent.tokenizer.decode(
            token_list[self.length_system_prompt:],
            skip_special_tokens=False,
        )

        response_tokens = self.agent.tokenizer.convert_ids_to_tokens(
            token_list[self.length_system_prompt:]
        )

        # Store the system prompt dictionary (knowing its actual length)
        system_prompt_dict = {
            'input_ids': self.current_response_ids[:self.length_system_prompt],
            'text': system_prompt_text,
            'tokens': system_prompt_tokens,
            'response_index': len(self.responses) - 1,
            'length_system_prompt': self.length_system_prompt  # Needed to pass the offset for the proper indexing
        }

        # Append an InspectionResponse instance instead of a dict. We need to flag if it a system prompt or not.
        system_prompt_inspector = InspectionResponse(
            system_prompt_dict, agent=self.agent, inspector=self.inspector, is_system_prompt=True)
        self.responses[-1]['input'].append(system_prompt_inspector)

        # Store the generated response dictionary
        response_dict = {
            'input_ids': self.current_response_ids[self.length_system_prompt:],
            'text': response_text,
            'tokens': response_tokens,
            'response_index': len(self.responses) - 1
        }

        current_response_inspector = InspectionResponse(response_dict, agent=self.agent, inspector=self.inspector)
        self.responses[-1]['output'].append(current_response_inspector)

    def register_response_tokens(self, input_ids):
        """
        Accumulates only the newest generated token IDs across forward passes.

        :param input_ids: Tensor of token ids (batch, seq_len).
        :type input_ids: torch.Tensor
        """
        # Accumulate token IDs as a tensor (generated tokens only)
        if self.current_response_ids is None:
            self.current_response_ids = input_ids[0]
            self.length_system_prompt = len(input_ids[0]) - 1  # OFFSET by 1 for the first generated token.

            # The order is : print(inspector[0][0]) -> Input token
            #                print(inspector[0][0].act) -> Activation that serves to generate the new token
            #                print(inspector[0][1]) -> Output of token
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
    :param inspector: Inspector instance that owns this hook (for accessing top_k).
    :type inspector: Optional[Inspector]
    :meta private:
    """
    def __init__(self, cache_key, layer_key, agent, response_hook,
                 steering_function=None, steering_interval=(0, -1), inspector=None):
        super().__init__(layer_key, self._hook, agent=None)
        self.cache_key = cache_key
        self.agent = agent
        self.response_hook = response_hook
        self.steering_function = steering_function  # Store the optional function
        self.steering_interval = steering_interval
        self.inspector = inspector
        self._token_counter_steering = 0
        self.register(agent.base_model, self.inspector.inspect_input)

        # Initialize the nested cache
        _ = self.agent._hook_response_act[len(self.response_hook.responses)][self.cache_key]

    def _hook(self, module, input, output=None):
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
        activations = input if self.inspector.inspect_input else output
        activation_tensor = activations[0] if isinstance(activations, (tuple, list)) else activations

        # Ensure activation_tensor is a torch.Tensor before proceeding
        if not isinstance(activation_tensor, torch.Tensor):
            raise TypeError(f"Expected output to be a Tensor, got {type(activation_tensor)}")

        # Store representation only if the second dimension is 1
        if activation_tensor.ndim >= 2:
            min_token, max_token = self.steering_interval

            # Check if min_token should steer all system prompt tokens (non-integer or string "-1")
            steer_all_system_prompt = not isinstance(min_token, (int, np.integer))

            # Check if max_token should steer all generated tokens (any string or integer -1)
            steer_all_generated = (
                isinstance(max_token, str) or max_token == -1
            )

            if activation_tensor.shape[1] > 1:
                # Will store the system prompt
                # "Unbind" on dim 1 splits the tensor into tuple of tensors, "extend" adds them to _hook_response_act
                # Handle steering for system prompt tokens
                length_system_prompt = self.response_hook.length_system_prompt

                # Determine steering range for system prompt tokens
                if steer_all_system_prompt:
                    # Steer all system prompt tokens (min_token is non-integer)
                    start_system_steer = 0
                    end_system_steer = length_system_prompt  # exclusive end
                elif min_token < 0:
                    # Steer the last abs(min_token) system prompt tokens
                    # e.g., min_token=-3 means steer tokens at indices:
                    # length_system_prompt-3, length_system_prompt-2, length_system_prompt-1
                    start_system_steer = length_system_prompt + min_token  # e.g., length-3
                    end_system_steer = length_system_prompt  # exclusive end

                    # Assert that we're not trying to steer beyond system prompt boundaries
                    assert start_system_steer >= 0, (
                        f"Steering interval {self.steering_interval} tries to steer beyond system prompt "
                        f"boundary. System prompt has {length_system_prompt} tokens, but min_token={min_token} "
                        f"would start at index {start_system_steer}."
                    )
                else:
                    # No system prompt steering
                    start_system_steer = end_system_steer = 0

                # Apply steering function if specified and range is valid
                if self.steering_function is not None and start_system_steer < end_system_steer:
                    for i in range(start_system_steer, end_system_steer):
                        if type(self.steering_function) is list:
                            for func in self.steering_function:
                                activation_tensor[:, i, :] = func(activation_tensor[:, i, :])
                        elif callable(self.steering_function):
                            activation_tensor[:, i, :] = self.steering_function(activation_tensor[:, i, :])

                # Store all activations
                self.agent._hook_response_act[response_index][self.cache_key].extend(
                    activation_tensor.detach().cpu().unbind(dim=1)
                )

                self._token_counter_steering = 0  # Reset counter if more than one token

            else:
                # Single-token forward pass (subsequent generated tokens)
                # Steer if: counter >= 0 AND (steer all generated OR counter < max_token)
                steer_this_token = (
                    self._token_counter_steering >= 0
                    and (steer_all_generated or self._token_counter_steering < max_token)
                )

                # Append the last token (the newly generated token)
                # Each hook has its own cache_key (layer name), so duplicates won't occur across different layers
                # For the same layer, we rely on the fact that the hook is only called once per forward pass
                self.agent._hook_response_act[response_index][self.cache_key].append(
                    activation_tensor[:, -1, :].detach().cpu()
                )

                if steer_this_token:
                    # Now apply the steering function, if it exists
                    if self.steering_function is not None:
                        if type(self.steering_function) is list:
                            for func in self.steering_function:
                                activation_tensor[:, -1, :] = func(activation_tensor[:, -1, :])
                        elif callable(self.steering_function):
                            activation_tensor[:, -1, :] = self.steering_function(activation_tensor[:, -1, :])

                self._token_counter_steering += 1

        if isinstance(activations, (tuple, list)):
            if isinstance(activations, tuple):
                activations = (activation_tensor, *activations[1:])
            else:
                activations = [activation_tensor, *activations[1:]]
        else:
            activations = activation_tensor

        return activations


class LogitsHook(BaseHook):
    """
    A BaseHook for capturing logits from the language model head (lm_head) layer.
    This class is not meant to be used directly, but rather used by the `Inspector` class.

    :param cache_key: Key under which logits will be stored (typically "logits").
    :type cache_key: Union[str, int]
    :param layer_key: Layer name for lm_head (typically "lm_head").
    :type layer_key: str
    :param agent: the target Agent object.
    :type agent: Agent
    :param response_hook: ResponseHook instance (for current response index).
    :type response_hook: ResponseHook
    :param inspector: Inspector instance that owns this hook (for accessing top_k).
    :type inspector: Optional[Inspector]
    :meta private:
    """
    def __init__(self, cache_key, layer_key, agent, response_hook, inspector=None):
        super().__init__(layer_key, self._hook, agent=None)
        self.cache_key = cache_key
        self.agent = agent
        self.response_hook = response_hook
        self.inspector = inspector
        self.register(agent.base_model, self.inspector.inspect_input)

        # Initialize the logits cache for this response
        _ = self.agent._hook_response_logit[len(self.response_hook.responses)] = []

    def _hook(self, module, input, output):
        """
        Hook to extract and store logits from the lm_head output.

        :param module: The lm_head module.
        :type module: torch.nn.Module
        :param input: Forward pass inputs.
        :type input: tuple
        :param output: Logits tensor (batch_size, seq_len, vocab_size).
        :type output: torch.Tensor
        :return: Unmodified output.
        :rtype: torch.Tensor
        """
        response_index = len(self.response_hook.responses) - 1

        # Extract logits tensor
        if isinstance(output, (tuple, list)):
            logits_tensor = output[0]
        else:
            logits_tensor = output

        if not isinstance(logits_tensor, torch.Tensor):
            return output

        # Store logits for all forward passes (both multi-token and single-token)
        # Always extract the last token's logits, which represents the prediction for the next token
        if logits_tensor.ndim >= 2:
            # Get the current number of generated tokens to ensure we store logits in the correct order
            # The number of logits stored should match the number of generated tokens
            current_logits_count = len(self.agent._hook_response_logit[response_index])

            # For the first forward pass (multi-token), we only want to store the logits
            # for the first generated token. For subsequent forward passes (single-token),
            # we store the logits for each new token.
            # Only store if: (1) first forward pass (current_logits_count == 0), or
            # (2) single-token forward pass (logits_tensor.shape[1] == 1)
            # This ensures we store one logit entry per generated token
            if current_logits_count == 0 or logits_tensor.shape[1] == 1:
                # First forward pass (store logits for first token) or
                # single-token forward pass (store logits for new token)
                self.agent._hook_response_logit[response_index].append(
                    logits_tensor[:, -1, :].detach().cpu()
                )

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
    :param top_k: Number of top token predictions to store for each token. If None, logits are not captured.
                 If -1, all tokens in the vocabulary are returned with their logits. Defaults to None.
    :type top_k: Optional[int]
    :param lm_head_layer: Name of the language model head layer (e.g., "lm_head"). Defaults to "lm_head".
                          If the specified layer is not found, the code will attempt to auto-detect it.
    :type lm_head_layer: Optional[str]
    :param inspect_input: If True, captures activations before the layer processes them (input activations).
                         If False (default), captures activations after the layer processes them (output activations).
                         Defaults to False.
    :type inspect_input: bool
    """
    def __init__(self,
                 target: Union[Dict, List[str], str] = None,
                 agent: Optional[Any] = None,
                 steering_function: Optional[Callable] = None,
                 steering_interval: Optional[Tuple[int, int]] = (0, -1),
                 top_k: Optional[int] = None,
                 lm_head_layer: Optional[str] = "lm_head",
                 inspect_input: bool = False):
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
        self.top_k = top_k
        self.lm_head_layer = lm_head_layer
        self.inspect_input = inspect_input
        self._logits_hook = None

        if self.agent is not None and self.target:
            self.agent._add_activation_hooks(self.target, steering_function=self.steering_function,
                                             steering_interval=self.steering_interval, inspector=self)

        # Register logits hook if top_k is specified
        if self.agent is not None and self.top_k is not None:
            self._register_logits_hook()

    @property
    def input(self):
        class input_wrapper:
            def __init__(self, inspector):
                self.inspector = inspector

            def __getitem__(self, index):
                response = self.inspector.agent._hooked_responses[index]['input'][0]
                # Override the inspector reference to use this Inspector's top_k
                response.inspector = self.inspector
                return response

        return input_wrapper(self)

    def __len__(self):
        """Return number of completed responses captured so far."""
        return len(self.agent._hooked_responses)

    def __iter__(self):
        """Iterate over InspectionResponse objects (one per response)."""
        return (utt['output'][0] for utt in self.agent._hooked_responses)

    @property
    def vocab_size(self):
        """
        Return the vocabulary size of the tokenizer.

        :return: Vocabulary size (number of tokens in the tokenizer's vocabulary).
        :rtype: int
        :raises ValueError: If no agent is attached.
        """
        if self.agent is None:
            raise ValueError(
                "No agent attached to Inspector. Cannot determine vocab_size. "
                "Attach an agent to the Inspector first."
            )
        return len(self.agent.tokenizer)

    def __getitem__(self, index):
        """Return the InspectionResponse at given index.

        :param index: Response index (0-based).
        :type index: int
        :return: The InspectionResponse object with this Inspector's reference.
        :rtype: InspectionResponse
        """
        response = self.agent._hooked_responses[index]['output'][0]
        # Override the inspector reference to use this Inspector's top_k
        response.inspector = self
        return response

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

    def _register_logits_hook(self):
        """
        Register a LogitsHook to capture logits from the lm_head layer.
        Only one LogitsHook is registered per agent, even if multiple inspectors have top_k set.
        """
        if self.agent is None:
            return

        # Check if a LogitsHook already exists for this agent
        # We only need one LogitsHook per agent, regardless of how many inspectors have top_k
        if hasattr(self.agent, '_hook_logits') and self.agent._hook_logits:
            # A LogitsHook already exists, don't create another one
            # Just store a reference to the existing hook for cleanup purposes
            self._logits_hook = self.agent._hook_logits[0]
            return

        # Try to find the lm_head layer
        model = self.agent.base_model
        lm_head_key = self.lm_head_layer

        # Check if the specified layer exists
        if lm_head_key not in dict(model.named_modules()):
            logger.warning(
                f"Specified lm_head layer '{lm_head_key}' not found. "
                "Top-k predictions will not be available."
            )
            return

        # Register the logits hook
        response_hook = self.agent._hook_response_data
        if response_hook is None:
            # Ensure response hook exists with this inspector
            self.agent._set_hook_response_data(inspector=self)
            response_hook = self.agent._hook_response_data
        elif response_hook.inspector is None:
            # Update the inspector reference if not already set (needed for top_k access)
            response_hook.inspector = self

        self._logits_hook = LogitsHook(
            cache_key="logits",
            layer_key=lm_head_key,
            agent=self.agent,
            response_hook=response_hook,
            inspector=self
        )
        # Store the hook in agent's hook list for cleanup
        if not hasattr(self.agent, '_hook_logits'):
            self.agent._hook_logits = []
        self.agent._hook_logits.append(self._logits_hook)

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
                                             steering_interval=self.steering_interval,
                                             inspector=self)

        # Register logits hook if top_k is specified
        if self.top_k is not None:
            self._register_logits_hook()

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

        self.agent._add_activation_hooks(target, steering_function=self.steering_function, inspector=self)

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

        logger.info("")
        # Display top_k information if enabled
        if self.top_k is not None:
            if self.top_k == -1:
                logger.info("  Top-k predictions: Enabled (all tokens)")
            else:
                logger.info(f"  Top-k predictions: Enabled (k={self.top_k})")
            # Display vocabulary size
            vocab_size = len(self.agent.tokenizer)
            logger.info(f"  Vocabulary size: {vocab_size}")

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
    :param inspector: Inspector instance that owns this response (for accessing top_k).
    :type inspector: Optional[Inspector]
    :meta private:
    """
    def __init__(self, response, agent, inspector=None, is_system_prompt=False):
        self.response = response
        self.tokens = response['tokens']
        self.text = response['text']
        self.agent = agent
        self.inspector = inspector
        self.response_index = response.get('response_index', 0)
        # Flag to distinguish input (system prompt) from output (generated) tokens
        self.is_system_prompt = is_system_prompt
        # Store length_system_prompt for calculating activation indices (only for system prompt responses)
        self.length_system_prompt = response.get('length_system_prompt', 0)

    def __iter__(self):
        """Yield InspectionToken objects for each token."""
        for idx, token in enumerate(self.tokens):
            yield InspectionToken(
                token, self.agent, self, idx,
                response_index=self.response_index, is_system_prompt=self.is_system_prompt, inspector=self.inspector
            )

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
                InspectionToken(
                    token, self.agent, self, i,
                    response_index=self.response_index, is_system_prompt=self.is_system_prompt, inspector=self.inspector
                )
                for i, token in enumerate(self.tokens[index])
            ]
        return InspectionToken(
            self.tokens[index], self.agent, self, index,
            response_index=self.response_index, is_system_prompt=self.is_system_prompt, inspector=self.inspector
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
    :param inspector: Inspector instance that owns this token (for accessing top_k).
    :type inspector: Optional[Inspector]
    :meta private:
    """
    def __init__(self, token, agent, response, token_index, response_index, is_system_prompt=False, inspector=None):
        self.token = token
        self.token_index = token_index
        self.response = response  # Reference to parent response
        self.agent = agent
        self.response_index = response_index
        self.inspector = inspector
        # Flag to distinguish input (system prompt) from output (generated) tokens
        self.is_system_prompt = is_system_prompt

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

        # If inspector is available, use layer name from target (since we store by layer name)
        if self.inspector is not None and self.inspector.target:
            if len(self.inspector.target) > 1:
                # Multiple targets - return self for indexing by layer name
                return self
            else:
                # Single target - use layer name directly as cache key
                layer_name = next(iter(self.inspector.target.values()))
                return self[layer_name]

        # Fallback: return self if multiple cache keys, otherwise return the single activation
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
        rep_tensor_dict = rep_cache[self.response_index]

        # Check if key is an original cache key from target dict
        if key in self.inspector.target:
            # Map original cache key to layer name (which is the actual cache key)
            layer_name = self.inspector.target[key]
            key = layer_name

        # Get the activation tensor for this cache key
        rep_tensor = rep_tensor_dict[key]

        # Calculate activation index: system prompt tokens come first, then generated tokens
        if self.is_system_prompt:
            # For system prompt, normalize negative index relative to system prompt length
            if self.token_index < 0:
                activation_index = self.response.length_system_prompt + self.token_index
            else:
                activation_index = self.token_index
        else:
            # For generated tokens: positive indices need offset
            input_response = self.agent._hooked_responses[self.response_index]['input'][0]
            if self.token_index < 0:
                # Negative index: Python indexing from end of tensor (last generated token is at the end)
                activation_index = self.token_index
            else:
                # Positive index: add system prompt offset
                activation_index = input_response.length_system_prompt + self.token_index

        if hasattr(rep_tensor, '__getitem__'):
            return rep_tensor[activation_index]
        return rep_tensor

    @property
    def top_k(self):
        """
        Return top-k predicted tokens and their probabilities that led to this token being generated.

        Returns a list of tuples (token_string, probability) sorted by probability (descending).
        The number of predictions (k) is determined by the Inspector's top_k parameter.
        If top_k=-1, returns all tokens in the vocabulary with their probabilities.

        Note: System prompt tokens do not have top_k predictions (they are not generated).
        For generated tokens, this returns the probabilities that predicted this token.

        :return: List of (token_string, probability) tuples.
        :rtype: List[Tuple[str, float]]
        :raises KeyError: If logits are not available (logits hook not registered).
        :raises ValueError: If called on a system prompt token.
        """
        # System prompt tokens don't have predictions
        if self.is_system_prompt:
            raise ValueError("top_k is not available for system prompt tokens (they are not generated).")

        # Get top_k value from Inspector
        if self.inspector.top_k is None:
            raise KeyError("top_k is not set on the Inspector. Set top_k when creating the Inspector.")
        k = self.inspector.top_k

        # Get logits for this token from the separate logits cache
        if self.response_index not in self.agent._hook_response_logit:
            raise KeyError(
                "Logits not available for this response. "
                "The logits hook may not have captured any logits yet."
            )

        logits_tensor = self.agent._hook_response_logit[self.response_index]

        # For generated tokens, logits are stored per generated token
        # logits[i] contains the predictions that led to token[i] being generated
        logits_index = self.token_index

        # Get logits for this token position
        if hasattr(logits_tensor, '__getitem__'):
            token_logits = logits_tensor[logits_index]
        else:
            raise KeyError(f"Logits tensor is not indexable at position {logits_index}")

        # Handle both 1D and 2D logits tensors
        if token_logits.ndim > 1:
            token_logits = token_logits.squeeze()

        # Apply softmax to convert logits to probabilities
        probs = torch.softmax(token_logits, dim=-1)

        # Get vocab_size using len(tokenizer) (same as Inspector.vocab_size)
        vocab_size = len(self.agent.tokenizer)

        # If k == -1, return all tokens with their probabilities
        if k == -1:
            # Get all probabilities and indices
            # Sort by probability value (descending) to get all tokens sorted
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        else:
            # Get top-k probabilities
            top_probs, top_indices = torch.topk(probs, k=min(k, vocab_size), dim=-1)
            sorted_probs = top_probs
            sorted_indices = top_indices

        # Decode tokens
        tokenizer = self.agent.tokenizer
        results = []
        # Handle both 0D and 1D results
        if sorted_probs.ndim == 0:
            sorted_probs = sorted_probs.unsqueeze(0)
            sorted_indices = sorted_indices.unsqueeze(0)
        for prob, idx in zip(sorted_probs.tolist(), sorted_indices.tolist()):
            # Decode the token ID
            token_str = tokenizer.decode([int(idx)], skip_special_tokens=False)
            results.append((token_str, float(prob)))

        return results

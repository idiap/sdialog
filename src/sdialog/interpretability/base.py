from abc import ABC
from functools import partial


class BaseSteerer(ABC):
    """
    Abstract helper enabling operator overloading for adding steering functions to an Inspector.

    This class can be used to create concrete steerers that bind specific directions or
    strengths for steering functions. For instance, the built-int `DirectionSteerer` inherits from this class.

    :meta private:
    """
    inspector = None
    strength = None

    def _add_steering_function(self, inspector, function, **kwargs):
        """
        Internal utility to attach a steering function (with deferred strength if set via * operator).
        """
        if self.strength is not None:
            func_obj = function
            while isinstance(func_obj, partial):
                func_obj = func_obj.func
            func_code = getattr(func_obj, "__code__", None)
            if func_code and "strength" in func_code.co_varnames:
                if "strength" not in kwargs:
                    kwargs["strength"] = self.strength
                self.strength = None  # Reset strength after use
        inspector.add_steering_function(partial(function, **kwargs))
        self.inspector = inspector
        return inspector

    def __mul__(self, value):
        """
        Temporarily sets strength for the next steering function addition, or updates last one if possible.

        :param value: Numeric multiplier for steering strength.
        :type value: float
        :return: Self (for chaining).
        :rtype: Steerer
        """
        if isinstance(value, (float, int)):
            if self.inspector is not None and isinstance(self.inspector.steering_function, list) and \
               len(self.inspector.steering_function) > 0:
                last_func = self.inspector.steering_function[-1]
                func_obj = last_func
                while isinstance(func_obj, partial):
                    func_obj = func_obj.func
                func_code = getattr(func_obj, "__code__", None)
                if func_code and "strength" in func_code.co_varnames:
                    self.inspector.steering_function[-1] = partial(last_func, strength=value)
                else:
                    self.strength = value
            else:
                self.strength = value
        return self


class BaseHook(ABC):
    """
    Base class for registering and managing PyTorch forward hooks on model layers.
    This class is used to create specific hook classes, like `ResponseHook` and `ActivationHook`.

    :param layer_key: Dotted module path in model.named_modules().
    :type layer_key: str
    :param hook_fn: Callable with signature (module, input, output).
    :type hook_fn: Callable
    :param agent: Owning agent (may be None for generic hooks).
    :type agent: Agent
    :meta private:
    """
    def __init__(self, layer_key, hook_fn, agent):
        self.layer_key = layer_key
        self.hook_fn = hook_fn
        self.handle = None
        self.agent = agent

    def _hook(self):
        """Placeholder hook (override in subclasses)."""
        pass

    def register(self, model, is_pre_hook=False):
        """
        Registers the hook on the given model using the layer_key.

        :param model: Model whose layer will be hooked.
        :type model: torch.nn.Module
        :param is_pre_hook: Whether to register as a pre-forward hook.
        :type is_pre_hook: bool
        :return: The hook handle.
        :rtype: Any
        """
        layer = dict(model.named_modules())[self.layer_key]
        if is_pre_hook:
            self.handle = layer.register_forward_pre_hook(self.hook_fn)
        else:
            self.handle = layer.register_forward_hook(self.hook_fn)
        return self.handle

    def remove(self):
        """
        Removes the hook if it is registered.
        """
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

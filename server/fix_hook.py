# FIX HOOK.py
#   by Lut99
#
# Created:
#   09 Nov 2023, 10:41:00
# Last edited:
#   12 Feb 2024, 13:24:33
# Auto updated?
#   Yes
#
# Description:
#   File that implements fixes for Crypten's register_forward_hook().
#

import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import crypten 
import crypten.nn as cnn
import crypten.communicator as comm

import torch
import torch.nn.functional as F

# Get some debug stuff
DEBUG = "DEBUG" in os.environ and (os.environ["DEBUG"] == "true" or os.environ["DEBUG"] == "1")
# if DEBUG:
#     import pdb


##### LIBRARY #####
T = TypeVar('T', bound='cnn.Module')

class Hook:
    """
        Wrapper around a (forward) hook that also stores some settings.
    """

    ident: int
    hook: Union[
        Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
        Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
    ]
    with_kwargs: bool
    always_call: bool

    def __init__(
        self,
        ident: int,
        hook: Union[
            Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
        ],
        with_kwargs: bool,
        always_call: bool
    ):
        """
            Constructor for the Hook that creates a new object out of it.

            # Arguments
            - `ident`: Some identifier that allows the [`HookHandle`] to properly refer to this Hook.
            - `hook`: The hook/callback to store.
            - `with_kwargs`: Whether to call it with keyword arguments given at some other place or not, not sure what to do with this.
            - `always_call`: Whether to always call it... which, I think, would be sensible? Edit: No I'm assuming these are also called when the forward pass crashes. Let's also warn for this.
        """

        self.ident = ident
        self.hook = hook
        self.with_kwargs = with_kwargs
        self.always_call = always_call

        # Hit some warnings about stuff unimplemented
        if self.always_call:
            print("WARNING: cnn.Module.register_forward_hook(): Asked to add a forward hook with `always_call` set, but the custom Crypten implementation does nothing with this.", file=sys.stderr)

class HookHandle:
    """
        Handle such that [`Hook`]s may be removed.
    """

    hooks: List[Hook]
    ident: int

    def __init__(self, hooks: List[Hook], ident: int):
        """
            Constructor for the HookHandle.

            # Arguments
            - `hooks`: The list from which to remove a handle when the time comes.
            - `ident`: The identifier of the hook to remove when the time comes.
        """

        self.hooks = hooks
        self.ident = ident

    def remove(self):
        """
            Removes the referred hook from the parent class.
        """

        # Find the index of the item we want
        idx = None
        for i, hook in enumerate(self.hooks):
            if hook.ident == self.ident:
                idx = i
                break
        if idx is None: raise RuntimeError(f"Did not find hook '{self.ident}' in list of hooks to remove")

        # Now remove that element
        self.hooks.pop(idx)



def _mean(tensor, *args, **kwargs):
    """
        Computes the mean of the values in the given tensor.
    """

    tensor.mean(*args, **kwargs)

def _sum(tensor, *args, **kwargs):
    """
        Computes the sum of the values in the given tensor.
    """

    tensor.sum(*args, **kwargs)


counter = 0
def _register_forward_hook(
    self,
    hook: Union[
        Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
        Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
    ],
    *,
    prepend: bool = False,
    with_kwargs: bool = False,
    always_call: bool = False
):
    """
        Registers new hooks that are called *after* the forward pass has commenced.
    """

    global counter

    # if DEBUG: print(f"DEBUG: utils.fix_hook._register_forward_hook{type(self)}(): Registering forward hook '{hook}'")

    # Ensure the hooks list exist for this module
    try:
        getattr(self, "_forward_hooks")
    except AttributeError:
        self._forward_hooks = []

    # Either pre- or append the hook
    hook = Hook(counter, hook, with_kwargs, always_call)
    if prepend:
        self._forward_hooks.insert(0, hook)
    else:
        self._forward_hooks.append(hook)

    # Alrighty done; return a handle to remove it later
    handle = HookHandle(self._forward_hooks, counter)
    counter += 1
    return handle

def _forward_override(forward_func):
    """
        Override for the normal crypten forward that runs its forward, then calls hooks when a result has been produced.
    """

    def inner(self, *args, **kwargs):
        # Run the normal forward
        x = forward_func(self, *args, **kwargs)

        # Get the hooks, if any
        try:
            forward_hooks = self._forward_hooks
        except AttributeError as e:
            # Nothing to do get
            if DEBUG: print(f"DEBUG: utils.fix_hook._forward_override{type(self)}(): No hooks to call")
            forward_hooks = []

        # Call the hooks
        for hook in forward_hooks:
            if DEBUG: print(f"DEBUG: utils.fix_hook._forward_override{type(self)}(): Calling hook '{hook}'")

            # Alrighty-o, call the hook! (with or without keywords, we're not picky)
            hook_fn = hook.hook
            if hook.with_kwargs:
                hook_x = hook_fn(self, args, kwargs, x)
            else:
                hook_x = hook_fn(self, args, x)

            # Propagate the result, if any
            if hook_x is not None:
                x = hook_x

        # Done
        return x
    return inner


def _conv_init(init_func):
    def inner(
        self: cnn.Conv2d,
        in_channels: Any,
        out_channels: Any,
        kernel_size: Any,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True
    ):
        """
            Override for the `cnn.Conv2d` constructor to make it remember its input channels.
        """

        # Run the normal init
        init_func(self, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        # Remember the input channels
        self.in_channels = in_channels
        # Remember the kernel size
        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

    # Return the inner
    return inner



def fix_debug():
    """
        Used in addition to other fixers to inject debug functions/whatnot in various parts of the API.
    """

    # class GradFnWrapper:
    #     pass


    # # Build a wrapper around the grad_fn that logs everything it does
    # print("DEBUG: utils.fix_hook.fix_debug(): Injecting 'torch.Tensor.grad_fn'")
    # if isinstance(torch.Tensor.grad_fn, GradFnWrapper):
    #     print("DEBUG: utils.fix_hook.fix_crypten(): Not injecting `torch.Tensor.grad_fn` as it has already been wrapped")
    # else:
    #     # Dynamically build the object as a copy
    #     grad_fn = GradFnWrapper()
    #     for func in torch.Tensor.grad_fn.__dir__():
    #         # Skip some
    #         if func in [ "__class__" ]: continue

    #         # Define the wrapper function
    #         def wrapper(name, func):
    #             def wrapper_inner(self, *args, **kwargs):
    #                 print(f"Called grad_fn.{name}")
    #                 func(self.grad_fn, *args, **kwargs)
    #             return wrapper_inner

    #         # Wrap it
    #         print(f"DEBUG: utils.fix_hook.fix_debug(): Wrapping 'torch.Tensor.grad_fn.{func}'")
    #         setattr(grad_fn, func, wrapper(func, getattr(torch.Tensor.grad_fn, func)))

    #     # Override the wrapper in the thing
    #     grad_fn.grad_fn = torch.Tensor.grad_fn
    #     torch.Tensor.grad_fn = grad_fn
    ...

def fix_deps():
    """
        Fixes missing functions in crypten and pytorch to make them more interoparable.

        Specifically, applies the following functions for this library:
        - `fix_crypten()` to `crypten`;
        - `fix_crypten_module()` to `crypten.nn.Module`; and
        - `fix_conv()` to `crypten.nn.Conv2d`.
        - `fix_torch_tensor()` to `torch.Tensor`;

        Simply call this function \*once\* and you should be good to go.
    """

    fix_crypten()
    fix_crypten_module()
    fix_crypten_conv2d()
    fix_crypten_sequential()

    # Also fix torch lel
    fix_torch_tensor()

    # And do DEBUG stuff
    if DEBUG:
        fix_debug()

def fix_crypten():
    """
        Fixes stuff like `mean` and `sum` in the given Crypten library module.

        Specifically, injects:
        - `crypten.mean()` as an alias for `CrypTensor.mean()`
        - `crypten.sum()` as an alias for `CrypTensor.sum()`
    """

    # Inject mean if it does not exist
    if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten(): Injecting 'crypten.mean()'")
    try:
        getattr(crypten, 'mean')
        if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten(): Not fixing `mean` for given library as it apparently already exists")
    except AttributeError:
        crypten.mean = _mean

    # Inject sum if it does not exist
    if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten(): Injecting 'crypten.sum()'")
    try:
        getattr(crypten, 'sum')
        if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten(): Not fixing `sum` for given library as it apparently already exists")
    except AttributeError:
        crypten.sum = _sum

def fix_crypten_module():
    """
        Fixes `register_forward_hook()` not existing in the given Crypten Module.

        Specifically, injects:
        - `cnn.Module.register_forward_hook()`
        - `cnn.Module.apply()`
        - A wrapper around `cnn.Module.forward()` to implement the hooks. The old forward is re-injected as `cnn.Module._unhooked_forward()`.
    """

    def inject_forward_in_all_subclasses(ty):
        """
            Injects a function in a type _and_ all its subclasses.
        """

        # Check if we inserted it before
        if DEBUG: print(f"DEBUG: utils.fix_hook.fix_crypten_module(): Injecting '{ty}.forward()'")
        try:
            old_func = getattr(ty, "forward")
            new_func = _forward_override(old_func)
            if old_func == new_func:
                if DEBUG: print(f"DEBUG: utils.fix_hook.fix_crypten_module(): Not fixing `forward` because it already exists")
                return
        except AttributeError:
            pass

        # Insert it in this type
        ty.forward = new_func
        # Now all its subclasses
        for sty in ty.__subclasses__():
            inject_forward_in_all_subclasses(sty)


    # Inject the functions if they haven't been injected already
    if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten_module(): Injecting 'crypten.nn.Module.register_forward_hook()'")
    try:
        getattr(cnn.Module, 'register_forward_hook')
        if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten_module(): Not fixing `register_forward_hook` for given module as it apparently already exists")
    except AttributeError:
        cnn.Module.register_forward_hook = _register_forward_hook

    inject_forward_in_all_subclasses(cnn.Module)

    if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten_module(): Injecting 'crypten.nn.Module.apply()'")
    try:
        getattr(cnn.Module, "apply")
        if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten_module(): Not fixing `apply` because it already exists")
    except AttributeError:
        cnn.Module.apply = cnn.Module._apply

def fix_crypten_conv2d():
    """
        Fixes `in_channels` not existing in the Cond2d module.

        Specifically, injects:
        - A wrapper around `cnn.Conv2d.__init__()` to make it store its input channels under `in_channels`.

        Use it like so:
        ```python
        fix_conv(cnn.Conv2d)
        ```
    """

    # Inject the wrapper around __init__
    if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten_conv2d(): Injecting 'crypten.nn.Conv2d.__init__()'")
    if getattr(cnn.Conv2d, "__init__") is not _conv_init:
        cnn.Conv2d.__init__ = _conv_init(cnn.Conv2d.__init__)
    else:
        if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten_conv2d(): Not fixing `__init__` for given module as it is already overwritten")

def fix_crypten_sequential():
    """
        Fixes `__iter__` not existing in the Sequential module.

        Specifically, injects:
        - A wrapper around `cnn.Sequential.__iter__()` as a shorthand for `cnn.Sequential.modules()`.

        Use it like so:
        ```python
        fix_conv(cnn.Conv2d)
    """

    # Implement iteration as a shorthand for the module
    if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten_module(): Injecting 'crypten.nn.Sequential.__iter__()'")
    try:
        getattr(cnn.Sequential, '__iter__')
        if DEBUG: print("DEBUG: utils.fix_hook.fix_crypten_module(): Not fixing `__iter__` because it already exists")
    except AttributeError:
        cnn.Sequential.__iter__ = cnn.Sequential.modules

def fix_torch_tensor():
    """
        Fixes torch modules such that they have `x.conv2d()` functions, as required by Crypten.
    """

    def insert_func(module, attr, func):
        """
            Function that adds the given function as the given attribute in the given module.
        """
        try:
            getattr(module, attr)
            if DEBUG: print(f"DEBUG: utils.fix_hook.fix_torch_tensor(): Not fixing `{attr}` override for given module as it apparently already exists")
        except AttributeError:
            setattr(module, attr, func)

    def _wrap_functional(func):
        """
            Why this function is necessary is beyond me, but else we get errors that the functional function gets incorrect kinda inputs (probably due to some `self`-like structure?)
        """
        def wrapper(*args, **kwargs):
            # print(",".join([f"{a} ({type(a)})" for a in args]))
            # print(",".join([f"{a}={kwargs[a]} ({type(kwargs[a])})" for a in kwargs]))
            return func(*args, **kwargs)
        return wrapper
    def _wrap_batch_norm(input, weight, bias, running_mean, running_var, training, eps, momentum, inv_var=None):
        # Reshuffle the arguments a little bit
        # NOTE: So it looks like this inv_var is an optimisation trick to remember some computation from last time. I sincerely _hope_ so, since I'm not sure how to integrate it into pytorch shit lol
        # if inv_var is not None:
        #     print(f"WARNING: utils.fix_hook.fix_torch_tensor(): Batch normalization got given `inv_var`, but this is not supported in torch's version")
        return F.batch_norm(input, running_mean=running_mean, running_var=running_var, weight=weight, bias=bias, training=training, momentum=momentum, eps=eps)

    # Fix conv2d
    if DEBUG: print("DEBUG: utils.fix_hook.fix_torch_tensor(): Injecting 'torch.Tensor.conv2d()'")
    insert_func(torch.Tensor, "conv2d", _wrap_functional(F.conv2d))
    # Fix batchnorm
    if DEBUG: print("DEBUG: utils.fix_hook.fix_torch_tensor(): Injecting 'torch.Tensor.batchnorm()'")
    insert_func(torch.Tensor, "batchnorm", _wrap_batch_norm)
    # Fix avg_pool2d
    if DEBUG: print("DEBUG: utils.fix_hook.fix_torch_tensor(): Injecting 'torch.Tensor.avg_pool2d()'")
    insert_func(torch.Tensor, "avg_pool2d", _wrap_functional(F.avg_pool2d))
    # Fix dropout
    if DEBUG: print("DEBUG: utils.fix_hook.fix_torch_tensor(): Injecting 'torch.Tensor.dropout()'")
    insert_func(torch.Tensor, "dropout", _wrap_functional(F.dropout))

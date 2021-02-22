#!/usr/bin/python
import errno
import importlib
import inspect
import logging
import math
import os
import re
import tempfile
import zipfile
import pickle
import signal
import sys
import threading
import time
import traceback
import typing
import warnings
from contextlib import contextmanager
from functools import partial
from random import random
import six
from functools import wraps

from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

T = TypeVar("T")
CallbackMapping = Mapping[Callable, Optional[Iterable[Any]]]
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)
RT = TypeVar("RT")
NT = TypeVar("T", bound=NamedTuple)


class MultipleFunctionCallError(Exception):
    pass

def run_once(
    raise_on_multiple: bool = False,
    # pyre-fixme[34]: `Variable[T]` isn't present in the function's parameters.
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    A decorator to wrap a function such that it only ever runs once
    Useful, for example, with exit handlers that could be run via atexit or
    via a signal handler. The decorator will cache the result of the first call
    and return it on subsequent calls. If `raise_on_multiple` is set, any call
    to the function after the first one will raise a
    `MultipleFunctionCallError`.
    """

    def decorator(func: Callable[..., T]) -> (Callable[..., T]):
        signal: List[T] = []

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if signal:
                if raise_on_multiple:
                    raise MultipleFunctionCallError(
                        "Function %s was called multiple times" % func.__name__
                    )
                return signal[0]
            signal.append(func(*args, **kwargs))
            return signal[0]

        return wrapper

    return decorator


class retryable(object):
    """
    A decorator for retrying a function call. The function will be called
    until it succeeds either num_tries times, or until max_duration passes.
    The function must raise an exception to signify failure. For async retryable
    look at https://fburl.com/diffusion/42n586zh

    num_tries         - number of tries before giving up
    max_duration      - number of seconds to keep retrying the function
    sleep_time        - number of seconds to sleep between each failure
    exponential       - progressively double sleep_time between retries
    max_sleep         - max value for exponential sleep_time per retry iteration
    splay_time        - randomly sleep some extra time between 0 and splay_time
    retryable_exs     - list of exceptions that will cause a retry
                        (If not specified, all exceptions will be retried.)
    passthrough_exs   - list of exceptions that would not cause a retry
    debug             - print debug information on each retry
    print_ex          - print the encountered exceptions
    print_ex_clip     - maximum length of printed exception string
    use_callbacks     - whether to call any globally defined callback functions
    private_callbacks - dictionary of callback on this retryable, could have
                        list or arguments passed to callback or None
                        for None if callback accepts arguments func args would
                        be passed
    exception_to_raise - Final exception to be raised after retries are
                         exhausted or timeout
    is_retryable_ex   - callback to examine if an exception is retryable
    force_try_number  - always append try_number kwarg
    print_func        - an alternative to the builtin print function
    use_isinstance    - use `isinstance` instead of `type ==` for checking
                        `retryable_exs` and `passthrough_exs`. The default
                        is `False` for legacy reasons. Note that in async
                        retryable this is not optional and is always `True`.

    These parameters can be specified at the time the decorator is
    declared or on each function invocation.

    If "try_number" is a keyword argument of the function or "force_try_number"
    is true then the try number 1-n will be passed in. "force_try_number" is
    useful if your wrapped function accepts **kwargs dynamic arguments.

    @author Alex Renzin <arenzin@fb.com>
    """

    _callbacks: List[Callable] = []
    _callbacks_with_args: List[Callable] = []

    @classmethod
    def add_callback(cls, func):
        """
        Sets a global callback function that will be called with the exception
        every time there is one. This is useful for logging the exceptions when
        the errors are transparent to the process due to the nature of
        retryable.
        func must be a callable that accepts one parameter.
        """
        cls._callbacks.append(func)

    @classmethod
    def add_callback_with_args(cls, func):
        """
        Similar to add_callback, but the func should be a callable that accepts
        three parameters - exception, args and kwargs
        """
        cls._callbacks_with_args.append(func)

    @classmethod
    def _call_callbacks(cls, e, args=None, kwargs=None):
        for func in cls._callbacks:
            func(e)

    @classmethod
    def _call_callbacks_args(cls, e, args, kwargs):
        for func in cls._callbacks_with_args:
            func(e, args, kwargs)

    def __init__(
        self,
        num_tries: int = 1,
        max_duration: int = 0,
        sleep_time: float = 60,
        max_sleep: int = 0,
        exponential: bool = False,
        splay_time: int = 0,
        retryable_exs: Optional[List[Any]] = None,
        passthrough_exs: Optional[List[Any]] = None,
        print_ex: bool = False,
        print_ex_clip: Optional[int] = None,
        debug: bool = False,
        use_callbacks: bool = True,
        private_callbacks: Optional[CallbackMapping] = None,
        exception_to_raise: Optional[Union[Type[Exception], Exception]] = None,
        is_retryable_ex: Optional[Callable[[Exception], bool]] = None,
        force_try_number: bool = False,
        print_func: Callable = (print),
        use_isinstance: bool = False,
    ) -> None:
        self.num_tries = num_tries
        self.max_duration = max_duration
        self.sleep_time = sleep_time
        self.max_sleep = max_sleep
        self.exponential = exponential
        self.splay_time = splay_time
        self.retryable_exs = retryable_exs or []
        self.passthrough_exs = passthrough_exs or []
        self.print_ex = print_ex
        self.print_ex_clip = print_ex_clip
        self.debug = debug
        self.use_callbacks = use_callbacks
        self.private_callbacks = private_callbacks or {}
        self.exception_to_raise = exception_to_raise
        self.is_retryable_ex = is_retryable_ex
        self.force_try_number = force_try_number
        self.print_func = print_func
        self.use_isinstance = use_isinstance

    def __call__(self, func: F) -> F:
        @wraps(func)
        def new_func(*args, **kwargs):
            # See if we need to reset defaults
            vars = {}
            for arg in [
                "num_tries",
                "max_duration",
                "sleep_time",
                "max_sleep",
                "exponential",
                "retryable_exs",
                "passthrough_exs",
                "print_ex",
                "print_ex_clip",
                "debug",
                "splay_time",
                "use_callbacks",
                "exception_to_raise",
                "is_retryable_ex",
                "print_func",
                "use_isinstance",
            ]:
                if arg in kwargs:
                    vars[arg] = kwargs[arg]
                    del kwargs[arg]
                else:
                    vars[arg] = getattr(self, arg)

            if vars["max_duration"] and vars["num_tries"] == 1:
                # Unset default
                vars["num_tries"] = 0

            start_time = time.time()
            try_num = 1
            while True:
                try:
                    if six.PY2:
                        argspec = inspect.getargspec(func).args
                    else:
                        argspec = inspect.signature(func).parameters
                    if "try_number" in argspec or self.force_try_number:
                        kwargs["try_number"] = try_num
                except TypeError:
                    # getargspec does not work with partial()
                    pass
                try:
                    # TODO: Work out why we sometimes leave a socket open here
                    return func(*args, **kwargs)
                except Exception as e:

                    # Get the source information of the function so we
                    # can log which function is failing.
                    func_file = None
                    try:
                        if inspect.isbuiltin(func):
                            func_file = "built-in"
                        else:
                            code = func.__code__
                            if code is not None:
                                func_file = "%s:%d" % (
                                    code.co_filename,
                                    code.co_firstlineno,
                                )
                            else:
                                func_file = inspect.getsourcefile(func)
                    except Exception:
                        # keep any information we may have figured out
                        # if no information then use "unknown_file".
                        if func_file is None:
                            func_file = "unknown_file"

                    # do we need to catch exceptions here if func is
                    # a strange object?
                    calling_msg = "calling function '%s()' (%s)" % (
                        getattr(
                            func, "__qualname__", getattr(func, "__name__", "unknown")
                        ),
                        func_file,
                    )

                    if vars["debug"]:
                        vars["print_func"](
                            "cur: %d, tries: %d, max_time: %d, time: %d"
                            % (
                                try_num,
                                vars["num_tries"],
                                vars["max_duration"],
                                time.time() - start_time,
                            )
                        )

                    if vars["print_ex"]:
                        if vars["debug"]:
                            msg = traceback.format_exc()
                        else:
                            msg = str(e)
                        clip = vars["print_ex_clip"]
                        if clip and len(msg) > clip:
                            msg = msg[: clip - 3] + "..."

                        vars["print_func"](
                            "caught exception %s on try #%d: %s"
                            % (calling_msg, try_num, msg)
                        )

                    if vars["use_callbacks"]:
                        if self.__class__._callbacks:
                            self.__class__._call_callbacks(e)
                        if self.__class__._callbacks_with_args:
                            self.__class__._call_callbacks_args(e, args, kwargs)

                    # Check if exception is among retryable
                    if vars["retryable_exs"] and not (
                        type(e) in vars["retryable_exs"]
                        or (
                            vars["use_isinstance"]
                            and isinstance(e, tuple(vars["retryable_exs"]))
                        )
                    ):
                        if vars["debug"]:
                            vars["print_func"](
                                "Non-retryable exception %s: %s"
                                % (calling_msg, traceback.format_exc())
                            )
                        raise

                    # Check if exception is pass-through
                    if type(e) in vars["passthrough_exs"] or (
                        vars["use_isinstance"]
                        and isinstance(e, tuple(vars["passthrough_exs"]))
                    ):
                        if vars["debug"]:
                            vars["print_func"](
                                "Pass through exception %s: %s"
                                % (calling_msg, traceback.format_exc())
                            )
                        raise

                    # Check if an exception is retryable via callback
                    if vars["is_retryable_ex"] and not vars["is_retryable_ex"](e):
                        raise

                    try_num += 1
                    failed = False
                    # Check if time expired
                    if (
                        vars["max_duration"]
                        and time.time() > start_time + vars["max_duration"]
                    ):
                        if vars["debug"]:
                            vars["print_func"]("Time expired %s" % (calling_msg))
                        failed = True

                    # Check if we've exceeded allowed tries
                    elif vars["num_tries"] and try_num > vars["num_tries"]:
                        if vars["debug"]:
                            vars["print_func"]("Tries exceeded %s" % (calling_msg))
                        failed = True

                    else:
                        if vars["sleep_time"]:
                            sleep_time = vars["sleep_time"]
                            if vars["exponential"]:
                                sleep_time *= math.pow(2, try_num - 2)
                            if vars["max_sleep"]:
                                sleep_time = min(sleep_time, vars["max_sleep"])
                            time.sleep(sleep_time)
                        if vars["splay_time"]:
                            time.sleep(random() * vars["splay_time"])

                    # call private callbacks
                    for cb, cb_args in six.iteritems(self.private_callbacks):
                        if cb_args is not None:
                            cb(e, *cb_args)
                        elif inspect.isbuiltin(cb):
                            cb(e)
                        else:
                            if six.PY2:
                                callbackspec = inspect.getargspec(cb).args
                            else:
                                callbackspec = inspect.getfullargspec(cb).args
                            # if callback has args, pass ex and func args to it
                            if len(callbackspec) == 3:
                                cb(e, args, kwargs)
                            else:
                                cb(e)

                    if failed is True:
                        if vars["debug"]:
                            vars["print_func"](
                                "Failed and raising exception %s" % (calling_msg)
                            )
                        # raising due to exhausting retries or timing out
                        if vars["exception_to_raise"]:
                            raise vars["exception_to_raise"]
                        else:
                            raise

                    if vars["debug"]:
                        vars["print_func"]("Failed but trying again %s" % (calling_msg))

        new_func.__wrapped__ = func
        return new_func


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_dir_path(relative_path):
    """Return a path for a directory in this package, extracting if necessary

    For an entire directory within the par file (zip, fastzip) or lpar
    structure, this function will check to see if the contents are extracted;
    extracting each file that has not been extracted.  It returns the path of
    a directory containing the expected contents, making sure permissions are
    correct.

    Returns a string path, throws exeption on error
    """
    return os.path.dirname(importlib.import_module(relative_path).__file__)

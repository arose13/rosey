import functools
import inspect
import warnings


class FatalProgrammingError(Exception):
    pass


class BadVCFFile(Exception):
    pass


class UnexpectedFastqPairing(Exception):
    pass


class NotUbuntuError(Exception):
    pass


class MissingRequiredDependencies(Exception):
    pass


class MissingPrerequisiteFiles(Exception):
    pass


class IllegalArgumentsException(Exception):
    pass


class BadFormulaError(Exception):
    pass


class Deprecated(object):
    def __init__(self, reason):
        if inspect.isclass(reason) or inspect.isfunction(reason):
            raise TypeError('Reason for deprecation must be supplied')
        self.reason = reason

    def __call__(self, cls_or_func):
        if inspect.isfunction(cls_or_func):
            if hasattr(cls_or_func, 'func_code'):
                _code = cls_or_func.func_code
            else:
                _code = cls_or_func.__code__
            fmt = 'Call to deprecated function or method {name} ({reason}).'
            filename = _code.co_filename
            line_number = _code.co_firstlineno + 1

        elif inspect.isclass(cls_or_func):
            fmt = 'Call to deprecated class {name} ({reason}).'
            filename = cls_or_func.__module__
            line_number = 1

        else:
            raise TypeError(type(cls_or_func))

        msg = fmt.format(name=cls_or_func.__name__, reason=self.reason)

        @functools.wraps(cls_or_func)
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn_explicit(msg, category=DeprecationWarning, filename=filename, lineno=line_number)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return cls_or_func(*args, **kwargs)

        return new_func

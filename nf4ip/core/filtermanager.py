
from cement import minimal_logger

LOG = minimal_logger(__name__)

"""Cement core filters module."""

import operator
import types
from cement.core import exc
from cement.utils.misc import minimal_logger

LOG = minimal_logger(__name__)


class FilterManager(object):
    """
    Manages the filter system to define, get, run, etc filters within the
    the Cement Framework and applications Built on Cement (tm).

    """

    def __init__(self, app):
        self.app = app
        self.__filters__ = {}

    def list(self):
        """
        List all defined filters.

        Returns:
            filters (list): List of registered filter labels.
        """
        return list(self.__filters__.keys())

    def define(self, name):
        """
        Define a filter namespace that the application and plugins can register
        filters in.

        Args:
            name (str): The name of the filter, stored as filters['name']

        Raises:
            cement.core.exc.FrameworkError: If the filter name is already
                defined

        Example:

            .. code-block:: python

                from cement import App

                with App('myapp') as app:
                    app.filter.define('my_filter_name')

        """
        LOG.debug("defining filter '%s'" % name)
        if name in self.__filters__:
            raise exc.FrameworkError("Filter name '%s' already defined!" % name)
        self.__filters__[name] = []

    def defined(self, filter_name):
        """
        Test whether a filter name is defined.

        Args:
            filter_name (str): The name of the filter.
                I.e. ``my_filter_does_awesome_things``.

        Returns:
            bool: ``True`` if the filter is defined, ``False`` otherwise.

        Example:

            .. code-block:: python

                from cement import App

                with App('myapp') as app:
                    app.filter.defined('some_filter_name'):
                        # do something about it
                        pass

        """
        if filter_name in self.__filters__:
            return True
        else:
            return False

    def register(self, name, func, weight=0):
        """
        Register a function to a filter.  The function will be called, in order
        of weight, when the filter is run.

        Args:
            name (str): The name of the filter to register too.
                I.e. ``pre_setup``, ``post_run``, etc.
            func (function): The function to register to the filter.  This is an
            *un-instantiated*, non-instance method, simple function.

        Keywork Args:
            weight (int):  The weight in which to order the filter function.

        Example:

            .. code-block:: python

                from cement import App

                def my_filter_func(app):
                    # do something with app?
                    return True

                with App('myapp') as app:
                    app.filter.define('my_filter_name')
                    app.filter.register('my_filter_name', my_filter_func)

        """
        if name not in self.__filters__:
            LOG.debug("filter name '%s' is not defined! ignoring..." % name)
            return False

        LOG.debug("registering filter '%s' from %s into filters['%s']" %
                  (func.__name__, func.__module__, name))

        # filters are as follows: (weight, name, func)
        self.__filters__[name].append((int(weight), func.__name__, func))

    def run(self, name, *args, **kwargs):
        """
        Run all defined filters in the namespace.

        Args:
            name (str): The name of the filter function.
            args (tuple): Additional arguments to be passed to the filter
                functions.
            kwargs (dict): Additional keyword arguments to be passed to the
                filter functions.

        Yields:
            The result of each filter function executed.

        Raises:
            cement.core.exc.FrameworkError: If the filter ``name`` is not
                defined

        Example:

            .. code-block:: python

                from cement import App

                def my_filter_func(app):
                    # do something with app?
                    return True

                with App('myapp') as app:
                    app.filter.define('my_filter_name')
                    app.filter.register('my_filter_name', my_filter_func)
                    for res in app.filter.run('my_filter_name', app):
                        # do something with the result?
                        pass

        """
        if name not in self.__filters__:
            raise exc.FrameworkError("filter name '%s' is not defined!" % name)
        # Will order based on weight (the first item in the tuple)
        self.__filters__[name].sort(key=operator.itemgetter(0))
        for filter in self.__filters__[name]:
            LOG.debug("running filter '%s' (%s) from %s" %
                      (name, filter[2], filter[2].__module__))
            if type(args) != tuple:
                args = (args,)
            args = filter[2](*args, **kwargs)

        if type(args) == tuple and len(args) == 1:
            #unpack tuples with one element
            return args[0]

        return args

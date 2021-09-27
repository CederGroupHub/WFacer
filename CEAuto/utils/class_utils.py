import inspect

def derived_class_factory(class_name, base_class,
                          *args, **kwargs):
    """Return an instance of derived class from a given basis class.

    Args:
        class_name (str):
            name of class
        base_class (object):
            base class of derived class sought
        *args:
            positional arguments for class constructor
        **kwargs:
            keyword arguments for class constructor

    Returns:
        object: instance of class with corresponding constructor args, kwargs
    """
    try:
        derived_class = get_subclasses(base_class)[class_name]
        instance = derived_class(*args, **kwargs)
    except KeyError:
        raise NotImplementedError(f'{class_name} is not implemented.')
    return instance


def get_subclasses(base_class):
    """Get all non-abstract subclasses of a class.

    Gets all non-abstract classes that inherit from the given base class in
    a module. This is used to obtain all the available basis functions.
    """
    sub_classes = {}
    for sub_class in base_class.__subclasses__():
        if inspect.isabstract(sub_class):
            sub_classes.update(get_subclasses(sub_class))
        else:
            sub_classes[sub_class.__name__] = sub_class
    return sub_classes

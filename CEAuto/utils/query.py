"""Define rules to query a nested task documents and dictionaries."""
import random
from warnings import warn

from pydantic import BaseModel


def query_keypath(obj, keypath):
    """Query attributes of an object along a path.

    Args:
        obj(Object|dict):
            The object to be queried.
        keypath(list[str]):
            A path of attribute names to query.
    Returns:
        Any: the queried result.
    """
    if not isinstance(keypath, (list, tuple)):
        raise ValueError("A key path must be a list or tuple!")

    # Reached the end of query.
    if len(keypath) == 0:
        return obj

    k = keypath[0]
    if isinstance(obj, (list, tuple)):
        # List needs to be pre-processed.
        if "-" not in k and not k.startswith("^"):
            warn(
                f"Object {obj} is a list, but the exact index"
                f" of the member to refer to is not specified with"
                f" id-. Will query the first member in the list."
            )
            return query_keypath(obj[0], keypath)
        elif "-" in k:
            if len(k.split("-")) != 2:
                raise ValueError(
                    "Each level of keypath cannot have more than one dash -!"
                )
            ind = int(k.split("-")[0])
            new_k = k.split("-")[1]
            return query_keypath(obj[ind], [new_k] + keypath[1:])
        # Return the corresponding property or all members as a list.
        elif k.startswith("^"):
            new_k = k[1:]
            return [query_keypath(sub, [new_k] + keypath[1:]) for sub in obj]
    elif isinstance(obj, set):
        return query_keypath(random.choice(list(obj)), keypath)

    if isinstance(obj, dict):
        if k not in obj:
            raise ValueError(f"Dictionary {obj} does not have key {k}!")
        return query_keypath(obj[k], keypath[1:])
    elif isinstance(obj, BaseModel):
        if k not in obj.__fields__:
            raise ValueError(f"Object {obj} does not have field {k}!")
        return query_keypath(getattr(obj, k), keypath[1:])
    else:
        if hasattr(obj, k):
            return query_keypath(getattr(obj, k), keypath[1:])
        raise ValueError(f"Object {obj} does not have attribute" f" {k}")


def query_name_iteratively(obj, name):
    """Query an attribute from a nested object.

    Args:
        obj(Object|dict):
            The object to be queried.
        name(str):
            The attribute name.
    Returns:
        Any: the queried result. Will always return the first one
        found at the shallowest reference level.
    """
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        else:
            queries = [query_name_iteratively(v, name) for v in obj.values()]
            # Return the first result that is not None.
            for query in queries:
                if query is not None:
                    return query
            return None
    elif isinstance(obj, BaseModel):
        if name in obj.__fields__:
            return getattr(obj, name)
        else:
            queries = [
                query_name_iteratively(getattr(obj, f), name) for f in obj.__fields__
            ]
            for query in queries:
                if query is not None:
                    return query
            return None
    elif isinstance(obj, (list, tuple, set)):
        queries = [query_name_iteratively(s, name) for s in obj]
        for query in queries:
            if query is not None:
                return query
        return None
    else:
        # TODO: This might not be smart enough but should
        #  work for most cases?
        if hasattr(obj, name):
            return getattr(obj, name)
        # This is to parse entry object.
        if hasattr(obj, "data"):
            return query_name_iteratively(obj.data, name)
        return None


def get_property_from_object(obj, query_string):
    """Get a property value from a generic nested object.

    Args:
        obj(Object):
            An object to recursively parse property from.
            A task document generated as vasp task output by atomate2.
        query_string(str):
            A string that defines the rule to query the object.
            Three special characters are reserved: ".", "-" and "^":

            Dot "." represents levels of reference:
            For example, "output.volume" means to retrieve the
            obj.output.volume. Dictionary querying is also supported,
            where "." in the query string will represent the key to
            each level.

            If a level of reference is a list or tuple:
            1, Using f"{some_ind}-" as the prefix to this level will yield
            the corresponding key/attribute of the "some_ind"'th member
            in the list.
            2, Using "^" as the prefix to this level will yield the
            corresponding key/attribute of all the members in the list and
            return them as a list in the original order.
            3, Using f"{some_ind}-" as the prefix to this level will yield
            the corresponding key/attribute of the first member
            in the list.
            Do not use "-" or "^" prefix when the corresponding level is
            not a list or tuple. If a corresponding level is a set, a random
            element will be yielded.

            For example, "calcs_reversed.0-output.outcar.magnetization.^tot"
            will give you the total magnetization on each site of the structure
            in the final ionic step, if the input object is a valid atomate2
            TaskDocument.

            If a string with no special character is given, we will iteratively
            search through each level of attributes and dict keys until the
            key/attribute with the same name as the given string is found, or all
            key/attributes have been explored.

            If you decide to use special characters, please always make sure you
            have specified the exact full path to retrieve the desired item.

        Returns:
            any: value of the queried property.
    """
    # Add more special conversion rules if needed.
    query = query_string.split(".")

    # The property name contains special character.
    if len(query) > 1:
        queried = query_keypath(obj, query)
    # A single string without special character.
    else:
        queried = query_name_iteratively(obj, query[0])

    if queried is None:
        raise ValueError(
            f"Cannot find query string {query_string}" f" in the task document!"
        )

    return queried

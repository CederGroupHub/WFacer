"""Formatting utilities."""

__author__ = 'Fengyu Xie'


def merge_dicts(ds, keep_all=False):
    """Merge dictionaries by taking union of keys.

    For duplicate keys, will take the first value as default.
    Args:
       ds(List[Dict]):
           A list of dicts to merge.
       keep_all(Boolean):
           If true, will return values as a list of all values
           in all input dicts.
           Default to false, only the first occurrence will be kept.
    Returns:
       Dict.
    """
    merged = {}
    if not keep_all:
        for d in ds:
            for k in d:
                if k not in merged:
                    merged[k] = d[k]
    else:
        for d in ds:
            for k in d:
                if k not in merged:
                    merged[k] = [d[k]]
                else:
                    merged[k].append(d[k])

    return merged

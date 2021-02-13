"""
Formatting utilities.
"""

__author__ = 'Fengyu Xie'


def merge_dicts(ds,keep_all=False):
    """
    Merge dictionaries by taking union of keys. For duplicated keys, will take
    the first value as default.
    Args:
       ds(List[Dict]):
           A list of dicts to merge.
       keep_all(Boolean):
           If true, will return values as a list of all values in all input dicts.
           If false, only the first occurence will be kept.
           Default to false.
    Returns:
       Dict.
    """
    merged = {}
    if not keep_all:
        for d in ds:
            for k in d:
                if k not in merged:
                    merged[k]=d[k]
    else:
        for d in ds:
            for k in d:
                if k not in merged:
                    merged[k]=[d[k]]
                else:
                    merged[k].append(d[k])

    return merged


# flatten and de-flatten a 2d array
def flatten_2d(lst_2d,remove_nan=True):
    """
    Sequentially flatten any 2D list into 1D, and gives a deflatten rule.
    Inputs:
        lst_2d(List of list):
            A 2D list to flatten
        remove_nan(Boolean):
            If true, will disregard all None values in lst_2d when compressing.
            Default is True
    Outputs:
        flat(List):
            Flatten list
        deflat_rule(List of int):
            Deflatten rule. Each integer speciefies the length of each sub-list
            in the original 2D list. When deflattening, will serially read and
            unpack to each sublist based on these numbers.
    """
    deflat_rule = []
    flat = []
    for sl in lst_2d:
        if remove_nan:
            sl_return = [item for item in sl if item is not None]
        else:
            sl_return = sl
        flat.extend(sl_return)
        deflat_rule.append(len(sl_return))

    return flat,deflat_rule

def deflat_2d(flat_lst,deflat_rule,remove_nan=True):
    """
    Sequentially decompress a 1D list into a 2D list based on rule.
    Inputs:
        flat(List):
            Flatten list
        deflat_rule(List of int):
            Deflatten rule. Each integer speciefies the length of each sub-list
            in the original 2D list. When deflattening, will serially read and
            unpack to each sublist based on these numbers.
        remove_nan(Boolean):
            If true, will first deflat on deflat rule, then remove all 
            None values in resulting lst_2d.
            Default is True.

    Outputs:
        lst_2d(List of list):
            Deflatten 2D list
    """
    if sum(deflat_rule)!= len(flat_lst):
        raise ValueError("Deflatten length does not match original length!")

    lst_2d = []
    it = 0
    for sl_len in deflat_rule:
        sl = []
        for i in range(it,it+sl_len):
            if flat_lst[i] is None and remove_nan:
                continue
            sl.append(flat_lst[i])
        lst_2d.append(sl)
        it = it + sl_len
    return lst_2d

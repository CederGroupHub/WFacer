"""
Formatting utilities.
"""

__author__ = 'Fengyu Xie'


# flatten and de-flatten a 2d array
def flatten_2d(2d_lst,remove_nan=True):
    """
    Sequentially flatten any 2D list into 1D, and gives a deflatten rule.
    Inputs:
        2d_lst(List of list):
            A 2D list to flatten
        remove_nan(Boolean):
            If true, will disregard all None values in 2d_lst when compressing.
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
    for sl in 2d_lst:
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
            None values in resulting 2d_lst.
            Default is True.

    Outputs:
        2d_lst(List of list):
            Deflatten 2D list
    """
    if sum(deflat_rule)!= len(flat_lst):
        raise ValueError("Deflatten length does not match original length!")

    2d_lst = []
    it = 0
    for sl_len in deflat_rule:
        sl = []
        for i in range(it,it+sl_len):
            if flat_lst[i] is None and remove_nan:
                continue
            sl.append(flat_lst[i])
        2d_lst.append(sl)
        it = it + sl_len
    return 2d_lst

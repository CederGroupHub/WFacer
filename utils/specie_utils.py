def get_oxi(ion):
    """
    This tool function helps to read the charge from a given specie(in string format).
    Inputs:
        ion: a string specifying a specie.
    """
    #print(ion)
    if ion[-1]=='+':
        return int(ion[-2]) if ion[-2].isdigit() else 1
    elif ion[-1]=='-':
        #print(ion[-2])
        return int(-1)*int(ion[-2]) if ion[-2].isdigit() else -1
    else:
        return 0

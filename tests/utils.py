from monty.json import MSONable, MontyDecoder
import json

def assert_msonable(obj, test_if_subclass=True):
    """
    Tests if obj is MSONable and tries to verify whether the contract is
    fulfilled.
    By default, the method tests whether obj is an instance of MSONable.
    This check can be deactivated by setting test_if_subclass to False.
    """
    if test_if_subclass:
        assert isinstance(obj, MSONable)
    assert obj.as_dict() == obj.__class__.from_dict(obj.as_dict()).as_dict()
    _ = json.loads(obj.to_json(), cls=MontyDecoder)

def assert_dict_equal(d1, d2):
    assert sorted(list(d1.keys())) == sorted(list(d2.keys()))
    for k in d1.keys():
        if isinstance(d1[k], dict) and isinstance(d2[k], dict):
            assert_dict_equal(d1[k], d2[k])
        else:
            if d1[k] != d2[k]:
                print("Difference in key: {}, d1: {}, d2: {}"
                      .format(k, d1[k], d2[k]))
            assert d1[k] == d2[k]

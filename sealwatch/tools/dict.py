import collections.abc


def append_features(target, source, prefix=None):
    """
    Copy features from a source dict to a target dict. Prepend a prefix to the source's keys before copying.
    :param target: (ordered) dict
    :param source: (ordered) dict
    :param prefix: append this prefix to the target key name
    :return: target
    """
    assert isinstance(target, collections.abc.Mapping), "Expected target to be a dict"
    assert isinstance(source, collections.abc.Mapping), "Expected source to be a adict"

    for source_key, value in source.items():
        if prefix is None:
            target_key = source_key
        else:
            target_key = prefix + "_" + source_key

        target[target_key] = value

    return target

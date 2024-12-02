from collections import namedtuple
import importlib
from pytracking.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "pytracking.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    # gtot
    gtot=DatasetInfo(module=pt % "gtot", class_name="GTOTDataset", kwargs=dict(dtype='rgbt')),
    # rgbt210
    rgbt210=DatasetInfo(module=pt % "rgbt210", class_name="RGBT210Dataset", kwargs=dict(dtype='rgbt')),
    # rgbt234
    rgbt234=DatasetInfo(module=pt % "rgbt234", class_name="RGBT234Dataset", kwargs=dict(dtype='rgbt')),
    # lasher_test
    lasher_test=DatasetInfo(module=pt % "lasher", class_name="LasHeRDataset", kwargs=dict(dtype='rgbt', split='test')),
    # lasher_train=DatasetInfo(module=pt % "lasher", class_name="LasHeRDataset", kwargs=dict(dtype='rgbt', split='train')),
    # vtuav
    vtuav_test_st=DatasetInfo(module=pt % "vtuav", class_name="VTUAVDataset", kwargs=dict(dtype='rgbt', split='test_ST')),
    vtuav_test_lt=DatasetInfo(module=pt % "vtuav", class_name="VTUAVDataset", kwargs=dict(dtype='rgbt', split='test_LT')),
    # vtuav_train_st=DatasetInfo(module=pt % "vtuav", class_name="VTUAVDataset", kwargs=dict(dtype='rgbt', split='train_ST')),
    # vtuav_train_lt=DatasetInfo(module=pt % "vtuav", class_name="VTUAVDataset", kwargs=dict(dtype='rgbt', split='train_LT')),
)


def load_dataset(name: str, **kwargs):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs, **kwargs)  # Call the constructor
    return dataset


def get_dataset(*args, **kwargs):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name, **kwargs).get_sequence_list())
    return dset


def get_dataset_attributes(name, mode='short', **kwargs):
    """ Get a list of strings containing the short or long names of all attributes in the dataset. """
    dset = load_dataset(name , **kwargs)
    dsets = {}
    if not hasattr(dset, 'get_attribute_names'):
        dsets[name] = get_dataset(name)
    else:
        for att in dset.get_attribute_names(mode):
            dsets[att] = get_dataset(name, attribute=att)

    return dsets
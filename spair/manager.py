import torch
from torch.utils import data as torch_data
from spair import config as cfg
instance = None

class RunManager():
    ''' Singleton class for manage all run session related configurations and manages the main run loop '''
    global_step = 0
    _dataset = None
    device = None
    writer = None
    run_args = None
    def __init__(self, run_name, dataset, device, writer, run_args, cfg):
        global instance
        assert instance is None, 'There is already a RunManager running, please do not re-initialize it'
        instance = self

        RunManager.run_name = run_name
        RunManager._dataset = dataset
        RunManager.device = device
        RunManager.writer = writer
        RunManager.run_args = run_args
        RunManager.cfg = cfg

    def iterate_data(self, max_iter = 20000):
        ''' Dataset iterator class '''
        for epoch in range(100000):
            dataloader = torch_data.DataLoader(RunManager._dataset,
                                               batch_size=cfg.BATCH_SIZE,
                                               pin_memory=True,
                                               num_workers=1,
                                               drop_last=True,
                                               )
            for batch_idx, batch in enumerate(dataloader):
                global_step = epoch * len(dataloader) + batch_idx
                RunManager.global_step = global_step
                if global_step > max_iter:
                    return
                yield global_step, batch


def get_run_manager_session():
    assert instance is not None, "Run Manager hasn't been initialized yet"
    return instance
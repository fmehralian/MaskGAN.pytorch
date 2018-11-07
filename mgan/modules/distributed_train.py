from .distributed_model import DistributedModel
import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.parallel import DataParallel

class DistributedTrain:
    def __init__(self, model, opt, device=torch.device("cuda")):
        assert(isinstance(model, DistributedModel))
        self.model = model
        self.model = self.model.to(device)
        self.distributed_model = DataParallel(model)
        self.opt = opt

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def state_dict(self):
        return self.model.state_dict()

    def __call__(self, *args, **kwargs):
        self.opt.zero_grad()
        loss, samples = self.distributed_model(*args, **kwargs)
        loss = loss.mean()
        loss.backward()
        self.opt.step()
        return (loss.item(), samples)

    def eval(self, *args, **kwargs):
        with torch.no_grad():
            loss, _ = self.model(*args, **kwargs)
            loss = loss.mean()
            return loss.item()
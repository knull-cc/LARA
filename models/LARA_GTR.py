from models.GTR import Model as GTRModel
from models.LARA_DLinear import Model as LARABaseModel


class Model(LARABaseModel):
    """LARA adapter using GTR as the host forecaster."""

    def __init__(self, configs):
        host_checkpoint = getattr(configs, "lara_host_ckpt", "")
        setattr(configs, "lara_host_ckpt", "")
        super().__init__(configs)
        setattr(configs, "lara_host_ckpt", host_checkpoint)

        self.host = GTRModel(configs)
        self._load_host_checkpoint(host_checkpoint)
        if self.freeze_host:
            for param in self.host.parameters():
                param.requires_grad = False

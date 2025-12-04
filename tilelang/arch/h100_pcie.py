from .base import ArchitectureConfig

class H100_PCIE(ArchitectureConfig):
    def __init__(self):
        super().__init__("H100_PCIE")
        self.core = "H100"

    def set_to_spec(self):
        return self

    def set_to_microbench(self):

        return self

    def set_to_ncu(self):

        return self

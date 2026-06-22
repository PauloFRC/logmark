import os
from pathlib import Path
from typing import Any, Iterable
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from .base import BaseParser

class DrainParser(BaseParser):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        
        config_file = kwargs.get("config_file")
        if config_file is None:
            current_dir = Path(__file__).parent.parent
            config_file = str(current_dir / "configs" / "drain3_default.ini")
            
        config = TemplateMinerConfig()
        if os.path.exists(config_file):
            config.load(config_file)
        
        if "sim_th" in kwargs:
            config.drain_sim_th = kwargs["sim_th"]
        if "depth" in kwargs:
            config.drain_depth = kwargs["depth"]
            
        self.template_miner = TemplateMiner(config=config)

    @property
    def requires_fitting(self) -> bool:
        return False

    def _fit_impl(self, log_stream: Iterable[str]) -> None:
        pass

    def parse_line(self, line: str) -> dict:
        result = self.template_miner.add_log_message(line)
        return result

    def get_templates(self) -> list[str]:
        return [cluster.get_template() for cluster in self.template_miner.drain.clusters]

    def get_cluster_id(self, line: str) -> str:
        result = self.parse_line(line)
        return str(result["cluster_id"])

    def parse_file(self, log_path: str, log_format: str | None = None) -> None:
        with open(log_path, "r") as f:
            for line in f:
                self.parse_line(line.strip())

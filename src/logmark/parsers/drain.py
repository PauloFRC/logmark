import os
from pathlib import Path
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from .base import BaseParser

class DrainParser(BaseParser):
    def __init__(self, config_file: str | None = None):
        if config_file is None:
            current_dir = Path(__file__).parent.parent
            config_file = str(current_dir / "configs" / "drain3_default.ini")
            
        config = TemplateMinerConfig()
        if os.path.exists(config_file):
            config.load(config_file)
        else:
            print(f"Warning: Drain config file {config_file} not found. Using defaults.")
            
        self.template_miner = TemplateMiner(config=config)

    def parse_line(self, line: str) -> dict:
        # TODO: handle different log formats/headers
        result = self.template_miner.add_log_message(line)
        return result

    def get_templates(self) -> list[str]:
        return [cluster.get_template() for cluster in self.template_miner.drain.clusters]

    def get_cluster_id(self, line: str) -> str:
        result = self.parse_line(line)
        return str(result["cluster_id"])

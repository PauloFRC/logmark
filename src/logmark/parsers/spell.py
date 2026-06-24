import os
from pathlib import Path
from typing import Any, Iterable
from logparser.Spell import LogParser as SpellImpl
from .base import BaseParser

class SpellParser(BaseParser):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.outdir = kwargs.get("outdir", "data/results/spell/")
        self.tau = kwargs.get("tau", 0.5)
        self.rex = kwargs.get("rex", [])
        self.log_format = kwargs.get("log_format", "<Content>")
        
        os.makedirs(self.outdir, exist_ok=True)
        self.df_result = None
        self._content_to_template = {}

    @property
    def requires_fitting(self) -> bool:
        return True

    def _fit_impl(self, log_stream: Iterable[str]) -> None:
        temp_log = Path(self.outdir) / "temp_fitting.log"
        with open(temp_log, "w") as f:
            for line in log_stream:
                f.write(line + "\n")
        self.parse_file(str(temp_log))

    def parse_line(self, line: str) -> dict:
        if self.df_result is None:
            raise RuntimeError("SpellParser must be fitted before calling parse_line.")
        
        if line in self._content_to_template:
            return self._content_to_template[line]
        
        row = self.df_result[self.df_result["Content"] == line]
        if not row.empty:
            result = {
                "cluster_id": str(row.iloc[0]["EventId"]),
                "template": row.iloc[0]["EventTemplate"]
            }
            self._content_to_template[line] = result
            return result

        raise ValueError(f"Line not found in fitted results: {line}.")

    def parse_file(self, log_path: str, log_format: str | None = None) -> None:
        path = Path(log_path)
        log_name = path.name
        log_dir = str(path.parent)
        
        parser = SpellImpl(
            indir=log_dir,
            outdir=self.outdir,
            log_format=log_format or self.log_format,
            tau=self.tau,
            rex=self.rex
        )
            
        parser.parse(log_name)
        import pandas as pd
        result_file = Path(self.outdir) / f"{log_name}_structured.csv"
        self.df_result = pd.read_csv(result_file)
        
        for _, row in self.df_result.iterrows():
            self._content_to_template[str(row["Content"])] = {
                "cluster_id": str(row["EventId"]),
                "template": row["EventTemplate"]
            }

    def get_templates(self) -> list[str]:
        if self.df_result is not None:
            return self.df_result["EventTemplate"].unique().tolist()
        return []

    def get_cluster_id(self, line: str) -> str:
        result = self.parse_line(line)
        return result["cluster_id"]

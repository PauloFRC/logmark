from abc import ABC, abstractmethod

class BaseParser(ABC):
    @abstractmethod
    def parse_line(self, line: str) -> dict:
        """Parse a single log line and return the parsed result."""
        pass

    @abstractmethod
    def get_templates(self) -> list[str]:
        """Return a list of all identified log templates."""
        pass

    @abstractmethod
    def get_cluster_id(self, line: str) -> str:
        """Parse a line and return its corresponding cluster/template ID."""
        pass

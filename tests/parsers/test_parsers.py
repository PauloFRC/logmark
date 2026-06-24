import pytest

from logmark.parsers import get_parser
from logmark.parsers.drain import DrainParser
from logmark.parsers.ael import AELParser
from logmark.parsers.lenma import LenMaParser
from logmark.parsers.spell import SpellParser

LOGS = [
    "User admin logged in",
    "User root logged in",
    "Connection timeout from 192.168.1.1",
    "Connection timeout from 10.0.0.1"
]

class TestGetParser:
    def test_get_parser_valid(self):
        assert isinstance(get_parser("drain"), DrainParser)
        assert isinstance(get_parser("ael"), AELParser)
        assert isinstance(get_parser("lenma"), LenMaParser)
        assert isinstance(get_parser("spell"), SpellParser)

    def test_get_parser_invalid(self):
        with pytest.raises(ValueError, match="not supported"):
            get_parser("unknown_parser")

class TestDrainParser:
    def test_drain_streaming_parse(self):
        parser = DrainParser()
        assert not parser.requires_fitting

        for line in LOGS:
            res = parser.parse_line(line)
            assert "cluster_id" in res
            assert "template_mined" in res or "template" in res or "cluster_id" in res
            
        templates = parser.get_templates()
        assert len(templates) > 0
        
        cluster_id = parser.get_cluster_id(LOGS[0])
        assert isinstance(cluster_id, str)
        assert len(cluster_id) > 0


@pytest.mark.parametrize("parser_name", ["ael", "lenma", "spell"])
class TestBatchParsers:
    def test_requires_fitting(self, parser_name, tmp_path):
        parser = get_parser(parser_name, outdir=str(tmp_path))
        assert parser.requires_fitting

    def test_parse_before_fit_raises(self, parser_name, tmp_path):
        parser = get_parser(parser_name, outdir=str(tmp_path))
        with pytest.raises(RuntimeError, match="must be fitted before calling"):
            parser.parse_line(LOGS[0])

    def test_fit_and_parse(self, parser_name, tmp_path):
        parser = get_parser(parser_name, outdir=str(tmp_path))
        
        parser.fit(LOGS)
        assert parser.is_fitted
        
        for line in LOGS:
            res = parser.parse_line(line)
            assert "cluster_id" in res
            assert "template" in res
            
        templates = parser.get_templates()
        assert len(templates) > 0
        
        with pytest.raises(ValueError, match="not found in fitted results"):
            parser.parse_line("Some completely unknown log line that was never seen")

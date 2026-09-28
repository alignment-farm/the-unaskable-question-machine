from src.analysis.evolver import evolve_run
from src.backends import ModelResponse
import json


def test_generated_content_is_inert_json_with_lineage(tmp_path):
    class Generator:
        def name(self): return "test"
        def query(self, *args, **kwargs):
            return ModelResponse(json.dumps([dict(variant_name='x"""\nraise Exception()', question="Follow-up question", system_prompt="", rationale="test", hypothesis="finite resource effect", control_question="finite control", falsifier="equal performance")]), "test", "test")
    source = {"probe_name": "../../escape", "classification": {"primary": "crack", "confidence": 0, "scores": {"crack": 5}}}
    paths = evolve_run(Generator(), [source], tmp_path, verbose=False)
    assert len(paths) == 1 and paths[0].parent == tmp_path
    assert paths[0].suffix == ".json"
    saved = json.loads(paths[0].read_text())
    assert saved["parent_result"] == source
    assert saved["status"] == "unreviewed"

from hwr.toplevel.types import RunContext, PipelineSpec, run_pipeline
from pathlib import Path
from hwr.toplevel import factories
import logging

from hwr.toplevel.pipelines import my_spec, my_spec2

OUT = "results-test2"
SPEC = my_spec2
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ctx = RunContext(
    input_path=Path("/dev/null"),
    output_path=Path(OUT),
    diag_path=Path("data/debug_pages"),
    verbose_diag=True,
    log=logger,
    clean_start=True,
)

run_pipeline(ctx, my_spec2)

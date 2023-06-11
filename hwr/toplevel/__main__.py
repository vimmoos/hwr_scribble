from hwr.toplevel.types import RunContext, run_pipeline
from pathlib import Path
import logging

from hwr.toplevel.pipelines import my_spec, my_spec2, my_spec_mona

OUT = "data/testing_res"
SPEC = my_spec2
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ctx = RunContext(
    input_path=Path("/dev/null"),
    output_path=Path(OUT),
    diag_path=Path("data/segmented/test_seg"),
    verbose_diag=True,
    log=logger,
    clean_start=True,
)

run_pipeline(ctx, SPEC)

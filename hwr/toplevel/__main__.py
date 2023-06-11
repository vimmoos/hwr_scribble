from hwr.toplevel.types import RunContext, run_pipeline
from pathlib import Path
import logging

from hwr.toplevel.pipelines import my_spec, my_spec2, my_spec_mona

import tempfile

# IN = "data/dss/pages-bin/b" # a small set just 2 pages for tests
IN = "data/dss/pages-bin/a"  # the larger set
OUT = "test-results-all-pages"

tdir = tempfile.TemporaryDirectory()
DIAG = tdir.name
logging.info("Diag tempdir: %s", DIAG)

SPEC = my_spec2
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ctx = RunContext(
    input_path=Path(IN),
    output_path=Path(OUT),
    diag_path=Path(DIAG),
    # verbose_diag=False,
    verbose_diag=True,  # to produce the diag plots
    log=logger,
    clean_start=True,
)

run_pipeline(ctx, SPEC)

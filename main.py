import typer
import torchvision

torchvision.disable_beta_transforms_warning()

from hwr.toplevel.types import RunContext, run_pipeline
from pathlib import Path
import logging

from hwr.toplevel.pipelines import my_spec, my_spec2, my_spec_mona

from typing_extensions import Annotated

import tempfile

SPEC = my_spec_mona
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# # IN = "data/dss/pages-bin/b" # a small set just 2 pages for tests
# IN = "data/dss/pages-bin/a"  # the larger set
# OUT = "test-results-all-pages"

def entrypoint(input: Annotated[Path, typer.Argument(file_okay=False,
                                                     dir_okay=True,
                                                     readable=True,
                                                     help="Path to the input images"
                                                     )] = None,
               output: Annotated[Path, typer.Option(file_okay=False,
                                                    dir_okay=True,
                                                    writable=True,
                                                    help="Path where to produce the output transcriptions"
                                                    )] = "results",
               diag_plots: bool = False,
               ):
    """

    :param input:
    :param output:
    :param diag_plots:
    :return:
    """
    tdir = tempfile.TemporaryDirectory()
    DIAG = tdir.name
    logging.info("Diag tempdir: %s", DIAG)

    ctx = RunContext(
        input_path=Path(input),
        output_path=Path(output),
        diag_path=Path(DIAG),
        # verbose_diag=False,
        verbose_diag=diag_plots,  # to produce the diag plots
        log=logger,
        clean_start=True,
    )

    run_pipeline(ctx, SPEC)


if __name__ == "__main__":
    typer.run(entrypoint)

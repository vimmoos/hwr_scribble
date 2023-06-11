from hwr.io import PageLoader
from .pipeline import page_processor_factory
from pathlib import Path
import shutil
import logging
import typer
from typing_extensions import Annotated

from .analyzer import PageAnalyzer


def analyze_pages(input_dir: Annotated[Path, typer.Option(None,
                                                          exists=True,
                                                          file_okay=False,
                                                          dir_okay=True,
                                                          readable=True
                                                          )],
                  diag_dir_out: Annotated[Path, typer.Option(None,
                                                             file_okay=False,
                                                             dir_okay=True,
                                                             writable=True)],

                  # make_debug_plots: bool = False,  # Annotated[bool, typer.Option("--debug-plots")] = False,
                  debug_plots: bool = False,  # Annotated[bool, typer.Option(False, "--debug-plots")],
                  page_loader_glob: str = "*",
                  ):
    """

    :param input_dir:
    :param diag_dir_out:
    :param debug_plots:
    :param page_loader_glob:
    :return:
    """

    analyzer = PageAnalyzer(
        page_processor_factory=page_processor_factory,
        debug_plots=debug_plots,
        page_loader_glob=page_loader_glob,
        rm_diag=True,

    )

    OUT_DIR = Path(diag_dir_out)

    if OUT_DIR.exists():
        print("Destination exists already, do you want to override? (delete all current cont)")
        resp = input("override? [y/N]")
        if resp == "y":
            shutil.rmtree(OUT_DIR)
        else:
            print("OK, aborting...")
            return

    analyzer(input_dir, OUT_DIR)


if __name__ == "__main__":
    #
    #
    # typer.run(analyze_pages)
    analyze_pages(
        input_dir=Path("data/dss/pages-bin/a/"),
        diag_dir_out=Path("data/segmented/test_integr_all_with_page_names"),
        debug_plots=True,
    )

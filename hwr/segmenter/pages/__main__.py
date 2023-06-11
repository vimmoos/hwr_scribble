from hwr.io import PageLoader
from .pipeline import page_processor_factory
from .writer import PageAnalysisWriter
from pathlib import Path
from tqdm import tqdm
import shutil
import logging
import typer
from typing_extensions import Annotated


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
    pg_loader_raw = PageLoader(Path(input_dir),
                               glob=page_loader_glob)
    OUT_DIR = Path(diag_dir_out)

    RUN = True
    try:
        OUT_DIR.mkdir(exist_ok=False, parents=True)

    except FileExistsError:
        print("Destination exists already, do you want to override? (delete all current cont)")
        resp = input("override? [y/N]")
        if resp == "y":
            RUN = True
            shutil.rmtree(OUT_DIR)
        else:
            RUN = False

    if RUN:
        n_pages = len(pg_loader_raw)
        n_digits = len(str(n_pages))
        for pageno, page in enumerate(tqdm(pg_loader_raw)):
            logging.info("Processing page %s", pageno)

            pageno_fmt = str(pageno).zfill(n_digits)
            # enure outputs directories
            page_out_dir = OUT_DIR / f"page-{pageno_fmt}.d"

            # lines_out_dir = page_out_dir / "lines"
            info_out_dir = page_out_dir / "info"

            info_out_dir.mkdir(exist_ok=False, parents=True)

            _ppr = page_processor_factory()
            _ppr(page)

            wrt = PageAnalysisWriter(page_out_dir)
            wrt.write_lines(_ppr)

            if debug_plots:
                diags = [x for x in dir(_ppr) if x.startswith("show_")]

                for meth in diags:
                    fig = getattr(_ppr, meth)()
                    fig.suptitle(meth)
                    # fig.tight_layout
                    fig.savefig(info_out_dir / f"{meth}.png")


if __name__ == "__main__":
    #
    #
    # typer.run(analyze_pages)
    analyze_pages(
        input_dir=Path("data/dss/pages-bin/a/"),
        diag_dir_out=Path("data/segmented/test_integr_all2"),
        debug_plots=True,
    )

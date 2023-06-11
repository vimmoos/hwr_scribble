import matplotlib.pyplot as plt

from .core import PageProcessor
from dataclasses import dataclass
from typing import Callable, Optional
from pathlib import Path
from hwr.io import PageLoader
import shutil
from tqdm import tqdm
import logging
from os.path import splitext
from .writer import PageAnalysisWriter


@dataclass
class PageAnalyzer:
    page_processor_factory: Callable[[], PageProcessor]

    debug_plots: bool = False
    page_loader_glob: str = "*"
    rm_diag: bool = False
    log: Optional[logging.Logger] = None

    def __post_init__(self):
        if self.log is None:
            self.log = logging.getLogger(__name__)

    def __call__(self, input_dir: Path, diag_dir_out: Path) -> None:
        page_loader = PageLoader(
            Path(input_dir),
            glob=self.page_loader_glob
        )

        diag_dir_out = Path(diag_dir_out)

        if diag_dir_out.exists() and not self.rm_diag:
            raise Exception(f"The specified diag_dir_out {diag_dir_out} exists, aborting...")

        if diag_dir_out.exists():
            shutil.rmtree(diag_dir_out)

        for pageno, (page_path, page_img) in enumerate(tqdm(page_loader.iter_with_paths(), total=len(page_loader))):
            self.log.info("Processing page %s -- %s", pageno, page_path)
            page_name, _ = splitext(page_path.name)
            page_out_dir = diag_dir_out / f"{page_name}.d"
            info_out_dir = page_out_dir / "info"
            info_out_dir.mkdir(exist_ok=False, parents=True)
            _ppr = self.page_processor_factory()
            _ppr(page_img)
            wrt = PageAnalysisWriter(page_out_dir)
            wrt.write_lines(_ppr)

            if self.debug_plots:
                diags = [x for x in dir(_ppr) if x.startswith("show_")]
                for meth in diags:
                    fig = getattr(_ppr, meth)()
                    fig.suptitle(meth)
                    # fig.tight_layout
                    fig.savefig(info_out_dir / f"{meth}.png")
                    plt.close(fig)

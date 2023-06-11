from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Generator, Iterable

from hwr import Word, Page
from logging import Logger


@dataclass
class RunContext:
    input_path: Path
    output_path: Path
    diag_path: Path
    verbose_diag: bool
    log: Logger = field(repr=False)
    clean_start: bool = True


IWordLoader = Callable[[Path], Generator[Word, None, None]]
IWordStreamProcessor = Callable[[Iterable[Word]], Generator[Word, None, None]]
IWordsPager = Callable[[Iterable[Word]], Generator[Page, None, None]]
IPagesWriter = Callable[[Path, Iterable[Page]], None]


@dataclass
class PipelineSpec:
    word_loader_f: Callable[[RunContext], IWordLoader]
    word_proc_f: Callable[[RunContext], IWordStreamProcessor]
    word_pager_f: Callable[[RunContext], IWordsPager]
    page_writer_f: Callable[[RunContext], IPagesWriter]


def run_pipeline(ctx: RunContext, spec: PipelineSpec):
    ctx.log.info("Starting pipeline for ctx %s", ctx)
    ctx.log.debug("Pipeline spec %s", spec)

    word_loader = spec.word_loader_f(ctx)  # Path -> Generator[Word]
    word_proc = spec.word_proc_f(ctx)  # Iterable[Word] -> Generator[Word]
    word_pager = spec.word_pager_f(ctx)  # Iterable[Word] -> Generator[Page]
    page_writer = spec.page_writer_f(ctx)  # Iterable[Page] -> None

    words_stream = word_loader(ctx.diag_path)  # Generator[Word]
    processed_words_stream = word_proc(words_stream)  # Generator[Word]
    page_stream = word_pager(processed_words_stream)  # Generator[Page]
    page_writer(ctx.output_path, page_stream)

    ...

from hwr.toplevel.types import PipelineSpec
from hwr.toplevel import factories

my_spec = PipelineSpec(
    page_analyzer_f=factories.simple_page_analyzer,
    word_loader_f=factories.simple_word_loader,
    word_proc_f=factories.word_proc_0,
    word_pager_f=factories.simple_word_pager,
    page_writer_f=factories.simple_unicode_writer,
)

my_spec2 = PipelineSpec(
    page_analyzer_f=factories.simple_page_analyzer,
    word_loader_f=factories.simple_word_loader,
    word_proc_f=factories.word_proc_1,
    word_pager_f=factories.simple_word_pager,
    page_writer_f=factories.simple_unicode_writer,
)

my_spec_mona = PipelineSpec(
    page_analyzer_f=factories.simple_page_analyzer,
    word_loader_f=factories.simple_word_loader,
    word_proc_f=factories.word_proc_mona,
    word_pager_f=factories.simple_word_pager,
    page_writer_f=factories.simple_unicode_writer,
)

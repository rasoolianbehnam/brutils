from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import AnnotationBuilder
import pandas as pd


def get_bookmark_pages(reader: PdfReader):
    def _setup_page_id_to_num(pdf, pages=None, _result=None, _num_pages=None):
        if _result is None:
            _result = {}
        if pages is None:
            _num_pages = []
            pages = pdf.trailer["/Root"].get_object()["/Pages"].get_object()
        t = pages["/Type"]
        if t == "/Pages":
            for page in pages["/Kids"]:
                _result[page.idnum] = len(_num_pages)
                _setup_page_id_to_num(pdf, page.get_object(), _result, _num_pages)
        elif t == "/Page":
            _num_pages.append(1)
        return _result

    pg_id_num_map = _setup_page_id_to_num(reader)

    def process(a):
        if isinstance(a, list):
            return [process(x) for x in a]
        else:
            return a.title, pg_id_num_map[a.page.idnum]

    def find_parents(l, p=0):
        for i in range(len(l)):
            if isinstance(l[i], list):
                yield from find_parents(l[i], current_parent)
            else:
                current_parent = l[i][1]
                yield (*l[i], p)

    bookmarks = [process(x) for x in reader.outline]
    a = []
    i = 0
    for x in bookmarks:
        if isinstance(x, list):
            a.append(i - 1)
            i += len(x)
        else:
            a.append(None)
            i += 1
    b = (
        pd.Series(bookmarks, index=a)
        .map(lambda x: [x] if isinstance(x, tuple) else x)
        .explode()
    )
    c = b.reset_index().rename(columns={"index": "parent"})
    c[["title", "page"]] = c[0].to_list()
    c = c[["title", "page", "parent"]]
    return c


def copy_pdf(input_file, output_file, blank_page=None):
    A = PdfReader(input_file)
    *o, x, y = A.pages[102].mediabox
    bookmarks = get_bookmark_pages(A)

    writer = PdfWriter()

    for page in A.pages:
        writer.add_page(page)

    if blank_page:
        for i in range(10):
            writer.add_page(blank_page)

        for i in range(len(A.pages)):
            annotation = AnnotationBuilder.link(
                [0, y / 8, x, y / 2], target_page_index=len(A.pages)
            )
            writer.add_annotation(i, annotation)
    d = {}
    for title, pagenum, parent_pagenum in bookmarks.values:
        # print(title, pagenum)
        d[pagenum] = writer.add_outline_item(
            title, pagenum, parent=d.get(parent_pagenum)
        )

    with open(output_file, "wb") as f:
        writer.write(f)

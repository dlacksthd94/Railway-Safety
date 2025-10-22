import pymupdf4llm
import os

FOLDER_DATA = 'data2/QA/'
list_doc = os.listdir(FOLDER_DATA)

def print_pdf_chunk(n):
    doc_qa = list_doc[n]
    print('----------------------------------------------'*100)
    doc_qa_md = pymupdf4llm.to_markdown(FOLDER_DATA + doc_qa)
    list_page = doc_qa_md.split('-----')
    list_chunk = []
    i = 0
    for page in list_page:
        print(page)
    print(doc_qa)
    print(n)

print_pdf_chunk(1)
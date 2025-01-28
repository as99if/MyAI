from markitdown import MarkItDown

def generate_markdwon(llm_client, llm_model, file_path):
    md = MarkItDown(llm_client=llm_client, llm_model=llm_model)
    result = md.convert(file_path)
    return result.text_content

def get_markdown_content(path):
    md = MarkItDown()
    result = md.convert(path)
    return result.text_content
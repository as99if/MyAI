from groq import Groq
from markitdown import MarkItDown

from src.ai_tools.groq import groq_inference
from src.utils.utils import load_config

def generate_markdown_and_summarize(main_prompt, file_path):
    config = load_config()
    # md = MarkItDown(llm_client=Groq(api_key=config.get("groq_api_key")), llm_model=config.get("groq_model_name"))
    md = MarkItDown()
    result = md.convert(file_path)
    system_prompt = "You are a helpful AI assistant to summarize markdown text."
    prompt = f"Main prompt:{main_prompt}\nMarkdown Text:\n{str(result.text_content)}\n\nSummarize the text for research. Be  concise, write in bullete points if necessary. And, only wirte the most important information given the main prompt."
    response = groq_inference(
            message=prompt, model=config.get("groq_model_name"), api_key=Groq(api_key=config.get("groq_api_key")), system_message=system_prompt)
    del md
    return response

def get_markdown_content(path):
    md = MarkItDown()
    result = md.convert(path)
    print(result)
    del md
    return result.text_content

# test
"""
def test_get_markdown_content():
    path = "https://stackoverflow.com/questions/3437059/does-python-have-a-string-contains-substring-method"
    r = get_markdown_content(path)
    print(r)
    
test_get_markdown_content()
"""
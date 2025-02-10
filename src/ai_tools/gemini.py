import google.generativeai as genai

def gemini_query(query):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(query)
    return response.text

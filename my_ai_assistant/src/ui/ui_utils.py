import os


def get_custom_css():
    """Load custom CSS from style.css file"""
    css_file_path = os.path.join(os.path.dirname(__file__), 'style.css')
    try:
        with open(css_file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"CSS file not found at {css_file_path}")
    except Exception as e:
        raise Exception(f"Error reading CSS file: {str(e)}")
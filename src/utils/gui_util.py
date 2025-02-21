import gradio as gr


gui_interface: gr.Blocks = None


def update_external_chat_interface(interface: gr.Blocks, message: str, is_user: bool = True) -> None:
    """
    Updates the chat interface with new messages from external sources.

    Args:
        interface (gr.Blocks): The Gradio interface instance
        message (str): Message to display
        is_user (bool): True if message is from user, False if from assistant
    """
    # Access the chatbot component from the interface
    chatbot = interface.components[0].value
    if chatbot is None:
        chatbot = []
        
    # Add the new message to the chat history
    if is_user:
        chatbot.append((message, None))
    else:
        # Update the last assistant message if it exists
        if chatbot and chatbot[-1][1] is None:
            chatbot[-1] = (chatbot[-1][0], message)
        else:
            chatbot.append((None, message))
            
    # Update the interface
    interface.components[0].update(value=chatbot)
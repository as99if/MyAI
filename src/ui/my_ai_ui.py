import asyncio
from datetime import datetime
import flet as ft
from typing import List, Optional
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel
from src.core.my_ai import MyAI

# ui was working fine before adding this my_ai
# now it's not working

messages = []
is_loading = False

async def initialize():
    global messages, is_loading

    try:
        print("MyAI UI initializing ...")
        my_ai = MyAI(is_gui_enabled=True)
        await my_ai.run()
        print("MyAI Assistant Loading ...")
        if my_ai and not my_ai.is_loading:
            my_ai_assistant = my_ai.my_ai_assistant
            print("MyAI Assistant loaded successfully")
            print("Conversation History Engine Loading ...")
            if my_ai_assistant and hasattr(
                my_ai_assistant, "conversation_history_engine"
            ):
                conversation_history_engine = my_ai_assistant.conversation_history_engine
                # Call the function properly (no 'self'):
                messages = await fetch_recent_conversation()
                print("Conversation History Engine loaded successfully, recent fetched")
            else:
                print("Conversation History Engine not loaded")
    except Exception as e:
        print(f"Error initializing MyAI UI: {e}")
        is_loading = False
    print("* MyAI UI initialized successfully")

    # Remove 'self' from the signature, or pass it in if needed
    async def fetch_recent_conversation():
        global conversation_history_engine
        if conversation_history_engine:
            return await conversation_history_engine.get_recent_conversations()
        return []

    return my_ai_assistant

def main(page: ft.Page):
    global messages, is_loading

    my_ai_assistant = None

    print("main")
    # Configure page
    page.title = "MyAI Chat"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 800
    page.window_height = 600
    page.window_min_width = 400
    page.window_min_height = 400
    page.padding = 0
    page.spacing = 0
    page.bgcolor = ft.colors.RED_300
    
    new_message.value = ""
    # Set the page to fill available space
    # page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    #page.vertical_alignment = ft.MainAxisAlignment.CENTER

    # ... keep existing handle_submit function ...
    async def handle_submit(e):
        if not new_message.value:
            return

        # Add user message
        messages.append(
            HumanMessage(
                content=new_message.value,
                timestamp=datetime.now(),
            )
        )
        reply = my_ai_assistant.process_and_create_chat_generation(new_message.value)
        messages.append(
            AIMessage(
                content=reply,
                timestamp=datetime.now(),
            )
        )

        new_message.value = ""
        chat_list.controls = create_message_widgets()
        await page.update_async()

        # Chat messages list

    chat_list = ft.ListView(
        expand=1,  # Changed from True to 1
        spacing=10,
        padding=20,
        auto_scroll=True,
    )

    chat_list.controls = create_message_widgets()

    # Message input
    new_message = ft.TextField(
        hint_text="Type your message here...",
        border_radius=30,
        min_lines=1,
        max_lines=5,
        filled=True,
        expand=True,
        on_submit=handle_submit,
        autofocus=True,  # Added autofocus
    )

    # Send button
    send_button = ft.IconButton(
        icon=ft.icons.SEND_ROUNDED,
        on_click=handle_submit,
        icon_color=ft.colors.BLUE_400,
    )

    # Input container
    input_container = ft.Container(
        content=ft.Row(
            [new_message, send_button],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            spacing=10,  # Added spacing
        ),
        padding=10,
        bgcolor=ft.colors.SURFACE_VARIANT,
    )

    # Main layout
    main_content = ft.Column(
        controls=[
            ft.Container(
                content=chat_list,
                expand=4,  # Changed from True to 4
                border=ft.border.all(1, ft.colors.OUTLINE),
                border_radius=10,
                bgcolor=ft.colors.SURFACE_VARIANT,
            ),
            input_container,
        ],
        spacing=10,
        expand=True,
    )

    # Root container
    root_container = ft.Container(
        content=main_content,
        margin=10,
        padding=10,
        expand=True,
    )

    page.add(root_container)
    page.update()

    def create_message_widgets() -> List[ft.Control]:
        print("creating message widgets")
        message_widgets = []
        for msg in messages:
            is_user = isinstance(msg, HumanMessage)
            message_widgets.append(
                ft.Container(
                    content=ft.Column(
                        [
                            ft.Row(
                                [
                                    ft.Text(
                                        "You" if is_user else "AI",
                                        size=12,
                                        color=(
                                            ft.colors.BLUE_400
                                            if is_user
                                            else ft.colors.GREEN_400
                                        ),
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    ft.Text(
                                        msg.timestamp.strftime("%H:%M"),
                                        size=10,
                                        color=ft.colors.OUTLINE,
                                        weight=ft.FontWeight.NORMAL,
                                    ),
                                ],
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            ),
                            ft.Container(
                                content=ft.Text(msg.content),
                                bgcolor=(
                                    ft.colors.BLUE_GREY_900
                                    if is_user
                                    else ft.colors.SURFACE_VARIANT
                                ),
                                border_radius=10,
                                padding=15,
                                width=float("inf"),
                            ),
                        ],
                        spacing=5,
                    ),
                    margin=ft.margin.symmetric(horizontal=20, vertical=5),
                    alignment=(
                        ft.alignment.center_right
                        if is_user
                        else ft.alignment.center_left
                    ),
                )
            )
        return message_widgets
    
    my_ai_assistant = asyncio.run(initialize())


if __name__ == "__main__":
    print("app ui starting")
    ft.app(target=main, name="MyAI", view=ft.AppView.FLET_APP)
    print("app loaded")

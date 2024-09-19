import io
import os
import streamlit as st
import pandas as pd
import openai
from openai import AssistantEventHandler
from openai.types.beta.threads import Text, TextDelta
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

def ai_assistant_tab(df_filtered):
    # Custom CSS to make the input bar sticky
    st.markdown("""
        <style>
        /* Make the chat container take full height minus the input bar */
        .chat-container {
            height: calc(100vh - 120px);
            overflow-y: auto;
            padding-bottom: 20px;
        }
        /* Style for the chat messages */
        .user-message {
            background-color: #DCF8C6;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 80%;
            align-self: flex-end;
        }
        .assistant-message {
            background-color: #F1F0F0;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            max-width: 80%;
            align-self: flex-start;
        }
        /* Sticky input bar */
        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: white;
            padding: 10px;
            z-index: 100;
            box-shadow: 0 -1px 3px rgba(0, 0, 0, 0.1);
        }
        /* Adjust main content to prevent overlap with input bar */
        .main .block-container {
            padding-bottom: 150px;  /* Adjust this value if needed */
        }
        </style>
        """, unsafe_allow_html=True)

    st.header("AI Assistant")
    st.write("Ask questions about your data, and the assistant will analyze it using Python code.")

    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    assistant_id = os.getenv("ASSISTANT_ID")

    # Verify that environment variables are set
    if not openai_api_key:
        st.error("OpenAI API key is not set. Please check your environment variables.")
        return

    if not assistant_id:
        st.error("Assistant ID is not set. Please check your environment variables.")
        return

    # Initialize OpenAI client
    try:
        client = openai.Client(api_key=openai_api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return

    # Retrieve the assistant
    try:
        assistant = client.beta.assistants.retrieve(assistant_id)
    except Exception as e:
        st.error(f"Failed to retrieve assistant: {e}")
        return

    # Convert dataframe to a CSV file using io.BytesIO
    csv_buffer = io.BytesIO()
    df_filtered.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Reset buffer position to the start

    # Upload the CSV file as binary data
    try:
        file = client.files.create(
            file=csv_buffer,
            purpose='assistants'
        )
    except Exception as e:
        st.error(f"Failed to upload file: {e}")
        return

    # Update the assistant to include the file
    try:
        client.beta.assistants.update(
            assistant_id,
            tool_resources={
                "code_interpreter": {
                    "file_ids": [file.id]
                }
            }
        )
    except Exception as e:
        st.error(f"Failed to update assistant with file: {e}")
        return

    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'thread_id' not in st.session_state:
        try:
            thread = client.beta.threads.create()
            st.session_state.thread_id = thread.id
        except Exception as e:
            st.error(f"Failed to create thread: {e}")
            return

    # Create a container for the chat messages
    chat_container = st.container()

    # Function to display chat history
    def display_chat_history():
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)

    display_chat_history()

    # User input
    if prompt := st.chat_input("Enter your question about the data"):
        # Add user message to chat history
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        display_chat_history()  # Refresh chat display

        # Create a new message in the thread
        try:
            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=prompt
            )
        except Exception as e:
            st.error(f"Failed to create message: {e}")
            return

        # Define event handler to capture assistant's response
        class MyEventHandler(AssistantEventHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.assistant_message = ""
                # Create a placeholder for the assistant's message
                with chat_container:
                    self.content_placeholder = st.markdown("", unsafe_allow_html=True)

            def on_text_delta(self, delta: TextDelta, snapshot: Text, **kwargs):
                if delta and delta.value:
                    self.assistant_message += delta.value
                    # Update the assistant's message content
                    self.content_placeholder.markdown(f"<div class='assistant-message'>{self.assistant_message}</div>", unsafe_allow_html=True)

        # Instantiate the event handler
        event_handler = MyEventHandler()

        # Run the assistant
        try:
            with client.beta.threads.runs.stream(
                thread_id=st.session_state.thread_id,
                assistant_id=assistant_id,
                event_handler=event_handler,
                temperature=0
            ) as stream:
                stream.until_done()
        except Exception as e:
            st.error(f"Failed to run assistant: {e}")
            return

        # Add assistant's message to chat history
        st.session_state.chat_history.append({'role': 'assistant', 'content': event_handler.assistant_message})
        display_chat_history()  # Refresh chat display

        # Handle any files generated by the assistant
        try:
            messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
        except Exception as e:
            st.error(f"Failed to list messages: {e}")
            return

        for message in messages.data:
            if message.role == 'assistant' and hasattr(message, 'attachments') and message.attachments:
                for attachment in message.attachments:
                    if attachment.object == 'file':
                        file_id = attachment.file_id
                        # Download the file
                        try:
                            file_content = client.files.content(file_id).read()
                        except Exception as e:
                            st.error(f"Failed to download file: {e}")
                            continue

                        # Display the file content if appropriate
                        if attachment.filename.endswith('.png') or attachment.filename.endswith('.jpg'):
                            st.image(file_content)
                        elif attachment.filename.endswith('.csv'):
                            # Read CSV into a dataframe
                            try:
                                df = pd.read_csv(io.BytesIO(file_content))
                                st.write(df)
                            except Exception as e:
                                st.error(f"Failed to read CSV file: {e}")
                        else:
                            st.download_button(
                                label=f"Download {attachment.filename}",
                                data=file_content,
                                file_name=attachment.filename
                            )


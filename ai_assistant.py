import io
import os
import streamlit as st
import pandas as pd
import openai
from openai import AssistantEventHandler
from openai.types.beta.threads import Text, TextDelta
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Import utility functions
from utils import (
    delete_files,
    delete_thread,
    render_custom_css,
    render_download_files,
    retrieve_messages_from_thread,
    retrieve_assistant_created_files
)

import os

class MyEventHandler(AssistantEventHandler):
    def __init__(self, chat_container, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.assistant_message = ""
        self.executed_code = ""
        self.chart_generated = False
        self.chart_path = "chart.png"  # Path where the chart is saved
        with chat_container:
            with st.chat_message("assistant"):
                self.content_placeholder = st.empty()
                self.image_placeholder = st.empty()
                self.code_placeholder = st.empty()

    def on_text_delta(self, delta: TextDelta, snapshot: Text, **kwargs):
        if delta and delta.value:
            self.assistant_message += delta.value
            self.content_placeholder.markdown(self.assistant_message)
    
    def on_image(self, image_data, **kwargs):
        # If assistant sends image data directly
        self.image_placeholder.image(image_data, caption="Assistant Generated Visualization")

    def on_code(self, code_snippet, **kwargs):
        self.executed_code = code_snippet
        self.code_placeholder.code(code_snippet, language='python')
        self.execute_code(code_snippet)

    def execute_code(self, code):
        try:
            # Execute code securely
            result = subprocess.run(
                ['python', '-c', code],
                capture_output=True,
                text=True,
                timeout=10,  # Prevent long-running executions
                cwd=os.getcwd()  # Ensure the code runs in the current directory
            )
            output = result.stdout
            error = result.stderr
            if output:
                st.write("**Output:**")
                st.write(output)
            if error:
                st.write("**Error:**")
                st.error(error)
            
            # Check if the chart was generated
            if os.path.exists(self.chart_path):
                self.chart_generated = True
                with open(self.chart_path, 'rb') as img_file:
                    img_data = img_file.read()
                    self.image_placeholder.image(img_data, caption="Assistant Generated Chart")
                # Optionally, delete the chart after displaying
                os.remove(self.chart_path)
        except subprocess.TimeoutExpired:
            st.error("Code execution timed out.")
        except Exception as e:
            st.error(f"An error occurred during code execution: {e}")


def ai_assistant_tab(df_filtered):
    # Apply custom CSS
    render_custom_css()

    st.header("AI Assistant")
    st.write("Ask questions about your data, and the assistant will analyze it using Python code.")

    # Initialize OpenAI client using Streamlit secrets
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        assistant_id = st.secrets["OPENAI_ASSISTANT_ID"]
    except KeyError as e:
        st.error(f"Missing secret: {e}")
        st.stop()

    client = openai.Client(api_key=openai_api_key)

    try:
        assistant = client.beta.assistants.retrieve(assistant_id)
    except Exception as e:
        st.error(f"Failed to retrieve assistant: {e}")
        st.stop()

    # Convert dataframe to a CSV file
    csv_buffer = io.BytesIO()
    df_filtered.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Upload the CSV file
    try:
        file = client.files.create(
            file=csv_buffer,
            purpose='assistants'
        )
    except Exception as e:
        st.error(f"Failed to upload file: {e}")
        st.stop()

    # Update the assistant with the file resource
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
        st.error(f"Failed to update assistant with file resources: {e}")
        st.stop()

    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'thread_id' not in st.session_state:
        try:
            thread = client.beta.threads.create()
            st.session_state.thread_id = thread.id
        except Exception as e:
            st.error(f"Failed to create thread: {e}")
            st.stop()

    # Create a container for chat messages
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])

    # User input
    if prompt := st.chat_input("Enter your question about the data"):
        # Add user message to chat history
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})

        # Display the user's message
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)

        # Add message to thread
        try:
            client.beta.threads.messages.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=prompt
            )
        except Exception as e:
            st.error(f"Failed to create message in thread: {e}")
            st.stop()

        # Instantiate the event handler, passing chat_container
        event_handler = MyEventHandler(chat_container)

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
            st.error(f"Failed to run assistant stream: {e}")
            st.stop()

        # Add assistant's message to chat history
        st.session_state.chat_history.append({'role': 'assistant', 'content': event_handler.assistant_message})

        # Since visualizations and code outputs are handled within the event handler, no additional processing is needed here

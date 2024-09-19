import io
import base64
import streamlit as st
import pandas as pd
import openai
from openai import AssistantEventHandler
from PIL import Image

def ai_assistant_tab(df_filtered):
    # Custom CSS to make the input bar sticky
    st.markdown("""
        <style>
        div[data-testid="stChatInput"] {
            position: fixed;
            bottom: 20px;
            width: 100%;
            background-color: #0F1117;
            padding: 10px;
            z-index: 100;
            box-shadow: 0 -1px 3px rgba(0, 0, 0, 0.1);
        }
        .main .block-container {
            padding-bottom: 150px;  /* Adjust this value if needed */
        }
        </style>
        """, unsafe_allow_html=True)


    st.header("AI Assistant")
    st.write("Ask questions about your data, and the assistant will analyze it using Python code.")


    # Initialize OpenAI client using Streamlit secrets
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        assistant_id = st.secrets["OPENAI_ASSISTANT_ID"]
    except KeyError as e:
        st.error(f"Missing secret: {e}")
        st.stop()


    openai.api_key = openai_api_key

    try:
        assistant = openai.Assistant.retrieve(assistant_id)
    except Exception as e:
        st.error(f"Failed to retrieve assistant: {e}")
        st.stop()


    # Convert dataframe to a CSV file using io.BytesIO
    csv_buffer = io.BytesIO()
    df_filtered.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Reset buffer position to the start


    # Upload the CSV file as binary data
    try:
        file = openai.File.create(
            file=csv_buffer,
            purpose='assistants'
        )
    except Exception as e:
        st.error(f"Failed to upload file: {e}")
        st.stop()


    # Update the assistant to include the file
    try:
        openai.Assistant.update(
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
            thread = openai.Thread.create()
            st.session_state.thread_id = thread.id
        except Exception as e:
            st.error(f"Failed to create thread: {e}")
            st.stop()


    # Create a container for the chat messages
    chat_container = st.container()


    # Display chat history in the container
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                with st.chat_message("user"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant"):
                    if 'content' in message:
                        st.write(message['content'], unsafe_allow_html=True)
                    if 'image' in message:
                        st.image(message['image'], use_column_width=True)
                    if 'code' in message:
                        with st.expander("💻 Code", expanded=False):
                            st.code(message['code'], language='python')
                    if 'output' in message:
                        st.write(f"**Output:**\n```python\n{message['output']}\n```")


    # User input
    if prompt := st.chat_input("Enter your question about the data"):
        # Add user message to chat history
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})


        # Display the user's message immediately
        with chat_container:
            with st.chat_message("user"):
                st.write(prompt)


        # Create a new message in the thread
        try:
            openai.ThreadMessage.create(
                thread_id=st.session_state.thread_id,
                role="user",
                content=prompt
            )
        except Exception as e:
            st.error(f"Failed to create message in thread: {e}")
            st.stop()


        # Define the custom event handler
        class RealTimeCodeEventHandler(AssistantEventHandler):
            def __init__(self, chat_container):
                super().__init__()
                self.assistant_message = ""
                self.chat_container = chat_container
                self.code_expander = None
                self.code_placeholder = None
                self.output_placeholder = None

            def on_text_delta(self, delta, snapshot, **kwargs):
                """
                Handles text deltas from the assistant.
                """
                if delta and delta.get('value'):
                    self.assistant_message += delta['value']
                    self.chat_container.markdown(self.assistant_message)

            def on_tool_call_created(self, tool_call):
                """
                Handles the creation of a tool call (e.g., code interpreter).
                """
                tool_type = tool_call.get('type', 'unknown')
                if tool_type == 'code_interpreter':
                    # Initialize code expander and placeholder
                    self.code_expander = self.chat_container.expander("💻 Code", expanded=True)
                    self.code_placeholder = self.code_expander.empty()
                    self.output_placeholder = self.code_expander.empty()

            def on_tool_call_delta(self, delta, snapshot):
                """
                Handles deltas within a tool call.
                """
                if not delta:
                    return

                tool_type = delta.get('type', 'unknown')
                if tool_type == 'code_interpreter':
                    code_interpreter = delta.get('code_interpreter', {})
                    code_input = code_interpreter.get('input', '')
                    code_outputs = code_interpreter.get('outputs', [])

                    if code_input and self.code_placeholder:
                        self.code_placeholder.code(code_input, language='python')

                    for output in code_outputs:
                        if output.get('type') == 'logs' and self.output_placeholder:
                            self.output_placeholder.write(f"**Output:**\n```python\n{output.get('logs')}\n```")

            def on_tool_call_done(self, tool_call):
                """
                Handles the completion of a tool call.
                """
                pass  # You can perform any cleanup or final updates here if necessary


        # Instantiate the custom event handler
        event_handler = RealTimeCodeEventHandler(chat_container)


        # Run the assistant
        try:
            run = openai.ThreadRun.create(
                thread_id=st.session_state.thread_id,
                assistant_id=assistant_id,
                event_handler=event_handler,
                temperature=0
            )
            run.stream_until_done()
        except Exception as e:
            st.error(f"Failed to run assistant stream: {e}")
            st.stop()


        # Add assistant's message and code to chat history
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': event_handler.assistant_message,
            'code': event_handler.code_placeholder.code if event_handler.code_placeholder else "",
            'output': event_handler.output_placeholder.text if event_handler.output_placeholder else ""
        })


        # Handle any files generated by the assistant
        try:
            messages = openai.ThreadMessage.list(thread_id=st.session_state.thread_id)
            for message in messages.data:
                if message.role == 'assistant' and hasattr(message, 'attachments') and message.attachments:
                    for attachment in message.attachments:
                        if attachment.object == 'file':
                            file_id = attachment.file_id
                            # Download the file
                            file_content = openai.File.download(file_id).read()
                            # Check the file type and update chat history accordingly
                            if attachment.filename.endswith(('.png', '.jpg', '.jpeg')):
                                # Convert image bytes to displayable format
                                image = Image.open(io.BytesIO(file_content))
                                buffered = io.BytesIO()
                                image.save(buffered, format="PNG")
                                img_bytes = buffered.getvalue()
                                # Append image to chat history
                                st.session_state.chat_history[-1]['image'] = img_bytes
                            elif attachment.filename.endswith('.csv'):
                                # Read CSV into a dataframe and append to chat history
                                df = pd.read_csv(io.BytesIO(file_content))
                                st.session_state.chat_history[-1]['content'] += f"\n\n{df.to_html(index=False, escape=False)}"
                            else:
                                # Handle other file types as download buttons
                                st.session_state.chat_history[-1]['content'] += f"\n\n[Download {attachment.filename}](data:file/{attachment.filename.split('.')[-1]};base64,{base64.b64encode(file_content).decode()})"
        except Exception as e:
            st.error(f"Failed to handle assistant's attachments: {e}")
            st.stop()

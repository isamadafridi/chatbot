import streamlit as st
import google.generativeai as genai
from streamlit_chat import message # Keep this for displaying messages
from itertools import zip_longest # Keep this for iterating history

# --- Get API Key and Configure GenAI ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=google_api_key)
except KeyError:
    st.error("🚨 Google API Key not found. Please add `GOOGLE_API_KEY = \"YOUR_KEY_HERE\"` to your Streamlit secrets.", icon="🔑")
    st.stop()
except Exception as e:
    st.error(f"🚨 Error configuring Google GenAI: {e}", icon="🔥")
    st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Hope to Skill ChatBot - Gemini Native")
st.title("Mr. AI")

# --- Initialize Session State ---
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses
if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [] # Store history in the format genai expects

# --- System Prompt ---
# Define the system prompt content
SYSTEM_PROMPT_CONTENT = """your name is Mr.AI. You are an AI Technical Expert for Artificial Intelligence, here to guide and assist students with their AI-related questions and concerns. Please provide accurate and helpful information, and always maintain a polite and professional tone.

                1. Greet the user politely ask user name and ask how you can assist them with AI-related queries.
                2. Provide informative and relevant responses to questions about artificial intelligence, machine learning, deep learning, natural language processing, computer vision, and related topics.
                3. you must Avoid discussing sensitive, offensive, or harmful content. Refrain from engaging in any form of discrimination, harassment, or inappropriate behavior.
                4. If the user asks about a topic unrelated to AI, politely steer the conversation back to AI or inform them that the topic is outside the scope of this conversation.
                5. Be patient and considerate when responding to user queries, and provide clear explanations.
                6. If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
                7. Do Not generate the long paragarphs in response. Maximum Words should be 100.

                Remember, your primary goal is to assist and educate students in the field of Artificial Intelligence. Always prioritize their learning experience and well-being."""


# --- Initialize the Gemini Model ---
try:
    # Choose your model - 'gemini-pro' is widely available.
    # 'gemini-1.5-flash' is faster and cheaper for many tasks.
    # 'gemini-1.5-pro-latest' might offer better capabilities but might require specific access/region.
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        # For models supporting it (like 1.5 Pro/Flash), you could use:
        # system_instruction=SYSTEM_PROMPT_CONTENT
        )

    # Configuration for the generation
    generation_config = genai.GenerationConfig(
        temperature=0.5,
        max_output_tokens=100 # Control response length
        # top_p=0.9, # You can add other parameters as needed
        # top_k=40
    )

    # Safety settings (Optional, but recommended)
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    # Start a chat session (manages history automatically)
    # We'll manually manage history to integrate with session_state better for this example
    # chat_session = model.start_chat(history=st.session_state.chat_history)

except Exception as e:
    st.error(f"🚨 Error initializing Gemini model: {e}", icon="🔥")
    st.stop()


def format_chat_history():
    """
    Formats the Streamlit session state history ('past' and 'generated')
    into the list format required by genai's generate_content.
    Also includes the system prompt.
    """
    formatted_history = []

    # Add the system prompt as the initial context (acting like the first turn)
    # Gemini models often work well by integrating system instructions here
    # We put the system prompt as the first 'user' message context,
    # and the expected first 'model' response is a greeting.
    formatted_history.append({'role': 'user', 'parts': [SYSTEM_PROMPT_CONTENT]})
    # Add a placeholder initial greeting from the model if no history exists yet
    if not st.session_state['past']:
         formatted_history.append({'role': 'model', 'parts': ["Hello! I am AI Mentor. What is your name and how can I help you with AI today?"]})


    # Interleave past user messages and generated AI responses
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            formatted_history.append({'role': 'user', 'parts': [human_msg]})
        if ai_msg is not None:
            formatted_history.append({'role': 'model', 'parts': [ai_msg]})

    # The last message should be the current user prompt, which isn't added here yet.
    # It will be added to the list just before calling generate_content.

    return formatted_history


def generate_response(user_prompt):
    """
    Generate AI response using the Google Generative AI model.
    """
    # Get the formatted history
    current_history = format_chat_history()

    # Add the current user prompt to the history
    current_history.append({'role': 'user', 'parts': [user_prompt]})

    try:
        # Generate response using the model
        response = model.generate_content(
            current_history,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False, # Set to True for streaming response
        )

        # Check for safety blocks or other issues
        if not response.candidates:
             # Check prompt feedback for block reason
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
            st.warning(f"Response blocked due to: {block_reason}. Please modify your prompt.", icon="⚠️")
            return "I cannot provide a response to that request due to safety guidelines."

        # Extract the text content
        ai_response_text = response.text

        # Update the full chat history in session state *after* successful generation
        st.session_state.chat_history = current_history # Save history *before* adding AI response
        # Add the AI response to the history for the *next* turn
        st.session_state.chat_history.append({'role': 'model', 'parts': [ai_response_text]})

        return ai_response_text

    except ValueError as ve:
         # Handle potential errors like invalid API keys or resource exhaustion during generation
         st.error(f"🚨 A value error occurred: {ve}", icon="🔥")
         # Check if it's related to history format
         if "alternating role" in str(ve).lower():
             st.error("Internal Error: Chat history roles are not alternating correctly.", icon="⚙️")
         return "Sorry, I encountered an issue processing the request history."
    except Exception as e:
        st.error(f"🚨 An unexpected error occurred during response generation: {e}", icon="🔥")
        # Log the history for debugging if needed
        # print("Error occurred with history:", current_history)
        return "Sorry, I encountered an unexpected error."


# --- Streamlit UI Elements ---

# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""

# Create a text input for user
st.text_input('YOU: ', key='prompt_input', on_change=submit, placeholder="Ask your AI question here...")

# --- Process User Input and Generate Response ---
if st.session_state.entered_prompt != "":
    # Get user query
    user_query = st.session_state.entered_prompt

    # Append user query to past queries (for display purposes)
    st.session_state.past.append(user_query)

    # Generate response
    with st.spinner("AI Mentor is thinking..."):
        output = generate_response(user_query)

    # Append AI response to generated responses (for display purposes)
    st.session_state.generated.append(output)

    # Clear the entered_prompt *after* processing
    st.session_state.entered_prompt = ""
    # No need to manually update chat_history here, generate_response does it.

    # Rerun to display the new message immediately
    st.rerun()


# --- Display Chat History ---
if st.session_state['generated']:
    # Create a container for the chat history
    with st.container():
        # Display messages, newest first
        # Note: zip_longest ensures we handle cases where generation might fail
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            # Display AI response using streamlit-chat message component
            message(st.session_state["generated"][i], key=str(i) + "_gen") # Added _gen suffix
            # Display user message using streamlit-chat message component
            # Make sure 'past' has an entry corresponding to this 'generated' index
            if i < len(st.session_state['past']):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# Optional: Add a clear button
if st.button("Clear Chat History"):
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['entered_prompt'] = ""
    st.session_state['chat_history'] = [] # Clear the genai history too
    st.rerun() # Rerun the app to reflect the cleared state
    #done
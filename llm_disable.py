import time
import os
from basereal import BaseReal
from logger import logger

# Generates a response from a Large Language Model (LLM) and streams it to a real-time system.
def llm_response(message, nerfreal:BaseReal):
    # Records the start time.
    start = time.perf_counter()
    # Imports the OpenAI client library.
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(
        # Initializes the OpenAI client with an API key from an environment variable.
        api_key=os.getenv("LLM__OPENAI_KEY") 
    )

    # Records the time after client initialization.
    end = time.perf_counter()
    logger.info(f"llm Time init: {end - start}s")

    completion = client.chat.completions.create(
        # Specifies the LLM model to use.
        model="gpt-4o-mini",
        # Defines the conversation history with system and user roles.
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': message}
        ],
        # Enables streaming the response.
        stream = True,
        # Includes token usage information in the stream.
        stream_options = {"include_usage": True}
    )
    
    # Initializes a string to accumulate the streamed text.
    result = ""
    # Flag to track the first chunk of the response.
    first = True

    # Iterates through the streamed chunks.
    for chunk in completion:
        # Checks if the chunk contains a choice.
        if len(chunk.choices) > 0:

            # A commented out print statement to show the content.
                # print(chunk.choices[0].delta.content)

            # If this is the first chunk.        
            if first:
                # Records the time.
                end = time.perf_counter()
                # Logs the time to the first chunk.
                logger.info(f"llm Time to first chunk: {end-start}s")
                # Resets the flag.
                first = False

            # Gets the content of the current chunk.
            msg = chunk.choices[0].delta.content
            # Initializes a position tracker.
            lastpos = 0

            # A commented out line for splitting text.
                # msglist = re.split('[,.!;:，。！?]',msg)

            # Iterates through characters in the chunk to find punctuation marks.
            if msg:
                for i, char in enumerate(msg):
                    # Checks for various punctuation marks.
                    if char in ",.!;:，。！？：；" :
                        # Appends the text up to the punctuation.
                        result = result+msg[lastpos:i+1]
                        # Updates the position.
                        lastpos = i+1
                        # Checks if the accumulated text is long enough.
                        if len(result) > 10:
                            logger.info(result)
                            # Puts the sentence to the real-time system's TTS.
                            nerfreal.put_msg_txt(result)
                            # Resets the accumulated text.
                            result = ""

                # Appends any remaining text after the last punctuation.
                result = result+msg[lastpos:]

    # Records the end time after the loop finishes.
    end = time.perf_counter()
    logger.info(f"llm Time to last chunk: {end-start}s")
    # Puts any final remaining text to the real-time system's TTS.
    nerfreal.put_msg_txt(result)    
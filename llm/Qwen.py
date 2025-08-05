import os
import openai

'''
If `huggingface` connection fails, you can use `modelscope`.
`pip install modelscope`
'''
from modelscope import AutoModelForCausalLM, AutoTokenizer

# This line is commented out, indicating an alternative to use Hugging Face Transformers library.
    # from transformers import AutoModelForCausalLM, AutoTokenizer

# Sets an environment variable for CUDA. This can help with debugging CUDA errors by forcing synchronous execution, making it easier to pinpoint issues.
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class Qwen:
    def __init__(self, model_path="Qwen/Qwen-1_8B-Chat", api_base=None, api_key=None) -> None:
        # Currently, the API version is not implemented; it's similar to Linly-api. You can implement it if interested.
        
        # Default to local inference
        # Initializes a flag 'local' to True, indicating that by default, the model will run locally rather than through an API.
        self.local = True

        # Use OpenAPI if api_base and api_key are not empty
        if api_key is not None and api_base is not None:
            openai.api_base = api_base
            openai.api_key = api_key
            # Sets the 'local' flag to False, meaning API mode will be used.
            self.local = False
            # Exits the constructor early, as model loading is not needed for API mode.
            return

        # If not in API mode, calls 'init_model' to load the local Qwen model and its tokenizer.
        self.model, self.tokenizer = self.init_model(model_path)
        # Initializes an empty dictionary 'data', likely for storing conversation history or temporary data.
        self.data = {}

    # Defines a method to initialize the local Qwen model and tokenizer.
    def init_model(self, path="Qwen/Qwen-1_8B-Chat"):
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-1_8B-Chat",
                                                    # Automatically determines where to load the model (e.g., CPU, GPU).
                                                    device_map="auto",
                                                    # 'trust_remote_code=True' is necessary for custom model architectures.
                                                    # '.eval()' sets the model to evaluation mode (disables dropout, etc.).
                                                    trust_remote_code=True).eval()

        # Loads the tokenizer for the specified model path.
        # 'trust_remote_code=True' is also needed here.                                                    
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        return model, tokenizer

    def chat(self, question):
        # Prioritize calling Qwen OpenAPI
        if not self.local:
            # Request without streaming response
            response = openai.ChatCompletion.create(
                model="Qwen",
                messages=[
                    {"role": "user", "content": question}
                ],
                stream=False,
                stop=[]
            )
            return response.choices[0].message.content

        # Default local inference
        # Formats the question for local inference.
        self.data["question"] = f"{question} ### Instruction:{question}  ### Response:"

        try:
            # Calls the model's chat method.
            # It uses the tokenizer to prepare the input, passes the formatted question, and starts with no history.
            response, history = self.model.chat(self.tokenizer, self.data["question"], history=None)
            return response
        except:
            return "Sorry, your request has encountered an error. Please try again."

# Defines a test function
def test():
    llm = Qwen(model_path="Qwen/Qwen-1_8B-Chat")
    answer = llm.chat(question="如何应对压力？")
    print(answer)


if __name__ == '__main__':
    test()

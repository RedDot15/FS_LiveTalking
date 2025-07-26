import json
import requests

## This line is commented out, suggesting it was part of a previous implementation for database interaction, likely to retrieve chat history.
    # from core import content_db

class VllmGPT:

    def __init__(self, 
                host="192.168.1.3",
                port="8101",
                model="THUDM/chatglm3-6b",
                max_tokens="1024"):
        # Initializes the connection parameters for the VLLM server.
        self.host = host
        self.port = port
        self.model=model
        self.max_tokens=max_tokens

        # Constructs the full URL for the VLLM 'completions' API endpoint.
        # This endpoint is typically for raw text completion.
        self.__URL = "http://{}:{}/v1/completions".format(self.host, self.port)

        # Constructs the full URL for the VLLM 'chat completions' API endpoint.
        # This endpoint is typically for chat-like interactions with role-based messages.
        self.__URL2 = "http://{}:{}/v1/chat/completions".format(self.host, self.port)

    def chat(self,cont):
        # Initializes an empty list. This list was intended to hold chat history.
        chat_list = []

        # The following commented-out block indicates previous functionality for retrieving chat history
        # from a database (content_db) and formatting it for the API request.
            # contentdb = content_db.new_instance()
            # list = contentdb.get_list('all','desc',11)
            # answer_info = dict()
            # chat_list = []
            # i = len(list)-1
            # while i >= 0:
            #     answer_info = dict()
            #     if list[i][0] == "member":
            #         answer_info["role"] = "user"
            #         answer_info["content"] = list[i][2]
            #     elif list[i][0] == "fay":
            #         answer_info["role"] = "bot"
            #         answer_info["content"] = list[i][2]
            #     chat_list.append(answer_info)
            #     i -= 1

        # Define content
        content = {
            "model": self.model,
            "prompt":"Please reply briefly." +  cont,
            "history":chat_list}

        # Sets the target URL to the 'completions' endpoint.            
        url = self.__URL
        # Converts the Python dictionary 'content' into a JSON formatted string.
        req = json.dumps(content)
        
        # Sets the HTTP header to indicate that the request body is JSON.
        headers = {'content-type': 'application/json'}
        # Sends a POST request to the VLLM server with the JSON data.
        r = requests.post(url, headers=headers, data=req)
        # Parses the JSON response received from the server into a Python dictionary.
        res = json.loads(r.text)
        
        # Return
        return res['choices'][0]['text']

    def question2(self,cont):
        # Initializes an empty list, intended for chat history.
        chat_list = []
        # The following commented-out block is identical to the one in 'chat',
        # indicating the same un-used history retrieval logic.
            # contentdb = content_db.new_instance()
            # list = contentdb.get_list('all','desc',11)
            # answer_info = dict()
            # chat_list = []
            # i = len(list)-1
            # while i >= 0:
            #     answer_info = dict()
            #     if list[i][0] == "member":
            #         answer_info["role"] = "user"
            #         answer_info["content"] = list[i][2]
            #     elif list[i][0] == "fay":
            #         answer_info["role"] = "bot"
            #         answer_info["content"] = list[i][2]
            #     chat_list.append(answer_info)
            #     i -= 1

        # Defines content.            
        content = {
            "model": self.model,
            "prompt":"请简单回复我。" +  cont,
            "history":chat_list}
        # Sets the target URL to the 'chat completions' endpoint.
        url = self.__URL2
        # Converts the Python dictionary 'content' into a JSON formatted string.
        req = json.dumps(content)
        
        # Sets the HTTP header to indicate that the request body is JSON.
        headers = {'content-type': 'application/json'}
        # Sends a POST request to the VLLM server with the JSON data.
        r = requests.post(url, headers=headers, data=req)
        # Parses the JSON response received from the server into a Python dictionary.
        res = json.loads(r.text)
        
        # Return
        return res['choices'][0]['message']['content']


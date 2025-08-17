from llm_client import LLMService
from llm_client import LLMServiceInput
from llm_client import LLMSetting
from llm_client import CompletionMessage
from llm_client import MessageRole

import os
from dotenv import load_dotenv

load_dotenv()

open_ai_key = os.getenv('OPENAI_API_KEY')

llm_setting = LLMSetting(
    open_ai_key=open_ai_key,
    model_name='gpt-4o-mini'
)

llm = LLMService(settings=llm_setting)

message = [CompletionMessage(
    content='Xin chào',
    role=MessageRole.SYSTEM
)]

result = llm.process(
    LLMServiceInput(
        message=message,
        model='gpt-4o-mini',
    )
)

print(result)
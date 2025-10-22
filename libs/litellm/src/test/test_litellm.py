from __future__ import annotations

import os

from dotenv import load_dotenv
from litellm import CompletionMessage
from litellm import LiteLLMEmbeddingInput
from litellm import LiteLLMChatInput # Đã bao gồm 'tools'
from litellm import LiteLLMService
from litellm import MessageRole

load_dotenv()

LITELLM__URL = os.getenv('LITELLM__URL')
LITELLM__MODEL = os.getenv('LITELLM__MODEL', 'gemini-2.0-flash')

LITELLM__TEMPERATURE = float(os.getenv('LITELLM__TEMPERATURE', 0.7))
LITELLM__TOP_P = float(os.getenv('LITELLM__TOP_P', 0.9))
LITELLM__FREQUENCY_PENALTY = float(os.getenv('LITELLM__FREQUENCY_PENALTY', 0.0))
LITELLM__PRESENCE_PENALTY = float(os.getenv('LITELLM__PRESENCE_PENALTY', 0.0))
LITELLM__MAX_COMPLETION_TOKENS = int(os.getenv('LITELLM__MAX_COMPLETION_TOKENS', 4096))
LITELLM__N = int(os.getenv('LITELLM__N', 1))

LITELLM__EMBEDDING_MODEL = os.getenv('LITELLM__EMBEDDING_MODEL', 'embedding-004')
LITELLM__ENCODING_FORMAT = os.getenv('LITELLM__ENCODING_FORMAT', 'float')
LITELLM__DIMENSIONS = int(os.getenv('LITELLM__DIMENSIONS', 768))
LITELLM__MAX_LENGTH = int(os.getenv('LITELLM__MAX_LENGTH', 2048))

class TestLiteLLMService:
    
    def __init__(self):

        self.litellm = LiteLLMService(
            url=LITELLM__URL,
            model=LITELLM__MODEL,
            embedding_model=LITELLM__EMBEDDING_MODEL,
            frequency_penalty=LITELLM__FREQUENCY_PENALTY,
            n=LITELLM__N,
            presence_penalty=LITELLM__PRESENCE_PENALTY,
            temperature=LITELLM__TEMPERATURE,
            top_p=LITELLM__TOP_P,
            max_completion_tokens=LITELLM__MAX_COMPLETION_TOKENS,
            encoding_format=LITELLM__ENCODING_FORMAT,
            dimensions=LITELLM__DIMENSIONS,
            max_length=LITELLM__MAX_LENGTH,
        )

    def test_rewrite(self) -> None:
        print('\n--- Testing Rewrite ---')
        message: list[CompletionMessage] = [
            CompletionMessage(
                role=MessageRole.SYSTEM,
                content='You are helpful Slide Generator, Your task is rewrite query from user to more clearly understanding',
            ),
            CompletionMessage(
                role=MessageRole.USER,
                content='Tôi hoạt động ở PAYT Club.',
            ),
        ]
        
        with self.litellm.client as client:

            result = self.litellm.chat(
                client=client,
                inputs=LiteLLMChatInput(
                    message=message,
                    model=LITELLM__MODEL,
                )
            )
            print(result)

    def test_embedding(self):
        print('\n--- Testing Embedding ---')
        print('Embedding input models created successfully!')
        
        with self.litellm.client as client:

            result = self.litellm.embedding(client=client, inputs=LiteLLMEmbeddingInput(
                input='Hello, world! This is a test text for embedding.',
                encoding_format=LITELLM__ENCODING_FORMAT,
                count_tokens=True,
                embedding_model=LITELLM__EMBEDDING_MODEL,
                dimensions=LITELLM__DIMENSIONS
            ))
            
            print(f"Embedding vector length: {len(result.vector)}")

    # --- PHẦN ĐƯỢC THÊM MỚI ---
    def test_tool_call(self) -> None:
        """
        Test khả năng gọi tool của LLM.
        """
        print('\n--- Testing Tool Call ---')
        
        # 1. Định nghĩa các tool theo chuẩn OpenAI
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Lấy thông tin thời tiết hiện tại ở một địa điểm cụ thể",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Tên thành phố, ví dụ: Hanoi, Ho Chi Minh City",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        
        # 2. Định nghĩa message yêu cầu sử dụng tool
        message: list[CompletionMessage] = [
            CompletionMessage(
                role=MessageRole.USER,
                content="Thời tiết ở Hà Nội bây giờ thế nào?",
            ),
        ]
        
        # 3. Gọi service với tham số 'tools'
        # Giả định: LiteLLMService và LiteLLMChatInput đã được cập nhật
        # để xử lý 'tools' một cách chính xác (như trong LiteLLMToolService)
        with self.litellm.client as client:
            try:
                result = self.litellm.chat(
                    client=client,
                    inputs=LiteLLMChatInput(
                        message=message,
                        model=LITELLM__MODEL,
                        tools=tools,  # <-- Truyền danh sách tools vào đây
                    )
                )
                
                print("Tool Call Result:")
                print(result)
                # Nếu thành công, 'result.response' sẽ là một
                # List[ToolCall] (nếu LLM quyết định gọi tool)
                # hoặc một 'str' (nếu LLM trả lời bằng văn bản).

            except Exception as e:
                print(f"Error during tool call test: {e}")
                print("\n*** LƯU Ý QUAN TRỌNG ***")
                print("Đã xảy ra lỗi. Điều này RẤT CÓ THỂ do class 'LiteLLMChatService'")
                print("gốc (ở prompt đầu tiên) của bạn không được thiết kế để:")
                print("  1. Truyền 'tools' vào payload request.")
                print("  2. Xử lý 'tool_calls' trong response.")
                print("Bạn cần sử dụng logic từ 'LiteLLMToolService' (ở prompt thứ hai) để test này hoạt động.")

if __name__ == '__main__':
    litellm = TestLiteLLMService()
    
    litellm.test_rewrite()
    litellm.test_embedding()
    litellm.test_tool_call()
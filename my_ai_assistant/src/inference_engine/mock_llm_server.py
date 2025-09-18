class MockLLMServer:
    def ainvoke(self, prompt: str) -> str:
        return {
            "role": "assistant",
            "content": "This is a mock response from the LLM client."
        }
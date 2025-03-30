from datetime import datetime
import pprint
from src.core.my_ai_assistant import MyAIAssistant
from src.inference_engine.inference_processor import InferenceProcessor


message_history = [
    {
        "role": "user",
        "timestamp": "2024-01-26T10:00:00Z",
        "content": [
            {
                "type": "text",
                "text": "Hello! I'm trying to understand the basics of quantum computing. Can you help me?"
            }
        ]
    },
    {
        "role": "assistant",
        "timestamp": "2024-01-26T10:00:30Z",
        "content": [
            {
                "type": "text",
                "text": "Hi there! Absolutely. I can definitely help with that. Quantum computing is a fascinating field. Where would you like to start? Perhaps with qubits and superposition?"
            }
        ]
    },
    {
        "role": "user",
        "timestamp": "2024-01-26T10:01:00Z",
        "content": [
            {
                "type": "text",
                "text": "Yes, let's start with qubits. What makes them different from regular bits?"
            }
        ]
    },
    {
        "role": "assistant",
        "timestamp": "2024-01-26T10:01:45Z",
        "content": [
            {
                "type": "text",
                "text": "The key difference is that a bit can be either 0 or 1, whereas a qubit can be 0, 1, or a *superposition* of both. Think of it like a coin. A regular bit is heads or tails. A qubit is spinning in the air, existing in both states simultaneously until you observe it."
            }
        ]
    },
    {
        "role": "user",
        "timestamp": "2024-01-26T10:02:15Z",
        "content": [
            {
                "type": "text",
                "text": "Okay, that makes some sense. So, what does 'observing' a qubit do?"
            }
        ]
    },
    {
        "role": "assistant",
        "timestamp": "2024-01-26T10:03:00Z",
        "content": [
            {
                "type": "text",
                "text": "When you 'observe' a qubit, it collapses from its superposition into either a 0 or a 1. This collapse is probabilistic, meaning there's a chance it will be 0 and a chance it will be 1, based on its initial superposition state. This is where quantum algorithms gain their power."
            }
        ]
    },
    {
        "role": "user",
        "timestamp": "2024-01-26T10:03:45Z",
        "content": [
            {
                "type": "text",
                "text": "What about quantum entanglement? I keep hearing that mentioned."
            }
        ]
    },
    {
        "role": "assistant",
        "timestamp": "2024-01-26T10:04:30Z",
        "content": [
            {
                "type": "text",
                "text": "Entanglement is where two or more qubits become linked together. The state of one qubit instantly influences the state of the other, regardless of the distance between them. It's like having two coins flipped at the same time – you might not know what either one is until you look, but if they are entangled, you know that if one is heads, the other is tails (or vice-versa, depending on how they were entangled)."
            }
        ]
    },
    {
        "role": "user",
        "timestamp": "2024-01-26T10:05:15Z",
        "content": [
            {
                "type": "text",
                "text": "So, if I measure one entangled qubit and find it's 0, I instantly know the other is 1?"
            }
        ]
    },
    {
        "role": "assistant",
        "timestamp": "2024-01-26T10:06:00Z",
        "content": [
            {
                "type": "text",
                "text": "Exactly! That instant correlation is the key to entanglement. However, it's important to remember that entanglement cannot be used to transmit information faster than light. You can't control the outcome of the measurement on the first qubit to send a specific signal."
            }
        ]
    },
    {
        "role": "user",
        "timestamp": "2024-01-26T10:06:45Z",
        "content": [
            {
                "type": "text",
                "text": "Okay, that's good to know. What are some practical applications of quantum computing?"
            }
        ]
    },
    {
        "role": "assistant",
        "timestamp": "2024-01-26T10:07:30Z",
        "content": [
            {
                "type": "text",
                "text": "Quantum computing has potential in many areas, including drug discovery (simulating molecular interactions), materials science (designing new materials), financial modeling (optimizing portfolios), cryptography (breaking existing codes and creating new ones), and machine learning (developing faster algorithms)."
            }
        ]
    },
    {
        "role": "user",
        "timestamp": "2024-01-26T10:08:15Z",
        "content": [
            {
                "type": "text",
                "text": "Wow, that's a lot! Are quantum computers actually being used today, or are they still theoretical?"
            }
        ]
    },
    {
        "role": "assistant",
        "timestamp": "2024-01-26T10:09:00Z",
        "content": [
            {
                "type": "text",
                "text": "While still in their early stages, quantum computers are being actively researched and developed by companies like Google, IBM, and Microsoft. They are not yet at the point where they can solve complex, real-world problems that classical computers can't, but they are showing promise for specific tasks and represent a rapidly evolving field."
            }
        ]
    },
    {
        "role": "user",
        "timestamp": "2024-01-26T10:09:30Z",
        "content": [
            {
                "type": "text",
                "text": "Thanks for explaining all this! It's much clearer now."
            }
        ]
    },
    {
        "role": "assistant",
        "timestamp": "2024-01-26T10:09:50Z",
        "content": [
            {
                "type": "text",
                "text": "No worries."
            }
        ]
    },
]



def test_process_and_create_chat_generation():
    # test 1 - only inference without speech engine or conversation hitory processor
    
    inference_processor = InferenceProcessor()

    my_ai_assistant = MyAIAssistant(inference_processor=inference_processor, mock_conversation_history=message_history)
    
    user_message = {
        "role": "user",
        "timestamp": datetime.now(),
        "content": [
            {
                "type": "text",
                "text": "Can you summarize our conversation so far?"
            }
        ]
    }
    response = my_ai_assistant.create_chat_completion(
        message=user_message,
        is_api_request=False,
        is_audio_requested_in_api_response=False,
    )
    pprint.pprint(response)
    




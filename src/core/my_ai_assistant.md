```mermaid
classDiagram
    class MyAIAssistant {
        +CHUNK: int
        +FORMAT: pyaudio.paInt16
        +CHANNELS: int
        +RATE: int
        +THRESHOLD: int
        +CHARS_PER_SECOND: int
        -computer: Thread
        -config: dict
        -inference_processor: object
        -conversation_history_engine: object
        -voice_reply_enabled: bool
        -tts_engine: object
        -asr_engine: object
        -interruption: bool
        -spoken_text: str
        -remaining_text: str
        -gui_interface: gr.Blocks
        -gui_visualizer: SpectrogramWidget
        +__init__()
        +split_text_into_chunks()
        +listen()
        +voice_reply()
        +monitor_audio_interruption()
        +backup_conversation_history()
        +exit_gracefully()
        +create_gradio_ui()
        +update_chat_interface()
        +__run__()
        +run()
        +test_voice_reply()
    }

```


```mermaid
flowchart TB
    subgraph MyAIAssistant
        A[Initialize] --> B[Setup Components]
        B --> C[Launch GUI]
        
        subgraph Components
            D[Speech Recognition]
            E[Text-to-Speech]
            F[Conversation History]
            G[GUI Interface]
            H[Audio Visualization]
        end
        
        subgraph Main Loop
            I[Listen for Input] --> J{Valid Input?}
            J -->|Yes| K[Process Message]
            J -->|No| I
            K --> L[Get AI Response]
            L --> M{Voice Enabled?}
            M -->|Yes| N[Voice Reply]
            M -->|No| O[Text Reply]
            N --> P[Update GUI]
            O --> P
            P --> I
        end
        
        subgraph Interruption Handling
            Q[Monitor Thread] --> R{Space Pressed?}
            R -->|Yes| S[Stop TTS]
            R -->|No| Q
        end
        
        subgraph Cleanup
            T[Exit Signal] --> U[Stop Threads]
            U --> V[Release Resources]
            V --> W[Backup History]
        end
    end
```

```mermaid
sequenceDiagram
    participant User
    participant GUI
    participant ASR
    participant LLM
    participant Assistant
    participant Tool
    participant TTS
    participant History
    participant Backup_DB
    participant Vector_DB

    User->>GUI: Input Message
    GUI->>LLM: Process Input
    opt Voice Input
        User->>ASR: Record Audio
        ASR->>LLM: Transcribed Text
    end
    Assistant->>History: Store Message
    History->>Assistant: Recent Messages
    Assistant->>LLM: Process Prompt Messages
    LLM->>Assistant: Generate Reply

    opt Tool Call
        LLM->>Tool: Tool/Function/Agent Call
        Tool->>Vector_DB: Store Tool/Function/Agent Inner Process Responses
        Tool->>History: Store Tool/Function/Agent Conclusive Response
        Tool->>Backup_DB: Store Tool/Function/Agent Conclusive Response
        Tool->>LLM: Tool/Function/Agent Response
        LLM->>Assistant: Generate Reply
    end
    
    opt Voice Reply
        Assistant->>TTS: Convert to Speech
        TTS->>User: Play Audio
        User->>Assistant: Interrupt (Optional)
    end
    Assistant->>GUI: Update Interface (Show Reply)
    Assistant->>History: Store Reply
    Assistant->>Backup_DB: Store Reply
```

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Listening: Space Key Pressed
    Listening --> Processing: Space Key Released
    Processing --> Responding: Valid Input
    Processing --> Idle: Invalid Input
    
    state Responding {
        [*] --> GeneratingResponse
        GeneratingResponse --> VoiceReply: Voice Enabled
        GeneratingResponse --> TextReply: Voice Disabled
        VoiceReply --> [*]: Complete/Interrupted
        TextReply --> [*]
    }
    
    Responding --> Idle
    Idle --> [*]: Exit Signal
```



# process_and_create_chat_generation Flow

```mermaid
flowchart TD
    %% Initial message handling
    A[Start] --> B{Vision\nEnabled?}
    B -->|No| C[Create Basic Message]
    B -->|Yes| D[Create Vision Message]
    
    %% Vision message processing
    D -->|Screenshot| E1[Add Screenshot Text]
    D -->|Camera Feed| E2[Add Camera Feed Text]
    D -->|Video File| E3[Add Video File Text]
    D -->|Image URLs| E4[Process Image URLs]
    E1 & E2 & E3 & E4 --> F[Format Vision Message]
    
    C & F --> G{Check History\nEngine}
    
    %% Conversation history handling
    G -->|Available| H1[Get Recent History]
    G -->|Not Available| H2[Create Empty History]
    H1 & H2 --> I[Add User Message]
    I --> J[Store in Database]
    
    %% Generate AI Response
    I --> K[Generate AI Response]
    
    %% Process Response
    K --> L{Voice Reply\nEnabled?}
    
    %% Handle different response types
    L -->|No| M1[Return Text Response]
    L -->|Yes| N{Is API Request?}
    
    N -->|Yes & Audio| O1[Generate Audio]
    N -->|Yes & No Audio| M1
    N -->|No| O2[Generate Voice Reply]
    
    %% Voice reply processing
    O2 --> P1[Process Spoken Part]
    O2 --> P2[Handle Interruptions]
    
    %% Store responses
    P1 --> Q1[Store Spoken Reply]
    P2 --> Q2[Store Unspoken Reply]
    M1 --> Q3[Store Text Reply]
    
    %% Final response
    Q1 & Q2 & Q3 --> R[Return Final Response]
    O1 --> R
    R --> S[End]

    %% Styling
    classDef process fill:#f0f0f0,stroke:#333,stroke-width:2px
    classDef decision fill:#fffbd6,stroke:#333,stroke-width:2px
    classDef storage fill:#e1f3ff,stroke:#333,stroke-width:2px
    
    class A,C,D,E1,E2,E3,E4,F,I,K,M1,O1,O2,P1,P2 process
    class B,G,L,N decision
    class J,Q1,Q2,Q3 storage
```

## Flow Description

### 1. Message Processing
- Checks if vision features are enabled
- Creates appropriate message format based on input type
- Handles different media types (screenshots, camera feed, video files, images)

### 2. Conversation History
- Verifies availability of conversation history engine
- Either retrieves existing conversation or creates new context
- Stores user message in database

### 3. AI Response Generation
- Processes conversation through inference engine
- Generates AI response based on context

### 4. Response Processing
Three main paths:
1. **Text Only**
   - Direct text response without voice
   - Stores in history and returns

2. **API Request**
   - Handles audio generation if requested
   - Returns combined audio and text response

3. **Voice Reply**
   - Generates voice output
   - Handles potential interruptions
   - Manages both spoken and unspoken parts
   - Stores all components in history

### 5. Final Response
- Formats appropriate response type
- Ensures all responses are stored in conversation history
- Returns formatted message segment
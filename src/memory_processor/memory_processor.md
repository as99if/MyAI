```mermaid
sequenceDiagram
    participant Assistant
    participant Tool
    participant History
    participant Backup_DB
    participant Vector_DB
    Assistant->>History: Store User Message
    Assistant->>Backup_DB: Store User Message
    Assistant->>History: Store Reply
    Assistant->>Backup_DB: Store Reply
    Tool->>Backup_DB: Store Tool's Conclusive Reply
    Tool->>Vector_DB: Store Tool's Process Responses

    opt Conversation Summarizer
        History->>History: Remove history (by date limit)
        Backup_DB->>History: Summarised Message History
    end
```

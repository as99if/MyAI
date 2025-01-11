import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from datetime import datetime
from src.inference_engine import InferenceEngine

"""
    
    here, the _summarize_old_conversations method summarizes the first half of the history.. change it, so that it summarizes only the last two weeks of conversation from the timestamp in each conversation segment.

this is an example of conversation history
"""
class ConversationSummarizer:
    def __init__(self, engine: InferenceEngine, conversation_history_service):
        self.engine = engine
        self.conversation_history_service = conversation_history_service
        self.summarization_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._stop_event = threading.Event()

    def start_summarization(self):
        """Start the summarization process in a separate thread"""
        self.summarization_thread = threading.Thread(
            target=self._run_summarization,
            daemon=True  # Make it a daemon thread so it exits when the main program exits
        )
        self.summarization_thread.start()

    def stop_summarization(self):
        """Stop the summarization process"""
        self._stop_event.set()
        if self.summarization_thread:
            self.summarization_thread.join(timeout=2)

    def _run_summarization(self):
        """Run the summarization process"""
        try:
            # Submit the summarization task to the executor
            future = self.executor.submit(
                self._summarize_old_conversations
            )
            
            # Wait for the result with a timeout
            future.result(timeout=300)  # 5-minute timeout
            
        except Exception as e:
            print(f"Error in summarization thread: {e}")
        finally:
            self.executor.shutdown(wait=False)

    def _summarize_old_conversations(self) -> None:
        """
        Summarize and update the first half of conversation history
        """
        try:
            # backup conversation history
            # Get all conversations
            conversations = self.conversation_history_service.get_conversation_history_backup()
            if not conversations:
                return

            # Calculate midpoint
            midpoint = len(conversations) // 2
            if midpoint == 0:
                return

            # Get the first half of conversations
            old_conversations = conversations[:midpoint]
            
            # Format conversations for summarization
            conversation_text = ""
            for entry in old_conversations:
                timestamp = entry.get('datetime', 'unknown time')
                entry_type = entry.get('type', 'unknown')
                text = entry.get('text', '')
                conversation_text += f"[{timestamp}] {entry_type}: {text}\n"

            # Prepare prompt for summarization
            summarization_prompt = f"""
            Please provide a concise summary of the following conversations while preserving key information and context.
            Keep important details, decisions, and outcomes.

            Conversations to summarize:
            {conversation_text}

            Provide the summary in a format that can be stored as a single conversation entry.
            """

            # Get summary using the inference engine
            summary_result = self.engine.infer(summarization_prompt)
            if "error" in summary_result:
                print(f"Error summarizing conversations: {summary_result['error']}")
                return

            summary = summary_result["response"]

            # Create a new summary entry
            summary_entry = {
                'type': 'summary',
                'datetime': datetime.now().isoformat(),
                'text': summary,
                'metadata': {
                    'summarized_entries': midpoint,
                    'start_date': old_conversations[0].get('datetime'),
                    'end_date': old_conversations[-1].get('datetime')
                }
            }

            # Remove old conversations and add summary
            self.conversation_store.replace_conversations(
                start_index=0,
                end_index=midpoint,
                new_entry=summary_entry
            )

            print(f"Successfully summarized {midpoint} old conversations")

        except Exception as e:
            print(f"Error during conversation summarization: {e}") 
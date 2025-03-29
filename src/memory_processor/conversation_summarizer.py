import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from datetime import datetime

"""
    
A separat service to summarise conversations on regular basis scheduled.
"""
class ConversationSummarizer:
    def __init__(self, inference_engine, conversation_history_engine, app_config):
        self.inference_engine = inference_engine
        self.conversation_history_engine = conversation_history_engine
        self.app_config = app_config
        self.summarization_thread: Optional[threading.Thread] = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._stop_event = threading.Event()

    def start_summarization_thread(self):
        """Start the summarization process in a separate thread"""
        last_summarization_date = self.app_config.get("last_summarization_date", None)
        timedelta = datetime.now() - datetime.fromisoformat(last_summarization_date) if last_summarization_date else None
        # if last summarization date is more than 2 weeks ago, then summarize
        if timedelta and timedelta.days > 14:
            self.app_config["last_summarization_date"] = datetime.now().isoformat()
            self.app_config.save()
        
            self.summarization_thread = threading.Thread(
                target=self._run_summarization,
                daemon=True  # Make it a daemon thread so it exits when the main program exits
            )
            
            self.summarization_thread.start()
        else:
            print("Last summarisation date")

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
            future.result(timeout=600)  # 10-minute timeout
            
        except Exception as e:
            print(f"Error in summarization thread: {e}")
        finally:
            self.executor.shutdown(wait=False)

    
    def _split_conversation_by_a_point(backup_conversation):
        # Convert string dates to datetime objects and sort
        date_format = '%d-%m-%Y'
        dated_items = [(datetime.strptime(date, date_format), date, items) 
                    for date, items in backup_conversation.items()]
        dated_items.sort(key=lambda x: x[0])
        
        # Find midpoint
        mid_idx = len(dated_items) // 3
        
        # Split into two dictionaries
        first_part = {item[1]: item[2] for item in dated_items[:mid_idx]}
        second_part = {item[1]: item[2] for item in dated_items[mid_idx:]}
        
        return first_part, second_part

    def _summarize_old_conversations(self) -> None:
        """
        Summarize and update the first half of conversation history
        """
        try:
            # backup conversation history
            # Get all conversations
            conversations, backup_conversations = self.conversation_history_engine._get_all_data()
            if not backup_conversations or backup_conversations == []:
                return
            old_conversations, newer_conversations = self._split_conversation_by_a_point(backup_conversations)
            summarized_entries = len(old_conversations)
            
            # Format conversations for summarization
            old_conversations = json.dumps(old_conversations, indent=2)

            with open("src/prompts/system_prompts.json", "r") as f:
                self.system_prompt = json.load(f)
            # Prepare prompt for summarization
            summarization_prompt = f"{self.systm_prompt['summarise']}\nCOnversation history:\n{old_conversations}"

            # Get summary using the inference engine
            response = self.inference_engine.infer(summarization_prompt)['choices'][0]['message']['content']
            summary_results = []
            
            # re summarise
            if len(response['choices']) <= 3:
                summary_results.append(response['choices'][0]['message']['content'])

            summarization_prompt = f"{self.systm_prompt['secondary_summarise']}\nConversation summaries:\n{json.dumps(summary_results)}"
            summary_result = self.inference_engine.infer(summarization_prompt)['choices'][0]['message']['content']
            
            if "error" in summary_result:
                print(f"Error summarizing conversations: {summary_result['error']}")
                return

            summary = f"Summary of the conversation between {old_conversations[0].get('datetime')} to {old_conversations[-1].get('datetime')}: " + summary_result
            metadata = {
                    'summarized_entries': summarized_entries,
                    'start_date': old_conversations[0].get('datetime'),
                    'end_date': old_conversations[-1].get('datetime')
                }
            # Create a new summary entry
            summary_entry = [{
                'type': 'summary',
                'role': 'computer',
                'timestamp': datetime.now().isoformat(),
                'content': f'{summary} \m Metadata: {metadata}',
                'metadata': metadata
            }]
            end_date = old_conversations[-1].get('datetime').strftime('%d-%m-%Y')
            conversations = { end_date: summary_entry }
            conversations = conversations.extend(newer_conversations)

            # Refresh conversation history (not backup)
            self.conversation_history_engine.replace_conversation_history(conversations)
            
            print(f"Successfully summarized {summarized_entries} old conversations")

        except Exception as e:
            print(f"Error during conversation summarization: {e}") 
            
            
    async def summarise_and_process_conversation(self, date_range: int = 7, keep_alive: int = 7) -> str:
        """
        Summarize a conversation into a single sentence.
        
        Args:
            conversation (Dict[str, Any]): Conversation dictionary
            
        Returns:
            str: Summarized conversation
        """
        try:

            
            self._summarize_old_conversations()
            
            
        except Exception as e:
            self.logger.error(f"Error summarizing conversation: {str(e)}")
            return ""
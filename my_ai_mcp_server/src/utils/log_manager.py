import logging
from typing import List, Optional
from datetime import datetime

class LoggingManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggingManager, cls).__new__(cls)
            cls._instance.log_messages = []
            cls._instance.callbacks = []
        return cls._instance
    
    def add_message(self, message: str, level: str = "INFO", source: str = "system"):
        """Add a log message and notify all subscribers"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if level != "INFO":
            log_entry = f"[{timestamp}] [{level}] [{source}] {message}"
        else:
            log_entry = f"[{timestamp}] [{source}] {message}\n......"
        print("log_entry: ", log_entry)
        self.log_messages.append(log_entry)
        self._notify_subscribers()
    
    def get_logs(self) -> str:
        """Get all logs as a single string"""
        return "\n".join(self.log_messages)
    
    def subscribe(self, callback):
        """Subscribe to log updates"""
        self.callbacks.append(callback)
    
    def _notify_subscribers(self):
        """Notify all subscribers of log updates"""
        for callback in self.callbacks:
            callback(self.get_logs())
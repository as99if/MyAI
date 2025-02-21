from langchain_openai import ChatOpenAI

from src.utils.utils import load_config
from browser_use import Agent, Browser, BrowserConfig


class BrowserUse:
    def __init__(self, query):
        self.config = load_config()
        self.query = query
        self.llm_client=ChatOpenAI(base_url=f"http://{self.config.get('server_host')}:{self.config.get('server_port')}/v1")
        
        # llm or vlm?
        # Configure the browser to connect to your Chrome instance
        self.browser = Browser(
            config=BrowserConfig(
                # Specify the path to your Chrome executable
                chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  # macOS path
                # For Windows, typically: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
                # For Linux, typically: '/usr/bin/google-chrome'
            )
        )

        # await browser.close()

        
    async def ask_browser_user_agent(self, query):
        # get context from db
        # get_context()
        
        # if no context found
        context = await self.browser.new_context()
        
        self.agent = Agent(
                task=query,
                llm=self.llm_client,
                use_vision=True,
                save_conversation_path="logs/conversation.json",  # Save chat logs
                browser=self.browser,
                browser_context=context
            )
        result = await self.agent.run()
        # save context in db
        # save_context(context)
        
        # Access (some) useful information
        # result.urls()              # List of visited URLs
        # result.screenshots()       # List of screenshot paths
        # result.action_names()      # Names of executed actions
        # result.extracted_content() # Content extracted during execution
        # result.errors()           # Any errors that occurred
        # result.model_actions()     # All actions with their parameters
        
        return result
        
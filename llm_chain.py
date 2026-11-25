import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory

import config

load_dotenv()
logger = logging.getLogger(__name__)

class AdmissionsConversationManager:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        self.api_key = api_key or config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or config.GEMINI_MODEL
        self.temperature = temperature if temperature is not None else config.TEMPERATURE

        self.history = InMemoryChatMessageHistory()
        self.system_message = SystemMessage(content=config.SYSTEM_PROMPT)
        self.llm = None
        self._initialize_llm()

    def _initialize_llm(self):
        if not self.api_key:
            return
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                max_retries=2,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}", exc_info=True)
            self.llm = None

    def update_api_key(self, api_key: str):
        self.api_key = api_key
        self._initialize_llm()

    def add_user_message(self, content: str):
        self.history.add_user_message(content)

    def add_ai_message(self, content: str):
        self.history.add_ai_message(content)

    def get_messages(self) -> List[BaseMessage]:
        return self.history.messages

    def clear_memory(self):
        self.history.clear()

    def generate_response(self, student_input: str) -> str:
        if not student_input or not student_input.strip():
            return "I could not hear anything. Could you please repeat that?"

        if not self.llm:
            if not self.api_key:
                return "Gemini API key is missing. Please set your GEMINI_API_KEY in the sidebar or .env file."
            self._initialize_llm()
            if not self.llm:
                return "Unable to connect to Gemini LLM. Please check your network and API credentials."

        try:
            messages: List[BaseMessage] = [self.system_message]
            messages.extend(self.history.messages)
            messages.append(HumanMessage(content=student_input))

            ai_response = self.llm.invoke(messages)
            cleaned_text = ai_response.content.strip()

            self.history.add_user_message(student_input)
            self.history.add_ai_message(cleaned_text)

            return cleaned_text

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                return "Our admissions server is temporarily experiencing high demand. Please try asking again in a few moments."
            elif "INVALID_ARGUMENT" in err_msg or "api_key" in err_msg.lower():
                return "There seems to be an issue with the Gemini API key. Please verify your credentials."
            else:
                return f"I apologize, I encountered a brief glitch processing that: {err_msg[:120]}. Could you please ask again?"

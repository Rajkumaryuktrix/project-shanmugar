from typing import List, Dict, Optional, Union, Iterator
from groq import Groq
import os
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Represents a message in the chat conversation."""
    role: str
    content: str

class GroqChat:
    """A class to handle interactions with Groq's LLM models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the GroqChat client.
        
        Args:
            api_key (Optional[str]): Groq API key. If not provided, will look for GROQ_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Provide it as an argument or set GROQ_API_KEY environment variable.")
        
        self.client = Groq(api_key=self.api_key)
        logger.info("Initialized GroqChat client")

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stream: bool = False,
        stop: Optional[List[str]] = None
    ) -> Union[Dict, Iterator]:
        """
        Get a completion from the Groq model.
        
        Args:
            messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content' keys
            model (str): The model to use for completion
            temperature (float): Controls randomness in the output (0.0 to 1.0)
            max_tokens (int): Maximum number of tokens to generate
            top_p (float): Controls diversity via nucleus sampling
            stream (bool): Whether to stream the response
            stop (Optional[List[str]]): List of strings that stop the generation
            
        Returns:
            Union[Dict, Iterator]: The model's response or a stream of responses
            
        Raises:
            Exception: If the API call fails
        """
        try:
            logger.info(f"Attempting to get completion from model: {model}")
            
            completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                stop=stop
            )
            
            logger.info("Successfully received completion from model")
            return completion
            
        except Exception as e:
            logger.error(f"Error getting completion: {str(e)}")
            raise Exception(f"Failed to get completion: {str(e)}")

def example_usage():
    """Example usage of the GroqChat class."""
    try:
        # Initialize the chat client
        chat = GroqChat()
        
        # Create a conversation
        messages = [
            {
                "role": "user",
                "content": "Hi, Help me understand how this humanity was created is it a simulation or a real thing"
            }
        ]
        
        # Get the completion (non-streaming)
        response = chat.get_completion(messages)
        print(f"Response: {response.choices[0].message.content}")
        
        # Example of streaming response
        print("\nStreaming response:")
        stream_response = chat.get_completion(
            messages,
            stream=True
        )
        for chunk in stream_response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")
        
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")

if __name__ == "__main__":
    example_usage()
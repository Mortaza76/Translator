"""
Ollama Integration Module
Provides functionality to interact with Ollama and list available models.
"""

import ollama
from typing import List, Dict, Optional, Tuple
from language_codes import (
    LANGUAGE_NAME_TO_CODE,
    CODE_TO_LANGUAGE_NAME,
    SUPPORTED_LOCALE_CODES,
    extract_base_code,
    is_supported_locale,
    get_language_name_from_code
)


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    # Use comprehensive language mapping from language_codes module
    LANGUAGE_CODES = LANGUAGE_NAME_TO_CODE
    
    def __init__(self, host: str = "http://localhost:11434", default_model: str = "translategemma:4b"):
        """
        Initialize Ollama client.
        
        Args:
            host: Ollama server host URL (default: http://localhost:11434)
            default_model: Default model to use for generation (default: translategemma:4b)
        """
        self.host = host
        self.client = ollama.Client(host=host)
        self.default_model = default_model
    
    def _get_language_code(self, language: str) -> str:
        """
        Get language code from language name or locale code.
        
        Args:
            language: Language name (case-insensitive) or locale code (e.g., 'en-US', 'es-ES')
            
        Returns:
            Language code (ISO 639-1 format)
        """
        lang_lower = language.lower().strip()
        
        # If it's already a code or locale code, extract base code
        if '-' in lang_lower or len(lang_lower) <= 3:
            base_code = extract_base_code(lang_lower)
            if base_code in CODE_TO_LANGUAGE_NAME:
                return base_code
        
        # Try to find in language name mapping
        code = self.LANGUAGE_CODES.get(lang_lower)
        if code:
            return code
        
        # Fallback: try to extract code from the input
        if len(lang_lower) >= 2:
            return lang_lower[:2]
        
        return 'en'  # Default to English
    
    def _format_translate_prompt(self, text: str, source_lang: str, source_code: str, 
                                 target_lang: str, target_code: str) -> str:
        """
        Format the prompt according to TranslateGemma's expected format.
        
        TranslateGemma expects a specific prompt structure:
        - Professional translator role definition with source and target languages
        - Instruction to produce only the translation (no explanations)
        - Two blank lines before the text to translate
        
        Format:
        "You are a professional {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) translator. 
         Your goal is to accurately convey the meaning and nuances of the original {SOURCE_LANG} text 
         while adhering to {TARGET_LANG} grammar, vocabulary, and cultural sensitivities.
         Produce only the {TARGET_LANG} translation, without any additional explanations or commentary. 
         Please translate the following {SOURCE_LANG} text into {TARGET_LANG}:\n\n\n{TEXT}"
        
        Args:
            text: Text to translate
            source_lang: Source language name
            source_code: Source language code (ISO 639-1)
            target_lang: Target language name
            target_code: Target language code (ISO 639-1)
            
        Returns:
            Formatted prompt string following TranslateGemma's expected format
        """
        prompt = (
            f"You are a professional {source_lang} ({source_code}) to {target_lang} ({target_code}) translator. "
            f"Your goal is to accurately convey the meaning and nuances of the original {source_lang} text "
            f"while adhering to {target_lang} grammar, vocabulary, and cultural sensitivities.\n"
            f"Produce only the {target_lang} translation, without any additional explanations or commentary. "
            f"Please translate the following {source_lang} text into {target_lang}:\n\n\n"
            f"{text}"
        )
        return prompt
    
    def list_models(self) -> List:
        """
        List all available models in Ollama.
        
        Returns:
            List of Model objects containing model information
        """
        try:
            response = self.client.list()
            # The response is a ListResponse object with a 'models' attribute
            if hasattr(response, 'models'):
                return response.models
            elif isinstance(response, dict) and 'models' in response:
                return response['models']
            else:
                return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Optional:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model object containing model information or None if not found
        """
        try:
            models = self.list_models()
            for model in models:
                # Handle both Model objects and dictionaries
                model_name_attr = getattr(model, 'model', None) or model.get('model', None) or model.get('name', None)
                if model_name_attr == model_name:
                    return model
            return None
        except Exception as e:
            print(f"Error getting model info: {e}")
            return None
    
    def show_available_models(self) -> None:
        """
        Display all available models in a formatted way.
        """
        models = self.list_models()
        
        if not models:
            print("No models found. Make sure Ollama is running and you have models installed.")
            print("\nTo install a model, run: ollama pull <model_name>")
            print("Example: ollama pull llama2")
            return
        
        print(f"\n{'='*60}")
        print(f"Available Ollama Models ({len(models)} total)")
        print(f"{'='*60}\n")
        
        for i, model in enumerate(models, 1):
            # Handle both Model objects and dictionaries
            name = getattr(model, 'model', None) or model.get('model', None) or model.get('name', 'Unknown')
            size = getattr(model, 'size', None) or model.get('size', 0)
            modified_at = getattr(model, 'modified_at', None) or model.get('modified_at', 'Unknown')
            details = getattr(model, 'details', None) or model.get('details', None)
            
            # Format size
            if size and size > 0:
                if size >= 1024**3:
                    size_str = f"{size / (1024**3):.2f} GB"
                elif size >= 1024**2:
                    size_str = f"{size / (1024**2):.2f} MB"
                else:
                    size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = "Unknown"
            
            # Format modified_at
            if hasattr(modified_at, 'strftime'):
                modified_str = modified_at.strftime('%Y-%m-%d %H:%M:%S')
            else:
                modified_str = str(modified_at)
            
            print(f"{i}. Model: {name}")
            print(f"   Size: {size_str}")
            
            # Add additional details if available
            if details:
                param_size = getattr(details, 'parameter_size', None) or details.get('parameter_size', '')
                quant_level = getattr(details, 'quantization_level', None) or details.get('quantization_level', '')
                if param_size:
                    print(f"   Parameters: {param_size}")
                if quant_level:
                    print(f"   Quantization: {quant_level}")
            
            print(f"   Modified: {modified_str}")
            print()
        
        print(f"{'='*60}\n")
    
    def generate_text(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """
        Generate text using the specified model (default: translategemma:4b).
        
        Args:
            prompt: The input prompt/text to process
            model: Model name to use (defaults to self.default_model)
            **kwargs: Additional parameters for generation (stream, temperature, etc.)
            
        Returns:
            Generated text as a string
        """
        model_name = model or self.default_model
        
        try:
            response = self.client.generate(
                model=model_name,
                prompt=prompt,
                **kwargs
            )
            
            # Handle both streaming and non-streaming responses
            if kwargs.get('stream', False):
                # For streaming, collect all chunks
                full_response = ""
                for chunk in response:
                    if hasattr(chunk, 'response'):
                        full_response += chunk.response
                    elif isinstance(chunk, dict) and 'response' in chunk:
                        full_response += chunk['response']
                return full_response
            else:
                # Non-streaming response
                if hasattr(response, 'response'):
                    return response.response
                elif isinstance(response, dict) and 'response' in response:
                    return response['response']
                else:
                    return str(response)
        except Exception as e:
            print(f"Error generating text: {e}")
            raise
    
    def translate(self, text: str, source_language: str = "English", 
                  target_language: str = "English", source_code: Optional[str] = None,
                  target_code: Optional[str] = None, model: Optional[str] = None) -> str:
        """
        Translate text using translategemma:4b model with proper prompt format.
        
        Args:
            text: Text to translate
            source_language: Source language name (default: English). Can also be a locale code (e.g., 'en-US')
            target_language: Target language name (default: English). Can also be a locale code (e.g., 'es-ES')
            source_code: Source language code (ISO 639-1). If not provided, will be inferred from source_language.
            target_code: Target language code (ISO 639-1). If not provided, will be inferred from target_language.
            model: Model name to use (defaults to translategemma:4b)
            
        Returns:
            Translated text (only the translation, no explanations)
            
        Example:
            >>> client = OllamaClient()
            >>> translated = client.translate("Hello", source_language="English", 
            ...                                target_language="Spanish")
            >>> # Or with locale codes:
            >>> translated = client.translate("Hello", source_language="en-US", 
            ...                                target_language="es-ES")
        """
        model_name = model or self.default_model
        
        # Get language codes if not provided
        if source_code is None:
            source_code = self._get_language_code(source_language)
        else:
            # Extract base code if a locale code was provided
            source_code = extract_base_code(source_code)
            
        if target_code is None:
            target_code = self._get_language_code(target_language)
        else:
            # Extract base code if a locale code was provided
            target_code = extract_base_code(target_code)
        
        # Get proper language names for the prompt
        source_lang_name = get_language_name_from_code(source_code) if source_code else source_language
        target_lang_name = get_language_name_from_code(target_code) if target_code else target_language
        
        # Format prompt according to TranslateGemma's expected format
        prompt = self._format_translate_prompt(
            text=text,
            source_lang=source_lang_name,
            source_code=source_code,
            target_lang=target_lang_name,
            target_code=target_code
        )
        
        return self.generate_text(prompt, model=model_name)
    
    def generate_for_speech(self, text: str, model: Optional[str] = None) -> str:
        """
        Generate or process text optimized for text-to-speech conversion.
        This can be used to clean, format, or enhance text before TTS.
        
        Args:
            text: Input text to process
            model: Model name to use (defaults to translategemma:4b)
            
        Returns:
            Processed text ready for TTS
        """
        model_name = model or self.default_model
        
        # You can customize this prompt based on your needs
        prompt = f"Process the following text to make it clear and natural for speech: {text}"
        
        return self.generate_text(prompt, model=model_name)
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs) -> str:
        """
        Chat with the model using a conversation format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     Example: [{'role': 'user', 'content': 'Hello'}]
            model: Model name to use (defaults to translategemma:4b)
            **kwargs: Additional parameters for chat (stream, temperature, etc.)
            
        Returns:
            Model's response text
        """
        model_name = model or self.default_model
        
        try:
            response = self.client.chat(
                model=model_name,
                messages=messages,
                **kwargs
            )
            
            # Handle both streaming and non-streaming responses
            if kwargs.get('stream', False):
                full_response = ""
                for chunk in response:
                    if hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                        full_response += chunk.message.content
                    elif isinstance(chunk, dict):
                        if 'message' in chunk and 'content' in chunk['message']:
                            full_response += chunk['message']['content']
                return full_response
            else:
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    return response.message.content
                elif isinstance(response, dict) and 'message' in response:
                    return response['message'].get('content', '')
                else:
                    return str(response)
        except Exception as e:
            print(f"Error in chat: {e}")
            raise
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get a dictionary of all supported language codes and their names.
        
        Returns:
            Dictionary mapping language codes to language names
        """
        return CODE_TO_LANGUAGE_NAME.copy()
    
    def is_language_supported(self, language_code: str) -> bool:
        """
        Check if a language code or locale is supported.
        
        Args:
            language_code: Language code or locale (e.g., 'en', 'en-US', 'es-ES')
            
        Returns:
            True if the language is supported
        """
        return is_supported_locale(language_code) or extract_base_code(language_code) in CODE_TO_LANGUAGE_NAME
    
    def get_language_name(self, code: str) -> str:
        """
        Get the language name from a language code.
        
        Args:
            code: Language code (e.g., 'en', 'es', 'fr')
            
        Returns:
            Language name or the code itself if not found
        """
        return get_language_name_from_code(code)


def main():
    """Main function to demonstrate Ollama integration."""
    print("Initializing Ollama client...")
    client = OllamaClient()
    
    print("Fetching available models...")
    client.show_available_models()
    
    # Test translategemma:4b integration
    print(f"\n{'='*60}")
    print(f"Testing translategemma:4b Integration")
    print(f"{'='*60}\n")
    
    # Verify the model is available
    model_info = client.get_model_info("translategemma:4b")
    if model_info:
        print("✓ translategemma:4b is available and ready to use")
        print(f"  Default model set to: {client.default_model}\n")
        
        # Test a simple generation
        print("Testing text generation...")
        try:
            test_prompt = "Hello, how are you?"
            print(f"Input: {test_prompt}")
            response = client.generate_text(test_prompt)
            print(f"Output: {response}\n")
            print("✓ Model integration successful!\n")
        except Exception as e:
            print(f"⚠ Warning: Could not test generation - {e}\n")
    else:
        print("⚠ Warning: translategemma:4b not found in available models")
        print("  Make sure the model is installed: ollama pull translategemma:4b\n")
    
    # Also return the models list for programmatic use
    models = client.list_models()
    return models


if __name__ == "__main__":
    main()

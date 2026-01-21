"""
Multi-language translation using TranslateGemma (Ollama).
Supports translation between any two languages.
"""

from ollama_integration import OllamaClient


def translate_text_multi(text: str, source_language: str, target_language: str = "English") -> str:
    """
    Translate text from source language to target language using TranslateGemma.
    
    Args:
        text: Text to translate
        source_language: Source language name or code (e.g., "French", "fr", "French")
        target_language: Target language name or code (e.g., "Urdu", "ur", "ur-PK")
    
    Returns:
        Translated text in the target language
    """
    print(f"Translating from {source_language} to {target_language}...")
    
    try:
        client = OllamaClient()
        
        # Use the translate method which handles proper TranslateGemma prompt format
        translated_text = client.translate(
            text=text,
            source_language=source_language,
            target_language=target_language,
            model="translategemma:4b"
        )
        
        return translated_text.strip()
        
    except Exception as e:
        print(f"✗ Error during translation: {e}")
        raise

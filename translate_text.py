"""
Translate Urdu text to target language using TranslateGemma (Ollama).
Modular translation component that can be reused in the pipeline.
"""

from ollama_integration import OllamaClient
import sys


def translate_urdu_to_target(urdu_text: str, target_language: str = "English") -> str:
    """
    Translate Urdu text to a target language using TranslateGemma.
    
    This function uses the Ollama integration with TranslateGemma model
    to translate text from Urdu (ur-PK) to the specified target language.
    
    Args:
        urdu_text: Urdu text to translate (from speech-to-text)
        target_language: Target language name, code, or locale (default: "English")
                        Examples: "English", "en", "en-US", "Spanish", "es", "es-ES"
    
    Returns:
        Translated text in the target language (only the translation, no explanations)
    
    Example:
        >>> urdu_text = "آپ کیسے ہیں؟"
        >>> english = translate_urdu_to_target(urdu_text, "English")
        >>> print(english)  # "How are you?"
    """
    # Step 1: Initialize Ollama client
    # Uses translategemma:4b model by default with proper prompt format
    print("Initializing TranslateGemma model...")
    client = OllamaClient()
    print("✓ Model ready")
    print()
    
    # Step 2: Translate using TranslateGemma prompt format
    # The translate() method uses the proper prompt format:
    # - Professional translator role definition
    # - Source and target language specifications
    # - Instruction to produce only translation (no explanations)
    # - Two blank lines before text (as required by TranslateGemma)
    print(f"Translating from Urdu (ur-PK) to {target_language}...")
    print()
    
    try:
        # Use the translate method which handles the proper TranslateGemma prompt format
        translated_text = client.translate(
            text=urdu_text,
            source_language="ur-PK",  # Pakistani Urdu (source)
            target_language=target_language,  # Target language (user-specified)
            model="translategemma:4b"  # Explicitly use TranslateGemma model
        )
        
        return translated_text.strip()
        
    except Exception as e:
        print(f"✗ Error during translation: {e}")
        raise


def main():
    """
    Main function to run translation from command line.
    Can accept text as argument or read from speech-to-text output.
    """
    print("=" * 70)
    print("🌐 Urdu to Target Language Translation")
    print("=" * 70)
    print()
    
    # Step 1: Get Urdu text input
    # Option 1: From command line argument
    if len(sys.argv) > 1:
        urdu_text = sys.argv[1]
        print(f"Input Urdu text: {urdu_text}")
        print()
    else:
        # Option 2: Prompt user for input
        print("Enter Urdu text to translate:")
        print("(Or run: python3 translate_text.py 'your urdu text' 'target language')")
        print()
        urdu_text = input("Urdu text: ").strip()
        
        if not urdu_text:
            print("✗ No text provided")
            return
    
    # Step 2: Get target language
    if len(sys.argv) > 2:
        target_language = sys.argv[2]
    else:
        target_language = input("Target language (default: English): ").strip() or "English"
    
    print()
    print("=" * 70)
    
    # Step 3: Perform translation
    try:
        translated_text = translate_urdu_to_target(urdu_text, target_language)
        
        # Step 4: Display results
        print("✅ Translation Complete!")
        print("=" * 70)
        print()
        print(f"Source (Urdu): {urdu_text}")
        print()
        print(f"Target ({target_language}): {translated_text}")
        print()
        print("=" * 70)
        
        return translated_text
        
    except Exception as e:
        print(f"✗ Translation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

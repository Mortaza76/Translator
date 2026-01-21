"""
Text cleaning utilities for improving transcription quality.
Removes garbled text, special tokens, and normalizes Urdu text.
"""

import re
from typing import Optional


def clean_whisper_transcription(text: str) -> str:
    """
    Clean Whisper transcription output to remove common artifacts.
    
    Removes:
    - Special tokens like <|he|>, <|en|>, etc.
    - Mixed scripts (Cyrillic, random Latin)
    - Excessive whitespace
    - Common Whisper artifacts
    
    Args:
        text: Raw transcription text from Whisper
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove Whisper special tokens (e.g., <|he|>, <|en|>, <|transcribe|>)
    text = re.sub(r'<\|[^|]+\|>', '', text)
    
    # Remove standalone special tokens
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove Cyrillic characters (common in garbled transcriptions)
    text = re.sub(r'[а-яА-ЯёЁ]', '', text)
    
    # Remove Japanese characters (katakana/hiragana) - common in garbled transcriptions
    text = re.sub(r'[ぁ-ゟァ-ヿ]', '', text)  # Hiragana and Katakana
    text = re.sub(r'[一-龯]', '', text)  # Kanji (if any)
    
    # Remove random single Latin letters that are likely errors
    # Keep words that are 2+ characters or common single letters (a, i, o)
    words = text.split()
    cleaned_words = []
    for word in words:
        # If it's a single letter and not a common one, skip it
        if len(word) == 1 and word.lower() not in ['a', 'i', 'o', 'ا', 'و', 'ی']:
            continue
        # If word has mixed scripts suspiciously, try to clean it
        if re.search(r'[a-zA-Z]', word) and re.search(r'[ا-ی]', word):
            # Keep only Urdu characters if it's mostly Urdu
            urdu_chars = len(re.findall(r'[ا-ی]', word))
            latin_chars = len(re.findall(r'[a-zA-Z]', word))
            if urdu_chars > latin_chars:
                word = re.sub(r'[a-zA-Z]', '', word)
            elif latin_chars > urdu_chars:
                word = re.sub(r'[ا-ی]', '', word)
        cleaned_words.append(word)
    
    text = ' '.join(cleaned_words)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove very short transcriptions that are likely noise
    if len(text) < 3:
        return ""
    
    return text


def normalize_urdu_text(text: str) -> str:
    """
    Normalize Urdu text for better translation.
    
    Args:
        text: Urdu text to normalize
    
    Returns:
        Normalized Urdu text
    """
    if not text:
        return ""
    
    # Normalize Urdu punctuation
    text = text.replace('۔', '.')  # Urdu full stop
    text = text.replace('،', ',')  # Urdu comma
    
    # Normalize whitespace around punctuation
    text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'[.!?]{2,}', '.', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def clean_for_translation(text: str) -> str:
    """
    Clean text specifically for translation (combines cleaning + normalization).
    
    Args:
        text: Raw transcription text
    
    Returns:
        Cleaned text ready for translation
    """
    # First clean Whisper artifacts
    cleaned = clean_whisper_transcription(text)
    
    # Then normalize for translation
    cleaned = normalize_urdu_text(cleaned)
    
    return cleaned


def is_valid_transcription(text: str, min_length: int = 3) -> bool:
    """
    Check if transcription is likely valid (not just noise).
    
    Args:
        text: Transcription text
        min_length: Minimum length to consider valid
    
    Returns:
        True if transcription appears valid
    """
    if not text or len(text.strip()) < min_length:
        return False
    
    # Check if it has at least some Urdu characters
    urdu_chars = len(re.findall(r'[ا-ی]', text))
    if urdu_chars == 0:
        # Might be English or empty
        return len(text.strip()) >= min_length
    
    # If it has Urdu characters, it's likely valid
    return True

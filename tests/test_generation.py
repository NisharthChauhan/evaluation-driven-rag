# pyrefly: ignore [missing-import]
import pytest
from unittest.mock import patch, MagicMock
from src.generator import Generator

@patch("os.environ.get")
@patch("src.generator.OpenAI")
def test_generator(mock_openai, mock_env_get):
    mock_env_get.return_value = "dummy-key"
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "FastAPI is a fast web framework."
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    generator = Generator(model_name="dummy-model")
    
    context = [
        {"content": "FastAPI is a modern, fast (high-performance) web framework...", "metadata": {"source": "fastapi_intro.md"}},
        {"content": "It is based on standard Python type hints.", "metadata": {"source": "fastapi_intro.md"}},
    ]
    
    result = generator.generate_answer("What is FastAPI?", context)
    
    assert result["answer"] == "FastAPI is a fast web framework."
    assert "fastapi_intro.md" in result["sources"]
    assert len(result["sources"]) == 1 # Deduplicated


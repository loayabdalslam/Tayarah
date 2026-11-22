from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field

# ----------------------------------------------------
# Ollama Compatible Schemas
# ----------------------------------------------------

class OllamaChatMessage(BaseModel):
    role: str = "user"
    content: str = ""
    images: Optional[List[str]] = None

class OllamaOptions(BaseModel):
    num_predict: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stop: Optional[Union[str, List[str]]] = None

class OllamaGenerateRequest(BaseModel):
    model: str = "tayarah"
    prompt: str = ""
    stream: Optional[bool] = True
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    options: Optional[Dict[str, Any]] = None

class OllamaChatRequest(BaseModel):
    model: str = "tayarah"
    messages: List[OllamaChatMessage] = []
    stream: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None

class OllamaModelInfo(BaseModel):
    name: str
    model: str
    modified_at: str
    size: int
    digest: str
    details: Dict[str, Any]

# ----------------------------------------------------
# OpenAI / vLLM Compatible Schemas
# ----------------------------------------------------

class OpenAIChatMessage(BaseModel):
    role: str = "user"
    content: str = ""
    name: Optional[str] = None

class OpenAIChatCompletionRequest(BaseModel):
    model: str = "tayarah"
    messages: List[OpenAIChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 256
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0

class OpenAICompletionRequest(BaseModel):
    model: str = "tayarah"
    prompt: Union[str, List[str]] = ""
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 256

class OpenAIModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "tayarah"
    root: Optional[str] = None

class OpenAIModelList(BaseModel):
    object: str = "list"
    data: List[OpenAIModelItem]

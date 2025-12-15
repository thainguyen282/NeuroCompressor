import os
import logging
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)

class LLMModel:
    def __init__(self, **kwargs):
       model = AutoModelForCausalLM.from_pretrained(kwargs.get("model_name"))
######################Imports and set up#####################
from dotenv import load_dotenv
import os
import asyncio
import html
from typing import Literal, Optional

import requests
from openai import AsyncAzureOpenAI
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, set_tracing_disabled
from pypdf import PdfReader
import gradio as gr

load_dotenv(override=True)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = "gpt-5.4-ptu"

print("Endpoint loaded:", bool(AZURE_OPENAI_ENDPOINT))
print("API key loaded:", bool(AZURE_OPENAI_API_KEY))

##################Configure OpenAI SDK#######################
set_tracing_disabled(True)

azure_client = AsyncAzureOpenAI(
    azure_endpoint= AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-10-21",
)

azure_model = OpenAIChatCompletionsModel(
    model=DEPLOYMENT,
    openai_client=azure_client,
)

print("Azure model configured for the OpenAI Agents SDK.") 

###########################Persona###########################
persona = (
    "UC New Bearcat Helper, a helpful, caring, and professional counselor that helps "
    "answer questions and inquiries from new students and parents. Your main goal is to "
    "provide information about the University of Cincinnati and guiding them the right resources. "
    "You will help students and parents understand more about UC (tour information, admission requirements, "
    "tuition, how to apply, etc.). Record students information when they show interest. Encourage them to check "
    "program websites."
)
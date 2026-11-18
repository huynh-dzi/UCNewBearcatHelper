import os
import re
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, set_tracing_disabled
from financial_aid_data import FINANCIAL_AID_FACTS, ENROLLMENT_CONTACT

load_dotenv(override=True)
set_tracing_disabled(True)

DEPLOYMENT = "gpt-5.4-ptu"

azure_model = OpenAIChatCompletionsModel(
    model=DEPLOYMENT,
    openai_client=AsyncAzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-10-21",
    ),
)

STOPWORDS = {"a", "an", "the", "my", "i", "to", "of", "for", "do", "does", "is", "what", "where",
             "who", "how", "about", "on", "in", "and", "or", "me", "need", "want", "with", "can", "you"}

def _words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}

def find_financial_aid_info(question_topic: str) -> str:
    """Look up financial aid, scholarship, tuition, billing, or transcript information for a student
    or parent. Pass the key topic, for example 'FAFSA', 'scholarship', 'tuition cost', 'bill payment',
    or 'transcript'. Returns the matching answer and its official source link."""
    query = _words(question_topic)
    matches = []
    for fact in FINANCIAL_AID_FACTS:
        score = len(query & _words(fact["topic"]))
        if score > 0:
            matches.append((score, fact))
    if not matches:
        return ("No matching financial aid fact found. Refer the person to Enrollment Services: "
                f"{ENROLLMENT_CONTACT['phone']}, {ENROLLMENT_CONTACT['email']}, {ENROLLMENT_CONTACT['website']}")
    matches.sort(key=lambda m: m[0], reverse=True)
    fact = matches[0][1]
    return (f"Answer: {fact['answer']}\n"
            f"Source: {fact['source']}\n"
            f"Last checked: {fact['last_checked']}")

find_financial_aid_tool = function_tool(find_financial_aid_info)

financial_aid_agent = Agent(
    name="Financial Aid Agent",
    instructions=(
        "You help new students and parents with financial aid, scholarships, tuition, billing, "
        "and transcript questions. Always call find_financial_aid_info before answering. "
        "Give only facts the tool returns, and include the source link. Never state a dollar amount, "
        "deadline, or percentage unless the tool returned it. If the tool finds nothing, give the "
        "Enrollment Services contact details."
    ),
    model=azure_model,
    tools=[find_financial_aid_tool],
)

def run_offline_tests():
    print("=== OFFLINE TESTS (no API calls) ===")
    tests = [
        "how do I apply for financial aid",
        "do I qualify for scholarships",
        "how much is tuition per credit",
        "pay my bill",
        "I need my transcript",
        "pizza delivery",
    ]
    for q in tests:
        print(f"{q!r} -> {find_financial_aid_info(q).splitlines()[0][:100]}")
    print()

async def run_live_tests():
    print("=== LIVE AGENT TESTS ===")
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("No AZURE_OPENAI_API_KEY found in .env. Skipping live tests.")
        return
    questions = [
        "How do I apply for financial aid?",
        "Do I qualify for scholarships?",
        "How much is tuition per credit?",
        "I need my transcript.",
    ]
    for q in questions:
        result = await Runner.run(financial_aid_agent, q)
        print("Q:", q)
        print("A:", result.final_output)
        print()

if __name__ == "__main__":
    run_offline_tests()
    asyncio.run(run_live_tests())

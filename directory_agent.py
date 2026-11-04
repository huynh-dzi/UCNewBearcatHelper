import os
import re
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, set_tracing_disabled

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

UC_DIRECTORY = [
    {
        "department": "Admissions Office",
        "keywords": "admission admissions application applications apply first-year freshman transfer "
                    "requirements materials campus visit visits tour tours open house",
        "phone": "(513) 556-1100",
        "email": "admissions@uc.edu",
        "website": "https://www.admissions.uc.edu/",
    },
    {
        "department": "Enrollment Services",
        "keywords": "financial aid scholarship scholarships registrar records transcript transcripts "
                    "student status billing bill tuition cost credit fees fee bursar",
        "phone": "(513) 556-1000",
        "email": "enrollmentservices@uc.edu",
        "website": "https://www.onestop.uc.edu/",
    },
    {
        "department": "Housing",
        "keywords": "housing dorm dorms residence hall halls meal meals plan plans food deadline deadlines",
        "phone": "(513) 556-9800",
        "email": "uchousing@uc.edu",
        "website": "https://www.uc.edu/campus-life/housing.html",
    },
    {
        "department": "IT Help Desk",
        "keywords": "it email account accounts password reset ticket tickets technical tech login uc",
        "phone": "(513) 556-4357",
        "email": None,
        "website": "https://www.uc.edu/about/ucit/help.html (portal: uc.edu/ithelp)",
    },
    {
        "department": "Parking Services",
        "keywords": "parking park lot lots pass passes permit permits",
        "phone": "(513) 556-2283",
        "email": "parking@uc.edu",
        "website": "https://www.uc.edu/about/parking.html",
    },
]

STOPWORDS = {"a", "an", "the", "my", "i", "to", "of", "for", "do", "does", "is", "what", "where",
             "who", "how", "about", "on", "in", "and", "or", "me", "need", "want", "with", "can", "you"}

def _words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}

def find_department(question_topic: str) -> str:
    """Find the UC office that handles a student's or parent's question. Pass the key topic,
    for example 'parking pass', 'tuition cost', 'dorm meal plan', or 'password reset'.
    Returns the department name, phone, email, and website."""
    query = _words(question_topic)
    best, best_score = None, 0
    for entry in UC_DIRECTORY:
        score = len(query & _words(entry["keywords"] + " " + entry["department"]))
        if score > best_score:
            best, best_score = entry, score
    if best is None:
        return "No matching UC department found. Direct the person to https://www.uc.edu/"
    email = best["email"] or "No email; submit requests through the online portal"
    return (f"Department: {best['department']}\n"
            f"Phone: {best['phone']}\n"
            f"Email: {email}\n"
            f"Website: {best['website']}")

lookup_uc_department = function_tool(find_department)

directory_agent = Agent(
    name="Directory Agent",
    instructions=(
        "You help new students and parents find the right UC office. "
        "For any question about who to contact, call lookup_uc_department first. "
        "Reply with the department name, phone, email, and website it returns. "
        "If no department matches, say so and point to https://www.uc.edu/. "
        "Never invent phone numbers, emails, or URLs."
    ),
    model=azure_model,
    tools=[lookup_uc_department],
)

def run_offline_tests():
    print("=== OFFLINE TESTS (no API calls) ===")
    tests = [
        "parking pass",
        "tuition cost per credit",
        "reset my password",
        "transfer application",
        "dorm meal plan",
        "pizza delivery",
    ]
    for q in tests:
        print(f"{q!r} -> {find_department(q).splitlines()[0]}")
    print()

async def run_live_tests():
    print("=== LIVE AGENT TESTS ===")
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        print("No AZURE_OPENAI_API_KEY found in .env. Skipping live tests.")
        return
    questions = [
        "Who do I contact about parking passes?",
        "Who handles tuition and cost per credit?",
        "I forgot my UC password, what do I do?",
    ]
    for q in questions:
        result = await Runner.run(directory_agent, q)
        print("Q:", q)
        print("A:", result.final_output)
        print()

if __name__ == "__main__":
    run_offline_tests()
    asyncio.run(run_live_tests())

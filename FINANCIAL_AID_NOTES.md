# Financial Aid Agent (Enrollment Services)

Covers financial aid, scholarships, tuition, billing, and transcripts.

## Files
- `financial_aid_data.py`: the facts the agent can use. Each answer links to an official UC page.
- `financial_aid_agent.py`: the lookup tool, the agent, and the test cases.

## Run it
1. Put both files in the same folder, with a `.env` file that has `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`.
2. Run `python financial_aid_agent.py`.

## Verification status
- Source links point to the Enrollment Services page listed in the UC directory.
- Facts marked "TODO" in `last_checked` have not been checked against the official page yet.
- Rule: the agent states no dollar amounts, deadlines, or percentages unless the tool returned them.

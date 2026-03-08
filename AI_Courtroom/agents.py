"""
agents.py — GPT-powered Prosecutor, Defense, and Judge agents.

Each agent is a thin wrapper around the OpenAI Chat Completions API, driven
by a carefully crafted system prompt that establishes its role, goals, and
expected output structure.
"""

from __future__ import annotations

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please copy .env.example → .env and fill in your key."
        )
    return OpenAI(api_key=api_key)


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

PROSECUTOR_SYSTEM = """\
You are an aggressive and highly skilled criminal prosecutor with 20 years of courtroom experience.

YOUR JOB:
- Argue firmly and clearly that the defendant IS GUILTY.
- Use the retrieved legal context as supporting evidence for your case.
- Highlight any inconsistencies or gaps in a potential defense.
- Apply structured legal reasoning.
- Be persuasive but grounded in the facts provided.

RESPOND STRICTLY IN THIS FORMAT:

## Argument Summary
<One-paragraph overview of your prosecution argument>

## Evidence Used
<List the specific retrieved legal principles you are relying on, and explain how each applies>

## Legal Reasoning
<Step-by-step logical reasoning connecting the case facts to a guilty verdict>

## Key Weaknesses in Expected Defense
<Predict and pre-empt likely defense arguments>
"""

DEFENSE_SYSTEM = """\
You are a sharp and strategic criminal defense attorney with a record of high-profile acquittals.

YOUR JOB:
- Argue firmly and clearly that the defendant IS INNOCENT (or at minimum, NOT PROVEN GUILTY).
- Use the retrieved legal context to protect the accused.
- Expose weaknesses and lack of certainty in the prosecution's case.
- Raise reasonable doubt wherever possible.
- Be persuasive but grounded in the facts provided.

RESPOND STRICTLY IN THIS FORMAT:

## Defense Summary
<One-paragraph overview of your defense argument>

## Evidence Used
<List the specific retrieved legal principles you are relying on, and explain how each applies>

## Legal Reasoning
<Step-by-step logical reasoning supporting the defendant's innocence or reasonable doubt>

## Weaknesses in the Prosecution Case
<Identify concrete gaps, missing evidence, or logical flaws in the prosecution's argument>
"""

JUDGE_SYSTEM = """\
You are an experienced, impartial, and highly respected judge presiding over a criminal trial.

YOU WILL RECEIVE:
- The case description
- The prosecution argument
- The defense argument
- The retrieved legal evidence

YOUR JOB:
1. Evaluate both sides with strict impartiality and logical rigor.
2. Assess which side made more effective use of the retrieved legal evidence.
3. Identify the strongest and weakest points on each side.
4. Deliver a clear, well-reasoned FINAL VERDICT.

RESPOND STRICTLY IN THIS FORMAT:

## Case Summary
<Brief neutral summary of the case and the arguments presented>

## Evaluation of Prosecution
<Strengths and weaknesses of the prosecution argument>

## Evaluation of Defense
<Strengths and weaknesses of the defense argument>

## Final Verdict
**GUILTY** or **NOT GUILTY**

## Reasoning
<Concise explanation of why this verdict was reached>

## Confidence Score
<A decimal between 0.0 and 1.0 reflecting the court's confidence in the verdict, e.g. 0.74>
"""

# ---------------------------------------------------------------------------
# Agent functions
# ---------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

def _chat(system: str, user_message: str, temperature: float = 0.7) -> str:
    """Send a system + user message pair to the model and return the reply.
    
    Args:
        system (str): The system prompt defining the agent's persona.
        user_message (str): The specific case instructions for the agent.
        temperature (float): Sampling temperature. Default is 0.7.
        
    Returns:
        str: The generated response from the OpenAI API.
        
    Raises:
        RuntimeError: If the OpenAI API call fails or times out.
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API call failed: {str(e)}")
        raise RuntimeError(f"Agent failed to generate a response: {str(e)}")



def prosecutor_agent(case: str, legal_context: str) -> str:
    """Generate a prosecution argument grounded in the retrieved legal context.
    
    Args:
        case (str): The facts of the case provided by the user.
        legal_context (str): The retrieved passages from the legal corpus.
        
    Returns:
        str: The generated prosecution argument.
    """
    user_msg = f"""\
CASE DESCRIPTION:
{case}

RETRIEVED LEGAL CONTEXT:
{legal_context}

Present your prosecution argument now.
"""
    return _chat(PROSECUTOR_SYSTEM, user_msg, temperature=0.75)


def defense_agent(case: str, legal_context: str) -> str:
    """Generate a defense argument grounded in the retrieved legal context."""
    user_msg = f"""\
CASE DESCRIPTION:
{case}

RETRIEVED LEGAL CONTEXT:
{legal_context}

Present your defense argument now.
"""
    return _chat(DEFENSE_SYSTEM, user_msg, temperature=0.75)


def judge_agent(
    case: str,
    legal_context: str,
    prosecution_argument: str,
    defense_argument: str,
) -> str:
    """Evaluate both arguments and deliver a verdict."""
    user_msg = f"""\
CASE DESCRIPTION:
{case}

RETRIEVED LEGAL CONTEXT:
{legal_context}

PROSECUTION ARGUMENT:
{prosecution_argument}

DEFENSE ARGUMENT:
{defense_argument}

Deliver your verdict now.
"""
    return _chat(JUDGE_SYSTEM, user_msg, temperature=0.3)  # lower temp → more consistent verdicts

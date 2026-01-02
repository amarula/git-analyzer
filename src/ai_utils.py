"""OpenAI Commit Message Summarizer Module

This module provides a function to summarize an author's software development
contributions based on their Git commit messages using the OpenAI API.

"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def summarize_commit_messages(api_key: str, commit_messages_string: str,
                              n_months: int, author_name: str, ai_model: str) -> str:
    """Summarizes a string of commit messages using OpenAI's API.

    Args:
        api_key (str): Your OpenAI API key.
        commit_messages_string (str): A string containing all commit messages from an author.
        n_months (int): The period in months for which the commits were made.
        author_name (str): The author of commit message.

    Returns:
        str: A summary of the author's contributions based on the commit messages.

    """
    openai_model = ChatOpenAI(model=ai_model, temperature=0.2, api_key=api_key)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a Technical Lead creating executive summaries of engineering contributions. "
                    "Your goal is to produce a strict, factual, and concise report of the work done.\n\n"
                    "### STYLE GUIDELINES:\n"
                    "1. **Tone:** Clinical and technical. No fluff, no adjectives describing effort (e.g., remove words like 'meticulous', 'significant', 'comprehensive', 'showcasing').\n"
                    "2. **Structure:** Use active verbs to start sentences (e.g., 'Updated', 'Fixed', 'Refactored').\n"
                    f"3. **Identity:** Refer to the author ONLY as '{author_name}'. Do not use ANY pronouns (he, she, they, their, his). If a reference is needed, repeat the name or rephrase the sentence to be passive.\n"
                    "4. **Content:** Focus strictly on the *what* and *why* (technical changes and business value). Do not describe the *how* (e.g., do not mention 'instructions were included' unless it is a documentation task).\n"
                )
            ),
            HumanMessage(
                content=(
                    f"Author: {author_name}\n"
                    f"Timeframe: {n_months} months\n"
                    "Task: Summarize the following commit messages into a concise technical paragraph (max 4 sentences).\n\n"
                    f"Commit Messages:\n---\n{commit_messages_string}\n---"
                )
            ),
        ],
    )

    # Create a chain from the prompt and the model
    chain = prompt | openai_model

    # Invoke the chain with the formatted input
    try:
        response = chain.invoke(
            {
                "author_name": author_name,
                "n_months": n_months,
                "commit_messages_string": commit_messages_string,
            },
        )
        return response.content
    except Exception as e:
        return f"An unexpected error occurred: {e}"

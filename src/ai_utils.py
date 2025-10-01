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
                content="You are a helpful assistant that summarizes software development contributions.",
            ),
            HumanMessage(
                content=(
                    f"Summarize the following software development contributions from an author named {author_name},"
                    f" over a period of {n_months} months, based on commit messages.\n"
                    f"- Don't change the author name that must be {author_name}.\n"
                    f"- Do not use gendered pronouns (like he, she, or they); always refer to the author with {author_name};\n"
                    "- Focus on key features, bug fixes, improvements, and overall progress.\n"
                    "- Some entries may be changelogs, not commits. If it looks like a changelog,"
                    " summarize the overall changes, not individual items.\n"
                    "- Provide a concise yet comprehensive overview. Maximum 8 sentences. \n\n"
                    "- Important: Use no pronouns. Do not use “they”\n"
                    f"Commit Messages:\n---\n{commit_messages_string}\n---"
                ),
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

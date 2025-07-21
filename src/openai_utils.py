"""
OpenAI Commit Message Summarizer Module

This module provides a function to summarize an author's software development
contributions based on their Git commit messages using the OpenAI API.

"""

import openai

def summarize_commit_messages(api_key: str, commit_messages_string: str,
                              n_months: int, author_name: str, openai_model: str) -> str:
    """
    Summarizes a string of commit messages using OpenAI's API.

    Args:
        api_key (str): Your OpenAI API key.
        commit_messages_string (str): A string containing all commit messages from an author.
        n_months (int): The period in months for which the commits were made.
        author_name (str): The author of commit message.

    Returns:
        str: A summary of the author's contributions based on the commit messages.
    """
    openai.api_key = api_key

    # Craft a clear and concise prompt for OpenAI
    prompt = (
        f"Summarize the following software development contributions from an author named {author_name},"
        f"over a period of {n_months} months, based on commit messages.\n"
        f"- Don\'t change the author name that must be {author_name}."
        f"- Do not use gendered pronouns (like he, she, or they) always refer to the author with {author_name};\n"
        "- Focus on key features, bug fixes, improvements, and overall progress.\n"
        "- Some entries may be changelogs, not commits. If it looks like a changelog,"
        " summarize the overall changes, not individual items.\n"
        "- Provide a concise yet comprehensive overview. Maximum 8 sentences. \n\n"
        "- Important: Use no pronouns. Do not use “they”"
        f"Commit Messages:\n---\n{commit_messages_string}\n---"
    )

    try:
        response = openai.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes software development contributions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,  # Adjust as needed for desired summary length
            temperature=0.7  # Controls randomness. Lower values for more focused summaries.
        )
        return response.choices[0].message.content.strip()
    except openai.APIError as e:
        return f"An OpenAI API error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"


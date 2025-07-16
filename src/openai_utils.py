"""
OpenAI Commit Message Summarizer Module

This module provides a function to summarize an author's software development
contributions based on their Git commit messages using the OpenAI API.

"""

import openai

def summarize_commit_messages(commit_messages_string: str, api_key: str, n_months: int) -> str:
    """
    Summarizes a string of commit messages using OpenAI's API.

    Args:
        commit_messages_string (str): A string containing all commit messages from an author.
        api_key (str): Your OpenAI API key.
        n_months (int): The period in months for which the commits were made.

    Returns:
        str: A summary of the author's contributions based on the commit messages.
    """
    openai.api_key = api_key

    # Craft a clear and concise prompt for OpenAI
    prompt = (
        f"Summarize the following software development contributions from an author "
        f"over a period of {n_months} months based on their commit messages. "
        f"Focus on key features, bug fixes, improvements, and overall progress. "
        f"Provide a concise yet comprehensive overview.\n\n"
        f"Commit Messages:\n---\n{commit_messages_string}\n---"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" for potentially better results
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


"""
Git Utilities

This module provides essential functions for Git repository management. It allows
users to extract metadata and history from Git repositories, as well as perform
fundamental operations like cloning new repositories and pulling updates to
existing ones.

"""

import os
import shutil
import datetime
from datetime import datetime, timedelta
from git import Repo, InvalidGitRepositoryError, NoSuchPathError, GitCommandError


def generate_commit_hyperlink(repo_path, base_web_url, commit_hash_prefix):
    """
    Generates an hyperlink to a Git commit on a web platform.

    Args:
        repo_path (str): The path to the local Git repository.
        base_web_url (str): The base URL for the repository on the web
                            (e.g., "https://github.com/your_username/your_repo").
        commit_hash_prefix (str): A full or partial commit hash.

    Returns:
        str: The hyperlink string, or None if the commit is not found.
    """
    try:
        repo = Repo(repo_path)
        commit = repo.commit(commit_hash_prefix)

        commit_full_hash = commit.hexsha
        commit_message = commit.summary
        special_cases_prefixes = [
            "https://git.kernel.org",
            "https://git.openembedded.org"
        ]

        commit_url = None
        for prefix in special_cases_prefixes:
            if base_web_url.startswith(prefix):
                commit_url = f"{base_web_url}/commit/?id={commit_full_hash}"
                break

        if commit_url is None:
            if base_web_url.endswith(".git"):
                base_web_url = base_web_url[:-4]
            commit_url = f"{base_web_url}/commit/{commit_full_hash}"

        hyperlink = f'{commit_url}'
        return hyperlink

    except Exception as e:
        print(f"Error generating hyperlink: {e}")
        return None


def git_pull_or_clone(remote_url=None, repo_path="."):
    """
    Checks if a directory is a Git repository.
    If it is, performs a 'git pull'.
    If 'git pull' fails, or if the directory is not a valid Git repository initially,
    and a remote_url is provided, it attempts to clone (or re-clone) the repository
    into the specified path.

    Args:
        remote_url (str, optional): The URL of the remote Git repository to clone.
                                    Required if the directory is not a Git repo
                                    or if a pull fails and a re-clone is desired.
        repo_path (str): The path to the directory to check or clone into.
                         Defaults to the current directory.
    Returns:
        repo: Return the repository if the clone is successfull otherwise None.
    """
    # Normalize the path to ensure consistency
    abs_repo_path = os.path.abspath(repo_path)

    # Helper function to perform cloning
    def _perform_clone(url, path):
        print(f"Attempting to clone repository from '{url}' into '{path}'...")
        try:
            # Ensure the parent directory exists before cloning
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir)

            repo = Repo.clone_from(url, path)
            print(f"Repository successfully cloned into '{path}'.")
            return repo
        except GitCommandError as e:
            print(f"Error during 'git clone': {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during cloning: {e}")
            return None

    try:
        # Attempt to open the directory as an existing Git repository
        repo = Repo(abs_repo_path)

        print(f"'{abs_repo_path}' appears to be an existing Git repository.")
        print("Attempting to perform 'git pull' using GitPython...")

        try:
            # Perform git pull from the 'origin' remote
            repo.remotes.origin.pull()

            print("Git pull successful:")
            return Repo(abs_repo_path)
        except GitCommandError as e:
            print(f"Error during 'git pull': {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")

            if remote_url:
                print(f"Git pull failed. Attempting to remove '{abs_repo_path}' and re-clone...")
                # Remove the existing directory
                if os.path.exists(abs_repo_path):
                    try:
                        shutil.rmtree(abs_repo_path)
                        print(f"Removed existing directory '{abs_repo_path}'.")
                    except OSError as remove_e:
                        print(f"Error removing directory '{abs_repo_path}': {remove_e}")
                        return None # Cannot proceed with re-clone if removal fails

                # Now attempt to re-clone
                return _perform_clone(remote_url, abs_repo_path)

            print("Git pull failed and no remote URL provided for re-cloning.")
            return None # Operation was attempted, but failed without re-clone option

    except (InvalidGitRepositoryError, NoSuchPathError):
        # If it's not a valid Git repository or the path doesn't exist, try to clone
        print(f"'{abs_repo_path}' is not a valid Git repository or does not exist.")
        if remote_url:
            return _perform_clone(remote_url, abs_repo_path)

        print("No remote URL provided to clone the repository.")
        return None # No operation attempted

    except Exception as e: # Catch any other unexpected errors at the top level
        print(f"An unexpected error occurred: {e}")
        return None # An operation was attempted, even if it failed


def analyze_real_git_commits(
    repo_urls: list[str], company_identifier: str, months_back: int, deploy_dir_name: str
) -> dict:
    """
    Clones Git repositories, finds commits by a specified company within a timeframe,
    and returns structured commit data.

    Args:
        repo_urls: A list of Git repository URLs.
        company_identifier: A string to identify company commits (e.g., email domain or part
                            of the committer name).
        months_back: An integer representing the number of months to look back for commits.
        deploy_dir_name: The name of the directory where repositories will be cloned.

    Returns:
        A dictionary containing the structured commit data or an error message.
    """
    all_repo_commits_structured = []

    project_root = os.getcwd()
    deploy_target_dir = os.path.join(project_root, deploy_dir_name)

    try:
        # Ensure the deploy directory exists
        os.makedirs(deploy_target_dir, exist_ok=True)

        # Calculate the 'since' date for commit filtering
        since_date = datetime.now() - timedelta(days=months_back * 30)

        for repo_url in repo_urls:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_path = os.path.join(deploy_target_dir, repo_name)

            print(f"Cloning {repo_url} directly into {repo_path}...")
            try:
                repo = git_pull_or_clone(repo_url, repo_path)
                print(f"Successfully cloned {repo_name}.")

            except GitCommandError as e:
                error_msg = str(e)
                print(f"Error cloning repository {repo_url}: {error_msg}")
                all_repo_commits_structured.append(
                    {
                        "repo_name": repo_name,
                        "repo_url": repo_url,
                        "repo_path": repo_path,
                        "error": f"Failed to clone: {error_msg}",
                        "commits": [],  # No commits if clone failed
                    }
                )
                continue  # Skip to next repository
            except Exception as e:
                print(
                    f"An unexpected error occurred during cloning {repo_url}: {str(e)}"
                )
                all_repo_commits_structured.append(
                    {
                        "repo_name": repo_name,
                        "repo_url": repo_url,
                        "repo_path": repo_path,
                        "error": f"An unexpected error occurred during cloning: {str(e)}",
                        "commits": [],
                    }
                )
                continue

            print(f"Analyzing commits for {repo_name} since "
                  f"{since_date.strftime('%Y-%m-%d %H:%M:%S')}...")
            repo_commits_list = []
            try:
                # Iterate through commits
                # We filter by `after` date to get commits since `since_date`
                # and exclude merge commits by checking if the commit has more than one parent.
                # GitPython's log method can also take `after` and `no_merges` arguments directly.
                for commit in repo.iter_commits(since=since_date, no_merges=True):
                    author_name = commit.author.name
                    author_email = commit.author.email
                    commit_date = datetime.fromtimestamp(commit.authored_date).isoformat()
                    commit_message = commit.message.strip()
                    sha1_hash = commit.hexsha

                    # Filter by company identifier (case-insensitive)
                    if (
                        company_identifier.lower() in author_name.lower()
                        or company_identifier.lower() in author_email.lower()
                    ):
                        repo_commits_list.append(
                            {
                                "hash": commit.hexsha,
                                "author_name": author_name,
                                "author_email": author_email,
                                "date": commit_date,
                                "message": commit_message,
                                "sha1": sha1_hash,
                            }
                        )

                all_repo_commits_structured.append(
                    {
                        "repo_name": repo_name,
                        "repo_url": repo_url,
                        "repo_path": repo_path,
                        "commits": repo_commits_list,
                    }
                )

            except GitCommandError as e:
                error_msg = str(e)
                print(f"Error getting git log for {repo_name}: {error_msg}")
                all_repo_commits_structured.append(
                    {
                        "repo_name": repo_name,
                        "repo_url": repo_url,
                        "repo_path": repo_path,
                        "error": f"Failed to get commit log: {error_msg}",
                        "commits": [],
                    }
                )
            except Exception as e:
                print(f"Error processing commits for {repo_name}: {e}")
                all_repo_commits_structured.append(
                    {
                        "repo_name": repo_name,
                        "repo_url": repo_url,
                        "repo_path": repo_path,
                        "error": f"Error processing commits: {str(e)}",
                        "commits": [],
                    }
                )

        return {
            "commit_data": all_repo_commits_structured,
            "message": "Commit analysis complete.",
        }

    except Exception as e:
        print(f"An unexpected error occurred in analyze_real_git_commits: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

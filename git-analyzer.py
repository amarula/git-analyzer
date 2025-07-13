import os
import shutil
import datetime
import configparser
import argparse
from git import Repo, InvalidGitRepositoryError, NoSuchPathError, GitCommandError

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
        company_identifier: A string to identify company commits (e.g., email domain or part of the committer name).
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
        since_date = datetime.datetime.now() - datetime.timedelta(days=months_back * 30)

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
                        "error": f"An unexpected error occurred during cloning: {str(e)}",
                        "commits": [],
                    }
                )
                continue

            print(f"Analyzing commits for {repo_name} since {since_date.strftime('%Y-%m-%d %H:%M:%S')}...")
            repo_commits_list = []
            try:
                # Iterate through commits
                # We filter by `after` date to get commits since `since_date`
                # and exclude merge commits by checking if the commit has more than one parent.
                # GitPython's log method can also take `after` and `no_merges` arguments directly.
                for commit in repo.iter_commits(since=since_date, no_merges=True):
                    author_name = commit.author.name
                    author_email = commit.author.email
                    commit_date = datetime.datetime.fromtimestamp(commit.authored_date).isoformat()
                    commit_message = commit.message.strip()

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
                            }
                        )

                all_repo_commits_structured.append(
                    {
                        "repo_name": repo_name,
                        "repo_url": repo_url,
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

def generate_article_content(commit_data: list[dict], months_back: int) -> str:
    """
    Generates a blog article based on commit data.
    In a real tool, this would call a large language model API (e.g., Gemini).

    Args:
        commit_data: Structured commit data.
        months_back: The number of months the analysis covered.

    Returns:
        A string containing the blog article.
    """
    if not commit_data:
        return "No relevant commits found to generate an article."

    article_content = f"""
# Recent Developments: A Look at Our Codebase ({months_back} Months Review)

We're excited to share a summary of the significant progress made across our repositories in the last {months_back} months. Our dedicated team has been busy pushing new features, refining existing functionalities, and enhancing the overall stability of our products.

Here's a breakdown of key contributions by repository:

"""

    for repo in commit_data:
        article_content += f"## {repo['repo_name']}\n\n"
        article_content += f"Repository URL: {repo['repo_url']}\n\n"

        if "error" in repo:
            article_content += (
                f"**Error processing this repository:** {repo['error']}\n\n"
            )
        elif repo["commits"]:
            article_content += "Our team has made the following notable commits:\n\n"
            for commit in repo["commits"]:
                # Ensure the message is handled, even if it's empty or malformed
                first_line_message = (
                    commit["message"].split("\n")[0]
                    if commit["message"]
                    else "(No message)"
                )
                first_line_message = first_line_message.replace("_", r"\_")
                article_content += f"- **{commit['author_name']}** on {commit['date'].split('T')[0]}: {first_line_message}\n"
            article_content += "\n"
        else:
            article_content += f"No company-specific commits were identified in this repository during the last {months_back} months.\n\n"

    article_content += """
This overview highlights the continuous effort and innovation from our development team. We look forward to bringing even more exciting updates in the future!

---
*Generated by the Git Commit Article Generator*
"""
    return article_content


def load_config_from_ini(file_path: str) -> dict | None:
    """
    Loads configuration from an INI file.

    Args:
        file_path: The path to the INI configuration file.

    Returns:
        A dictionary containing the configuration, or None if the file cannot be read.
    """
    config = configparser.ConfigParser()
    try:
        config.read(file_path)
        git_config = {}
        if "GitConfig" in config:
            repo_urls_str = config["GitConfig"].get("repo_urls", "")
            git_config["repo_urls"] = [
                url.strip() for url in repo_urls_str.split(",") if url.strip()
            ]
            git_config["company_identifier"] = (
                config["GitConfig"].get("company_identifier", "").strip()
            )
            git_config["months_back"] = config["GitConfig"].getint("months_back", None)
            git_config["deploy_dir"] = config["GitConfig"].get("deploy_dir", None)
        return git_config
    except Exception as e:
        print(f"Error reading INI file {file_path}: {e}")
        return None


def main():
    """
    Main function to run the Git commit analysis and article generation tool.
    Supports optional configuration from an INI file and command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a blog article summarizing Git commits from repositories."
    )
    parser.add_argument(
        "-r",
        "--repo-urls",
        help="Comma-separated list of Git repository URLs.",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--company-identifier",
        help='String to identify company commits (e.g., email domain or "My Company Name").',
        type=str,
    )
    parser.add_argument(
        "-m",
        "--months-back",
        help="Number of months back to analyze commits.",
        type=int,
    )
    parser.add_argument(
        "-f",
        "--config-file",
        help="Path to an INI configuration file. If provided and successfully loaded, it will override other core parameters (-r, -c, -m).",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--save-to-file",
        help="Automatically save the generated article to a file (provide filename).",
        type=str,
        nargs="?",  # Allows the argument to be optional, if present without value, it's None
        const="git_report.md",  # Default value if -s is present without an argument
    )
    parser.add_argument(
        "-d",
        "--deploy-dir",
        help="Name of the directory to clone repositories into (default: 'deploy').",
        type=str,
        default="deploy",
    )

    args = parser.parse_args()

    repo_urls = []
    company_identifier = ""
    months_back = None
    save_file_name = None
    deploy_dir = None

    config_loaded_successfully = False

    # Attempt to load from INI file if specified
    if args.config_file:
        config_data = load_config_from_ini(args.config_file)
        if config_data:
            repo_urls = config_data.get("repo_urls", [])
            company_identifier = config_data.get("company_identifier", "")
            months_back = config_data.get("months_back", None)
            deploy_dir = config_data.get("deploy_dir", None)
            config_loaded_successfully = True
            print(f"Configuration loaded from {args.config_file}.")
        else:
            print(
                f"Warning: Failed to load configuration from {args.config_file}. Proceeding with command-line arguments or prompts."
            )

    # If config file was NOT successfully loaded, or not provided, then use CLI args
    if not config_loaded_successfully:
        if args.repo_urls:
            repo_urls = [
                url.strip() for url in args.repo_urls.split(",") if url.strip()
            ]
        if args.company_identifier:
            company_identifier = args.company_identifier.strip()
        if args.months_back is not None:
            months_back = args.months_back
        if args.deploy_dir:
            deploy_dir = args.deploy_dir

    if args.deploy_dir:
        deploy_dir = args.deploy_dir

    if deploy_dir is None:
        deploy_dir = "deploy"

    if args.save_to_file is not None:
        save_file_name = args.save_to_file

    if not repo_urls:
        repo_urls_input = input(
            "Enter Git repository URLs (comma-separated, e.g., https://github.com/org/repo1.git,https://github.com/org/repo2.git): "
        ).strip()
        repo_urls = [url.strip() for url in repo_urls_input.split(",") if url.strip()]

    if not repo_urls:
        print("No repository URLs provided. Exiting.")
        return

    if not company_identifier:
        company_identifier = input(
            "Enter your company identifier (e.g., @mycompany.com or 'My Company Name'): "
        ).strip()

    if not company_identifier:
        print("Company identifier cannot be empty. Exiting.")
        return

    if months_back is None:
        while True:
            try:
                months_back = int(
                    input("Enter number of months back to analyze (e.g., 3): ").strip()
                )
                if months_back <= 0:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter a positive integer for months.")

    print("\nStarting real Git analysis...")
    analysis_result = analyze_real_git_commits(
        repo_urls, company_identifier, months_back, deploy_dir
    )

    if "error" in analysis_result:
        print(f"\nError during Git analysis: {analysis_result['error']}")
        return

    commit_data = analysis_result.get("commit_data", [])

    article = generate_article_content(commit_data, months_back)

    print("\n--- Generated Article ---")
    print(article)
    print("\n--- End of Article ---")

    if save_file_name:
        try:
            with open(save_file_name, "w", encoding="utf-8") as f:
                f.write(article)
            print(f"Article automatically saved to {save_file_name}")
        except Exception as e:
            print(f"Error automatically saving file {save_file_name}: {e}")
    else:
        save_option = (
            input("\nDo you want to save the article to a file? (yes/no): ")
            .lower()
            .strip()
        )
        if save_option == "yes":
            file_name = input("Enter desired filename (e.g., git_report.md): ").strip()
            if not file_name:
                file_name = "git_report.md"
            try:
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(article)
                print(f"Article saved to {file_name}")
            except Exception as e:
                print(f"Error saving file: {e}")


if __name__ == "__main__":
    main()

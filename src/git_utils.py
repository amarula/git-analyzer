"""Git Utilities

This module provides essential functions for Git repository management. It allows
users to extract metadata and history from Git repositories, as well as perform
fundamental operations like cloning new repositories and pulling updates to
existing ones.

"""

import datetime
import os
import re
import shutil
from datetime import datetime, timedelta
from urllib.parse import urlparse

from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Repo


def generate_commit_hyperlink(base_web_url, commit_full_hash):
    """Build a web URL pointing to a specific commit.

    Args:
        base_web_url (str): The base URL for the repository on the web
                            (e.g., "https://github.com/your_username/your_repo").
        commit_full_hash (str): The full commit SHA (hexsha).

    Returns:
        str: The hyperlink string.

    """
    special_cases_prefixes = [
        'https://git.kernel.org',
        'https://git.openembedded.org',
    ]

    for prefix in special_cases_prefixes:
        if base_web_url.startswith(prefix):
            return f"{base_web_url}/commit/?id={commit_full_hash}"

    if base_web_url.endswith('.git'):
        base_web_url = base_web_url[:-4]
    return f"{base_web_url}/commit/{commit_full_hash}"


# ---------------------------------------------------------------------------
# Internal helpers for git_pull_or_clone
# ---------------------------------------------------------------------------

def _clone_repo(url, path, shallow_since=None):
    """Clone *url* into *path*, optionally shallow since *shallow_since*."""
    print(f"Attempting to clone repository from '{url}' into '{path}'...")
    clone_kwargs = {}
    if shallow_since:
        clone_kwargs['multi_options'] = [f"--shallow-since={shallow_since}"]
        print(f"  (shallow clone since {shallow_since})")
    try:
        parent_dir = os.path.dirname(path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        repo = Repo.clone_from(url, path, **clone_kwargs)
        print(f"Repository successfully cloned into '{path}'.")
        return repo
    except GitCommandError as e:
        result = None
        if shallow_since and 'shallow' in str(e).lower():
            print("Shallow clone failed, falling back to full clone...")
            try:
                if os.path.exists(path):
                    shutil.rmtree(path)
                result = Repo.clone_from(url, path)
                print(f"Full clone succeeded into '{path}'.")
            except GitCommandError as e2:
                print(f"Error during fallback 'git clone': {e2}")
                print(f"Stdout: {e2.stdout}")
                print(f"Stderr: {e2.stderr}")
        else:
            print(f"Error during 'git clone': {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
        return result
    except OSError as e:
        print(f"An unexpected error occurred during cloning: {e}")
        return None


def _oldest_commit_date(repo):
    """Return the authored date of the oldest commit in *repo*."""
    try:
        out = repo.git.log('--reverse', '--format=%at', '-1')
        return datetime.fromtimestamp(int(out.strip()))
    except (GitCommandError, ValueError):
        return None


def _remove_dir(path):
    """Best-effort removal of *path*; logs a warning on failure."""
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except OSError as exc:
        print(f"Error removing directory '{path}': {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def git_pull_or_clone(remote_url=None, repo_path='.', shallow_since=None):
    """Checks if a directory is a Git repository.
    If it is, performs a 'git pull'.
    If 'git pull' fails, or if the directory is not a valid Git repository
    initially, and a remote_url is provided, it attempts to clone (or
    re-clone) the repository into the specified path.

    Args:
        remote_url (str, optional): The URL of the remote Git repository.
        repo_path (str): The path to the directory to check or clone into.
                         Defaults to the current directory.
        shallow_since (str, optional): ISO date string (e.g. "2026-04-01").

    Returns:
        repo: Return the repository if the clone is successful otherwise None.

    """
    abs_repo_path = os.path.abspath(repo_path)
    repo = None

    try:
        repo = Repo(abs_repo_path)

        shallow_file = os.path.join(abs_repo_path, '.git', 'shallow')
        if shallow_since and os.path.exists(shallow_file):
            oldest = _oldest_commit_date(repo)
            try:
                needed = datetime.strptime(shallow_since, '%Y-%m-%d')
            except ValueError:
                needed = None

            if oldest and needed and oldest > needed:
                print(
                    f"Shallow repo oldest commit ("
                    f"{oldest.strftime('%Y-%m-%d')}) "
                    f"is newer than needed ({shallow_since}); "
                    f"re-cloning deeper."
                )
                repo.close()
                _remove_dir(abs_repo_path)
                return _clone_repo(remote_url, abs_repo_path, shallow_since)

        print(f"'{abs_repo_path}' appears to be an existing Git repository.")
        print("Attempting to perform 'git pull' using GitPython...")

        try:
            repo.remotes.origin.pull()
            print('Git pull successful.')
            repo.close()
            return Repo(abs_repo_path)
        except GitCommandError as e:
            print(f"Error during 'git pull': {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            repo.close()

            result = None
            if remote_url:
                print(f"Git pull failed. Attempting to remove "
                      f"'{abs_repo_path}' and re-clone...")
                _remove_dir(abs_repo_path)
                result = _clone_repo(remote_url, abs_repo_path, shallow_since)
            else:
                print('Git pull failed and no remote URL provided for '
                      're-cloning.')
            return result

    except (InvalidGitRepositoryError, NoSuchPathError):
        print(f"'{abs_repo_path}' is not a valid Git repository or does not "
              f"exist.")
        if remote_url:
            return _clone_repo(remote_url, abs_repo_path, shallow_since)
        print('No remote URL provided to clone the repository.')
        return None

    except OSError as e:
        print(f"An unexpected error occurred: {e}")
        return None


def _repo_dir_name(repo_url: str) -> str:
    """Derive a unique directory name from a repository URL.

    Uses the last two path segments (org/repo) to avoid collisions when
    different hosts or organisations have repos with the same final name
    (e.g. STMicroelectronics/linux vs torvalds/linux).
    """
    parsed = urlparse(repo_url)
    path = parsed.path.rstrip('/')
    if path.endswith('.git'):
        path = path[:-4]
    segments = [s for s in path.split('/') if s]
    if len(segments) >= 2:
        return '_'.join(segments[-2:])
    return segments[-1] if segments else 'unknown'


# ---------------------------------------------------------------------------
# Internal helpers for analyze_real_git_commits
# ---------------------------------------------------------------------------

def _make_error_entry(repo_name: str, repo_url: str, repo_path: str,
                      error_msg: str) -> dict:
    """Build a structured error entry for one repo."""
    return {
        'repo_name': repo_name,
        'repo_url': repo_url,
        'repo_path': repo_path,
        'error': error_msg,
        'commits': [],
    }


def _commit_matches_company(author_name: str, author_email: str,
                            commit_message: str, company_identifier: str
                            ) -> tuple[bool, str | None]:
    """Return (is_match, resolved_email) for a single commit.

    *resolved_email* is non-None only when the match came from a
    Signed-off-by trailer and a better email could be extracted.
    """
    comp_lower = company_identifier.lower()
    is_direct = (
        comp_lower in author_name.lower()
        or comp_lower in author_email.lower()
    )

    sob_pattern = r'Signed-off-by:.*' + re.escape(comp_lower)
    sob_match = re.search(sob_pattern, commit_message, re.IGNORECASE)
    is_signer = bool(sob_match)

    if not (is_direct or is_signer):
        return False, None

    resolved = None
    if is_signer and not is_direct:
        email_match = re.search(r'<([^>]+)>', sob_match.group(0))
        if email_match:
            resolved = email_match.group(1)

    return True, resolved


def _collect_repo_commits(repo, since_date, until_date,
                          company_identifier: str) -> list[dict]:
    """Iterate commits in *repo* and return those matching the company."""
    commits = []
    iter_kwargs = {'since': since_date, 'no_merges': True}
    if until_date is not None:
        iter_kwargs['before'] = until_date

    for commit in repo.iter_commits(**iter_kwargs):
        author_name = commit.author.name
        author_email = commit.author.email
        commit_message = commit.message.strip()

        is_match, resolved_email = _commit_matches_company(
            author_name, author_email, commit_message, company_identifier,
        )
        if not is_match:
            continue

        if resolved_email:
            author_email = resolved_email

        commits.append({
            'hash': commit.hexsha,
            'author_name': author_name,
            'author_email': author_email,
            'date': datetime.fromtimestamp(commit.authored_date).isoformat(),
            'message': commit_message,
            'sha1': commit.hexsha,
        })

    return commits


def _process_single_repo(repo_url: str, deploy_target_dir: str,
                         company_identifier: str, since_date,
                         until_date) -> dict:
    """Clone *repo_url* and return its structured commit data."""
    repo_name = _repo_dir_name(repo_url)
    repo_path = os.path.join(deploy_target_dir, repo_name)

    print(f"Cloning {repo_url} directly into {repo_path}...")
    shallow_arg = since_date.strftime('%Y-%m-%d') if since_date else None
    repo = None

    try:
        repo = git_pull_or_clone(repo_url, repo_path, shallow_since=shallow_arg)
    except GitCommandError as e:
        return _make_error_entry(repo_name, repo_url, repo_path,
                                 f"Failed to clone: {e}")
    except OSError as e:
        return _make_error_entry(
            repo_name, repo_url, repo_path,
            f"An unexpected error occurred during cloning: {e!s}",
        )

    if repo is None:
        _remove_dir(repo_path)
        return _make_error_entry(
            repo_name, repo_url, repo_path,
            "Failed to clone repository (remote returned an error "
            "or is unreachable)",
        )

    print(f"Successfully cloned {repo_name}.")
    print(f"Analyzing commits for {repo_name} since "
          f"{since_date.strftime('%Y-%m-%d %H:%M:%S')}...")

    try:
        commits = _collect_repo_commits(repo, since_date, until_date,
                                        company_identifier)
        result = {
            'repo_name': repo_name,
            'repo_url': repo_url,
            'repo_path': repo_path,
            'commits': commits,
        }
    except GitCommandError as e:
        result = _make_error_entry(repo_name, repo_url, repo_path,
                                   f"Failed to get commit log: {e}")
    except OSError as e:
        result = _make_error_entry(repo_name, repo_url, repo_path,
                                   f"Error processing commits: {e!s}")
    finally:
        if repo is not None:
            repo.close()

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_real_git_commits(
    repo_urls: list[str],
    company_identifier: str,
    months_back: int,
    deploy_dir_name: str,
    date_range: tuple | None = None,
) -> dict:
    """Clone repositories and return structured commit data.

    Args:
        repo_urls: A list of Git repository URLs.
        company_identifier: A string to identify company commits (e.g., email
                           domain or part of the committer name).
        months_back: Number of months to look back (fallback when
                    *date_range* is None).
        deploy_dir_name: Directory where repositories will be cloned.
        date_range: Optional ``(since_date, until_date)`` tuple that
                   overrides the *months_back* approximation.

    Returns:
        A dictionary containing the structured commit data or an error message.

    """
    all_repo_commits_structured = []

    project_root = os.getcwd()
    deploy_target_dir = os.path.join(project_root, deploy_dir_name)

    try:
        os.makedirs(deploy_target_dir, exist_ok=True)

        if date_range is not None:
            since_date, until_date = date_range
        else:
            since_date = datetime.now() - timedelta(days=months_back * 30)
            until_date = None

        for repo_url in repo_urls:
            entry = _process_single_repo(
                repo_url, deploy_target_dir, company_identifier,
                since_date, until_date,
            )
            all_repo_commits_structured.append(entry)

        return {
            'commit_data': all_repo_commits_structured,
            'message': 'Commit analysis complete.',
        }

    except (GitCommandError, OSError) as e:
        print(f"An unexpected error occurred in "
              f"analyze_real_git_commits: {e}")
        return {'error': f"An unexpected error occurred: {e!s}"}

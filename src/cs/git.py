import json
from pathlib import Path
from typing import Iterable, Optional, Union

from git import Repo


class GitRepo:
    """A simple Git repository wrapper using GitPython."""

    def __init__(
        self,
        local_path: Union[str, Path],
        repo_url: Optional[str] = None,
        branch: str = "master",
    ):
        self.local_path = Path(local_path)
        self.repo_url = repo_url
        self._ensure_repo(repo_url, branch)

    def _ensure_repo(self, repo_url: Optional[str], branch: str) -> None:
        """Ensure the local folder contains a valid Git repository."""
        self.local_path.mkdir(parents=True, exist_ok=True)
        git_dir = self.local_path / ".git"

        if git_dir.exists():
            try:
                self.repo = Repo(self.local_path)
            except Exception as e:
                raise ValueError(
                    f"{self.local_path} exists but is not a git repository."
                ) from e
            return

        if not repo_url:
            raise ValueError("repo_url is required when cloning a new repository.")

        self.repo = Repo.clone_from(repo_url, to_path=self.local_path, branch=branch)

    def pull(self, remote: str = "origin", branch: str = "main") -> str:
        """Pull the latest code from the specified remote branch."""
        return self.repo.git.pull("--progress", remote, branch)

    def push(self, remote: str = "origin", branch: str = "main") -> None:
        """Push the current branch to the specified remote."""
        self.repo.remote(remote).push(branch)

    @property
    def active_branch(self) -> str:
        """Return the name of the currently checked-out branch."""
        return str(self.repo.active_branch)

    def reset(self) -> None:
        """Discard all uncommitted local changes."""
        self.repo.git.reset("--hard", "HEAD")

    def delete_branch(self, branch_name: str, force: bool = False) -> None:
        """Delete a local branch."""
        self.repo.delete_head(branch_name, force=force)

    def status(self) -> str:
        """Return the repository status."""
        return self.repo.git.status()

    def add(self, paths: Optional[Iterable[Union[str, Path]]] = None) -> None:
        """Stage files for commit. If paths is None, stage all changes."""
        if paths:
            self.repo.index.add([str(path) for path in paths])
        else:
            self.repo.git.add(all=True)

    def commit(self, commit_msg: str) -> None:
        """Commit staged changes with the given message."""
        self.repo.index.commit(commit_msg)

    def create_branch(self, branch_name: str):
        """Create a new local branch."""
        return self.repo.create_head(branch_name)

    @property
    def local_branches(self) -> list[str]:
        """Return the names of local branches."""
        return [head.name for head in self.repo.heads]

    @property
    def remote_branches(self) -> list[str]:
        """Return the names of remote branches."""
        return [
            ref.remote_head
            for ref in self.repo.remote().refs
            if ref.remote_head != "HEAD"
        ]

    @property
    def commits(self) -> list[dict[str, str]]:
        """Return recent commits as dictionaries."""
        commit_log = self.repo.git.log(
            '--pretty={"commit":"%h","author":"%an","summary":"%s","date":"%cd"}',
            max_count=50,
            date="format:%Y-%m-%d %H:%M",
        )
        return [json.loads(line) for line in commit_log.splitlines() if line]

    @property
    def tags(self) -> list[str]:
        """Return all tag names."""
        return [tag.name for tag in self.repo.tags]

    def change_to_branch(self, branch: str) -> str:
        """Checkout an existing branch."""
        return self.repo.git.checkout(branch)

    def change_to_commit(self, branch: str, commit: str) -> str:
        """Checkout a branch and reset it to a specific commit."""
        self.change_to_branch(branch)
        # --soft 只重置 HEAD，不修改工作区和暂存区，保留未提交的更改
        return self.repo.git.reset("--soft", commit)

    def change_to_tag(self, tag: str) -> str:
        """Checkout a tag."""
        return self.repo.git.checkout(tag)

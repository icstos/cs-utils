import os
import json
from pathlib import Path
from typing import Iterable, Optional, Union

from git import Repo


class Git:
    @staticmethod
    def set_remote(git_repo_path: Path, repo_name: str | None):
        if git_repo_path.exists():
            if repo_name is None:
                repo_name = git_repo_path.name
            os.chdir(str(git_repo_path.absolute()))
            for command in Git.get_set_remote_command(git_repo_path, repo_name):
                os.system(command=command)
            return True

    @staticmethod
    def get_set_remote_command(git_repo_path: Path, repo_name: str | None):
        commands = []
        commands.append(f"git remote add cshub   git@192.168.1.100:cs/{repo_name}.git")
        commands.append(f"git remote add gitee   git@gitee.com:icstos/{repo_name}.git")
        commands.append(
            f"git remote add gitcode git@gitcode.com:icstos/{repo_name}.git"
        )
        commands.append(f"git remote add github  git@github.com:icstos/{repo_name}.git")
        return commands


class GitRepo:
    """基于 GitPython 的轻量 Git 仓库封装。"""

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
        """确保本地目录中存在可用的 Git 仓库。"""
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
        """从指定远程分支拉取最新代码。"""
        return self.repo.git.pull("--progress", remote, branch)

    def push(self, remote: str = "origin", branch: str = "main") -> None:
        """把当前分支推送到指定远程。"""
        self.repo.remote(remote).push(branch)

    @property
    def active_branch(self) -> str:
        """返回当前检出的分支名。"""
        return str(self.repo.active_branch)

    def reset(self) -> None:
        """丢弃所有未提交的本地改动。"""
        self.repo.git.reset("--hard", "HEAD")

    def delete_branch(self, branch_name: str, force: bool = False) -> None:
        """删除本地分支。"""
        self.repo.delete_head(branch_name, force=force)

    def status(self) -> str:
        """返回仓库状态。"""
        return self.repo.git.status()

    def add(self, paths: Optional[Iterable[Union[str, Path]]] = None) -> None:
        """把文件加入暂存区；paths 为 None 时暂存全部改动。"""
        if paths:
            self.repo.index.add([str(path) for path in paths])
        else:
            self.repo.git.add(all=True)

    def commit(self, commit_msg: str) -> None:
        """用给定信息提交已暂存的改动。"""
        self.repo.index.commit(commit_msg)

    def create_branch(self, branch_name: str):
        """创建新的本地分支。"""
        return self.repo.create_head(branch_name)

    @property
    def local_branches(self) -> list[str]:
        """返回本地分支名列表。"""
        return [head.name for head in self.repo.heads]

    @property
    def remote_branches(self) -> list[str]:
        """返回远程分支名列表。"""
        return [
            ref.remote_head
            for ref in self.repo.remote().refs
            if ref.remote_head != "HEAD"
        ]

    @property
    def commits(self) -> list[dict[str, str]]:
        """以字典列表返回最近的提交记录。"""
        commit_log = self.repo.git.log(
            '--pretty={"commit":"%h","author":"%an","summary":"%s","date":"%cd"}',
            max_count=50,
            date="format:%Y-%m-%d %H:%M",
        )
        return [json.loads(line) for line in commit_log.splitlines() if line]

    @property
    def tags(self) -> list[str]:
        """返回所有标签名。"""
        return [tag.name for tag in self.repo.tags]

    def change_to_branch(self, branch: str) -> str:
        """切换到已存在的分支。"""
        return self.repo.git.checkout(branch)

    def change_to_commit(self, branch: str, commit: str) -> str:
        """切换到某个分支并把它重置到指定提交。"""
        self.change_to_branch(branch)
        # --soft 只重置 HEAD，不修改工作区和暂存区，保留未提交的更改
        return self.repo.git.reset("--soft", commit)

    def change_to_tag(self, tag: str) -> str:
        """切换到某个标签。"""
        return self.repo.git.checkout(tag)

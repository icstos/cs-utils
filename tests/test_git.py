import sys
from pathlib import Path

import pytest

# 确保包源码可以被导入
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import types

# 测试收集阶段先放一个最小可用的假 `git` 模块，
# 这样即使测试环境没装 GitPython，导入 `cs_utils.git` 也不会失败。
fake_git = types.ModuleType("git")
fake_git.Repo = object  # 占位符，测试中会被 FakeRepo 替换
import sys

sys.modules.setdefault("git", fake_git)

import cs.git as gitmod
from cs.git import GitRepo


class FakeGit:
    def __init__(self):
        self.calls = []

    def pull(self, *args, **kwargs):
        self.calls.append(("pull", args, kwargs))
        return "pulled"

    def reset(self, *args, **kwargs):
        self.calls.append(("reset", args, kwargs))
        return "reset"

    def status(self, *args, **kwargs):
        self.calls.append(("status", args, kwargs))
        return "on branch fake"

    def add(self, *args, **kwargs):
        self.calls.append(("add", args, kwargs))

    def checkout(self, *args, **kwargs):
        self.calls.append(("checkout", args, kwargs))
        return "checked out"

    def log(self, *args, **kwargs):
        # 默认 log 返回空字符串；需要时测试可以覆盖 FakeRepo.git 上的实现。
        return ""


class FakeRemote:
    def __init__(self, refs=None):
        # GitPython 的 `remote().refs` 是属性（可迭代），不是方法。
        self.refs = refs or []
        self.pushed = []

    def push(self, branch):
        self.pushed.append(branch)


class FakeRef:
    def __init__(self, remote_head):
        self.remote_head = remote_head


class FakeTag:
    def __init__(self, name):
        self.name = name


class FakeHead:
    def __init__(self, name):
        self.name = name


class FakeRepo:
    clone_called = False
    clone_args = None

    def __init__(self, path=None):
        self.path = path
        self.git = FakeGit()
        self._remote = FakeRemote(refs=[FakeRef("main"), FakeRef("dev")])
        self.index = self
        self.heads = [FakeHead("main"), FakeHead("dev")]
        self.tags = [FakeTag("v1.0"), FakeTag("v2.0")]
        self.active_branch = "main"

    @classmethod
    def clone_from(cls, url, to_path, branch=None):
        cls.clone_called = True
        cls.clone_args = (url, to_path, branch)
        return cls(to_path)

    def remote(self, name=None):
        return self._remote

    # 索引（暂存区）操作
    def add(self, paths):
        self.git.add(paths)

    def commit(self, msg):
        # 模拟提交
        return f"committed: {msg}"

    def create_head(self, name):
        h = FakeHead(name)
        self.heads.append(h)
        return h

    def delete_head(self, name, force=False):
        self.heads = [h for h in self.heads if h.name != name]

    def git_log(self, *args, **kwargs):
        return self.git.log(*args, **kwargs)


def setup_fake_repo(monkeypatch, tmp_path: Path):
    # 把模块里的 Repo 替换为 FakeRepo
    monkeypatch.setattr(gitmod, "Repo", FakeRepo)


def test_clone_when_no_git_dir(monkeypatch, tmp_path: Path):
    setup_fake_repo(monkeypatch, tmp_path)

    # 确保目录下没有 .git
    repo = GitRepo(
        local_path=tmp_path, repo_url="https://example.com/repo.git", branch="main"
    )

    assert FakeRepo.clone_called is True
    assert isinstance(repo.repo, FakeRepo)


def test_use_existing_repo_when_git_dir_exists(monkeypatch, tmp_path: Path):
    # 创建 .git 目录
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(gitmod, "Repo", FakeRepo)

    repo = GitRepo(local_path=tmp_path, repo_url=None)
    assert isinstance(repo.repo, FakeRepo)


def test_pull_push_and_status(monkeypatch, tmp_path: Path):
    setup_fake_repo(monkeypatch, tmp_path)
    repo = GitRepo(local_path=tmp_path, repo_url="url")

    res = repo.pull(remote="origin", branch="main")
    assert res == "pulled"

    repo.push(remote="origin", branch="main")
    # 这里的 FakeRemote.push 会自行记录调用，无需经过 repo.remote().push
    assert repo.repo.remote().pushed == ["main"]

    status = repo.status()
    assert "on branch" in status


def test_add_commit_and_create_delete_branch(monkeypatch, tmp_path: Path):
    setup_fake_repo(monkeypatch, tmp_path)
    repo = GitRepo(local_path=tmp_path, repo_url="url")

    repo.add()
    repo.commit("msg")

    h = repo.create_branch("feature")
    assert h.name == "feature"
    assert "feature" in repo.local_branches

    repo.delete_branch("feature")
    assert "feature" not in repo.local_branches


def test_branches_commits_tags_and_checkouts(monkeypatch, tmp_path: Path):
    setup_fake_repo(monkeypatch, tmp_path)

    # 让 git.log 返回与真实实现一致的 JSON 行
    def fake_log(*args, **kwargs):
        return '{"commit":"abc","author":"me","summary":"s","date":"2020-01-01"}\n'

    FakeRepo.git = FakeGit()

    repo = GitRepo(local_path=tmp_path, repo_url="url")
    # 确保仓库实例使用我们伪造的 log 实现
    repo.repo.git.log = fake_log

    assert repo.local_branches == ["main", "dev"]
    assert repo.remote_branches == ["main", "dev"]

    commits = repo.commits
    assert isinstance(commits, list) and commits[0]["commit"] == "abc"

    assert repo.tags == ["v1.0", "v2.0"]

    co = repo.change_to_branch("dev")
    assert "checked out" in co

    r = repo.change_to_commit("dev", "abc123")
    assert r == "reset"

    t = repo.change_to_tag("v1.0")
    assert "checked out" in t


def test_context_manager_support(monkeypatch, tmp_path: Path):
    setup_fake_repo(monkeypatch, tmp_path)

    with GitRepo(local_path=tmp_path, repo_url="url") as repo:
        assert isinstance(repo, GitRepo)
        assert repo.active_branch == "main"

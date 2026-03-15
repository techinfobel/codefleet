"""Tests for git_ops.py - real git operations against temporary repositories."""

import subprocess
from pathlib import Path

import pytest

from agent_fleet_supervisor.git_ops import (
    GitError,
    create_worktree,
    delete_branch,
    get_git_path,
    get_repo_root,
    is_git_repo,
    remove_worktree,
    resolve_ref,
)


class TestIsGitRepo:
    def test_valid_repo(self, git_repo):
        assert is_git_repo(git_repo) is True

    def test_not_a_repo(self, tmp_path):
        plain_dir = tmp_path / "not_a_repo"
        plain_dir.mkdir()
        assert is_git_repo(plain_dir) is False

    def test_nonexistent_path(self, tmp_path):
        assert is_git_repo(tmp_path / "missing") is False

    def test_subdirectory_of_repo(self, git_repo):
        subdir = git_repo / "subdir"
        subdir.mkdir()
        assert is_git_repo(subdir) is True


class TestResolveRef:
    def test_resolve_head(self, git_repo):
        sha = resolve_ref(git_repo, "HEAD")
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_resolve_invalid_ref(self, git_repo):
        with pytest.raises(GitError, match="Failed to resolve"):
            resolve_ref(git_repo, "nonexistent-ref-xyz")


class TestCreateWorktree:
    def test_create_worktree(self, git_repo, tmp_path):
        wt_path = tmp_path / "worktree1"
        branch = "test/worktree-branch"
        create_worktree(git_repo, wt_path, branch, "HEAD")

        assert wt_path.exists()
        assert (wt_path / "README.md").exists()
        assert is_git_repo(wt_path)

        # Verify branch was created
        result = subprocess.run(
            ["git", "-C", str(git_repo), "branch", "--list", branch],
            capture_output=True,
            text=True,
        )
        assert branch in result.stdout

    def test_create_worktree_nested_parent(self, git_repo, tmp_path):
        wt_path = tmp_path / "deep" / "nested" / "worktree"
        create_worktree(git_repo, wt_path, "test/nested-branch", "HEAD")
        assert wt_path.exists()

    def test_create_worktree_invalid_ref(self, git_repo, tmp_path):
        wt_path = tmp_path / "worktree_bad"
        with pytest.raises(GitError, match="Failed to create worktree"):
            create_worktree(git_repo, wt_path, "test/bad-ref", "nonexistent-ref")

    def test_create_duplicate_branch_fails(self, git_repo, tmp_path):
        wt1 = tmp_path / "wt1"
        wt2 = tmp_path / "wt2"
        create_worktree(git_repo, wt1, "test/dup-branch", "HEAD")
        with pytest.raises(GitError):
            create_worktree(git_repo, wt2, "test/dup-branch", "HEAD")


class TestRemoveWorktree:
    def test_remove_worktree(self, git_repo, tmp_path):
        wt_path = tmp_path / "wt_remove"
        create_worktree(git_repo, wt_path, "test/remove-branch", "HEAD")
        assert wt_path.exists()

        remove_worktree(git_repo, wt_path)
        assert not wt_path.exists()

    def test_remove_nonexistent_worktree(self, git_repo, tmp_path):
        # Should not raise - function handles missing worktrees gracefully
        remove_worktree(git_repo, tmp_path / "missing_wt")


class TestDeleteBranch:
    def test_delete_branch(self, git_repo, tmp_path):
        wt_path = tmp_path / "wt_del"
        branch = "test/del-branch"
        create_worktree(git_repo, wt_path, branch, "HEAD")
        remove_worktree(git_repo, wt_path)

        delete_branch(git_repo, branch)

        result = subprocess.run(
            ["git", "-C", str(git_repo), "branch", "--list", branch],
            capture_output=True,
            text=True,
        )
        assert branch not in result.stdout

    def test_delete_nonexistent_branch(self, git_repo):
        with pytest.raises(GitError, match="Failed to delete branch"):
            delete_branch(git_repo, "nonexistent/branch")


class TestGetGitPath:
    def test_returns_path(self):
        path = get_git_path()
        assert path is not None
        assert "git" in path


class TestGetRepoRoot:
    def test_repo_root(self, git_repo):
        root = get_repo_root(git_repo)
        assert root is not None
        assert root == git_repo

    def test_repo_root_from_subdir(self, git_repo):
        subdir = git_repo / "sub"
        subdir.mkdir()
        root = get_repo_root(subdir)
        assert root == git_repo

    def test_non_repo(self, tmp_path):
        plain = tmp_path / "plain"
        plain.mkdir()
        root = get_repo_root(plain)
        assert root is None


class TestWorktreeIsolation:
    """Test that multiple worktrees are truly isolated."""

    def test_independent_worktrees(self, git_repo, tmp_path):
        wt1 = tmp_path / "wt_iso1"
        wt2 = tmp_path / "wt_iso2"

        create_worktree(git_repo, wt1, "test/iso1", "HEAD")
        create_worktree(git_repo, wt2, "test/iso2", "HEAD")

        # Write different files in each
        (wt1 / "file1.txt").write_text("from wt1")
        (wt2 / "file2.txt").write_text("from wt2")

        # Each worktree should only have its own file
        assert (wt1 / "file1.txt").exists()
        assert not (wt1 / "file2.txt").exists()
        assert (wt2 / "file2.txt").exists()
        assert not (wt2 / "file1.txt").exists()

        # Cleanup
        remove_worktree(git_repo, wt1)
        remove_worktree(git_repo, wt2)

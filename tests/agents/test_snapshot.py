"""Edge cases of the snapshot/rollback used by the SatisFix agent.

Every case takes a snapshot, lets a "step" change the repository, rolls back, and requires that
everything an agent could observe is identical to before the step: file contents, modes, symlinks,
directories and their mtimes (to the nanosecond), `git status`, branches, log, stash, reflog,
config, and the files under `.git` (objects excepted).
"""

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from minisweagent.agents.utils.snapshot import RepoSnapshots, Snapshot
from minisweagent.environments.local import LocalEnvironment

GIT_OBSERVABLES = [
    "git --no-optional-locks status --porcelain=v2 --untracked-files=all --branch",
    "git branch --list",
    "git log --oneline --all",
    "git stash list",
    "git reflog --all | wc -l",
    "git config --local --list",
    "cd .git && find . -type f ! -path './objects/*' | sort",
]


def sh(repo: Path, command: str) -> str:
    return subprocess.run(command, shell=True, cwd=repo, capture_output=True, text=True).stdout


def observable(repo: Path) -> dict:
    """Everything an agent could notice, keyed by path or command."""
    seen: dict = {}
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if not (root == str(repo) and d == ".git")]
        for name in dirs + files:
            path = Path(root) / name
            rel = str(path.relative_to(repo))
            st = os.lstat(path)
            if path.is_symlink():
                seen[rel] = ("link", os.readlink(path))
            elif path.is_dir():
                seen[rel] = ("dir", st.st_mtime_ns)
            else:
                seen[rel] = ("file", st.st_mtime_ns, st.st_mode & 0o777, hashlib.sha1(path.read_bytes()).hexdigest())
    seen["."] = ("dir", os.stat(repo).st_mtime_ns)
    for command in GIT_OBSERVABLES:
        seen[command] = sh(repo, command)
    seen["index"] = hashlib.sha1((repo / ".git/index").read_bytes()).hexdigest()
    return seen


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    sh(
        root,
        "git init -q -b main . && git config user.email a@b && git config user.name a && "
        "mkdir -p pkg/sub empty && echo 'x=1' > pkg/a.py && echo 'y=1' > pkg/sub/b.py && echo run > run.py && "
        "echo '__pycache__/\nbuild/' > .gitignore && ln -s pkg/a.py link.py && printf 'spaced' > 'my file.txt' && "
        "printf 'u' > 文件.txt && git add -A && git commit -qm init",
    )
    return root


def snapshots(repo: Path, tmp_path: Path) -> RepoSnapshots:
    return RepoSnapshots(LocalEnvironment(cwd=str(repo)), str(repo), state_dir=str(tmp_path / "state"))


def roundtrip(repo: Path, tmp_path: Path, step: str, ignore: set[str] = frozenset()) -> dict:
    """Snapshot, run `step`, snapshot again (must differ), roll back, and compare observables."""
    snaps = snapshots(repo, tmp_path)
    before = observable(repo)
    s0 = snaps.take()
    assert s0 is not None and s0.complete
    sh(repo, step)
    s1 = snaps.take()
    assert s1.tree != s0.tree, "the step must change the tree for this case to test anything"
    ok, details = snaps.rollback(s0)
    assert ok, details
    after = observable(repo)
    assert {k: v for k, v in after.items() if k not in ignore} == {k: v for k, v in before.items() if k not in ignore}
    return details


STEPS = {
    "modify tracked file": "echo 'x=2' >> pkg/a.py",
    "add file in new directory": "mkdir -p new/deep && echo n > new/deep/n.py",
    "delete tracked file": "rm pkg/sub/b.py",
    "rename tracked file": "mv pkg/a.py pkg/c.py",
    "chmod": "chmod +x run.py",
    "delete symlink": "rm link.py",
    "delete directory with files": "rm -rf pkg",
    "edit and stage": "echo 'x=2' > pkg/a.py && git add pkg/a.py",
    "stage a new file": "echo new > staged.py && git add staged.py",
    "commit on a new branch": "echo 'x=2' > pkg/a.py && git checkout -q -b feature && git commit -qam step",
    "stash": "echo 'x=2' > pkg/a.py && git stash -q && echo z > z.py",
    "config and edit": "git config user.name other && echo 'x=2' > pkg/a.py",
    "names with spaces and unicode": "echo more >> 'my file.txt' && rm 文件.txt && echo new > 'new file.txt'",
    "python rewrite then build artefacts in /tmp": "python3 -c \"open('pkg/a.py','w').write('x=3\\n')\" && echo tmp > /tmp/satisfix-test-tmp",
    "file replaced by a directory": "mv pkg/a.py /tmp/satisfix-test-a && mkdir pkg/a.py && mv /tmp/satisfix-test-a pkg/a.py/x",
    "directory replaced by a file": "rm -r pkg/sub && echo flat > pkg/sub",
    "symlink replaced by a file": "rm link.py && echo plain > link.py",
    "same content rewritten, another file changed": "cp pkg/a.py /tmp/satisfix-test-a && rm pkg/a.py && cp /tmp/satisfix-test-a pkg/a.py && touch run.py && echo z > z.py",
    "stale index.lock left behind": "echo 'x=2' > pkg/a.py && touch .git/index.lock",
}


@pytest.mark.parametrize("step", list(STEPS.values()), ids=list(STEPS))
def test_rollback_leaves_no_trace(repo: Path, tmp_path: Path, step: str):
    roundtrip(repo, tmp_path, step)


def test_symlink_target_mtime_untouched(repo: Path, tmp_path: Path):
    """Restoring a symlink must not touch the file it points to."""
    roundtrip(repo, tmp_path, "rm link.py && echo 'x=2' >> pkg/a.py")


def test_dirty_baseline_is_preserved(repo: Path, tmp_path: Path):
    sh(repo, "echo 'x=0' >> pkg/a.py && echo s > staged.py && git add staged.py && echo u > untracked.py")
    details = roundtrip(repo, tmp_path, "echo 'y=2' >> pkg/sub/b.py && rm untracked.py && git add pkg/a.py")
    assert details["restored"] == 2 and details["removed"] == 0


def test_empty_directories(repo: Path, tmp_path: Path):
    """git does not track empty directories; the mtime listing does."""
    roundtrip(repo, tmp_path, "mkdir -p made/also && rmdir empty && echo 'x=2' >> pkg/a.py")
    assert (repo / "empty").is_dir() and not (repo / "made").exists()


def test_ignored_files_are_not_covered(repo: Path, tmp_path: Path):
    """Documented limit: a file the repository ignores survives the rollback."""
    roundtrip(
        repo,
        tmp_path,
        "mkdir build && echo b > build/out && echo 'x=2' >> pkg/a.py",
        ignore={"build", "build/out", "."},
    )
    assert (repo / "build/out").exists()


def test_many_files_are_chunked(repo: Path, tmp_path: Path):
    details = roundtrip(repo, tmp_path, "mkdir many && for i in $(seq 1 450); do echo $i > many/f$i.txt; done")
    assert details["removed"] == 450 and not (repo / "many").exists()


def test_unchanged_step_keeps_tree(repo: Path, tmp_path: Path):
    snaps = snapshots(repo, tmp_path)
    s0 = snaps.take()
    sh(repo, "touch pkg/a.py && echo tmp > /tmp/satisfix-test-tmp && git status >/dev/null")
    assert snaps.take().tree == s0.tree


def test_snapshot_files_are_pruned_to_the_last_two(repo: Path, tmp_path: Path):
    snaps = snapshots(repo, tmp_path)
    for i in range(4):
        sh(repo, f"echo {i} > f{i}.txt")
        snaps.take()
    assert len(list((tmp_path / "state").glob("meta-*.tar"))) == 2
    assert len(list((tmp_path / "state").glob("mtimes-*"))) == 2


def test_not_a_repository(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert snapshots(plain, tmp_path).take() is None


def test_failed_rollback_reports_details(repo: Path, tmp_path: Path):
    snaps = snapshots(repo, tmp_path)
    snaps.take()
    ok, details = snaps.rollback(Snapshot(tree="0" * 40, status="# branch.head elsewhere\n"))
    assert not ok and details["attempts"] == 2 and details["tree_match"] is False
    assert details["target_status"] == "# branch.head elsewhere\n" and details["status_after"] and details["commands"]


def test_private_objects_stay_out_of_the_repository(repo: Path, tmp_path: Path):
    snaps = snapshots(repo, tmp_path)
    before = sh(repo, "git count-objects -v")
    snaps.take()
    sh(repo, "echo 'x=2' >> pkg/a.py")
    s1 = snaps.take()
    snaps.rollback(snaps.take() and Snapshot(s1.tree, s1.status))  # a no-op rollback
    assert sh(repo, "git count-objects -v") == before
    assert (tmp_path / "state" / "objects").is_dir()

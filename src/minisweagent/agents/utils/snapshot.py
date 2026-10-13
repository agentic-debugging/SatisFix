"""Snapshots of a git working tree, and a rollback the agent cannot notice.

A snapshot is a tree object written through a private index and a private object directory
(`GIT_INDEX_FILE`, `GIT_OBJECT_DIRECTORY`; the repository's objects are read through
`GIT_ALTERNATE_OBJECT_DIRECTORIES`), so the repository's own index, HEAD, refs and reflog are
never written. Alongside the tree, a snapshot keeps the repository's `.git` metadata (everything
but objects), the modification times of the files and directories git sees, and the output of
`git status --branch`, so a rollback restores what `git status`, `git branch`, `git log`, `ls -l`
and `find` show and is checked against the tree and the status. Not covered: ignored files,
changes outside the repository, nested repositories, processes.
"""

import shlex
import uuid
from dataclasses import dataclass

from minisweagent import Environment

GIT = "git -c safe.directory='*' -c core.quotePath=false"
STATUS = f"{GIT} --no-optional-locks status --porcelain=v2 --untracked-files=all --branch"
DIRS = "find . -type d ! -path ./.git ! -path './.git/*'"
MTIMES = f"{{ {GIT} ls-files -z; {DIRS} -print0; }} | xargs -0 stat --printf='%.9Y\\t%n\\n'"
CHUNK = 200  # paths per shell command


@dataclass
class Snapshot:
    tree: str
    status: str
    complete: bool = True
    """Whether the mtime listing and `.git` metadata were saved as well."""


class RepoSnapshots:
    def __init__(self, env: Environment, repo_dir: str = "", state_dir: str = ""):
        """Snapshots of the repository at `repo_dir` (the environment's working directory by default)."""
        self.env = env
        self.repo_dir = repo_dir
        self.state_dir = state_dir or f"/var/tmp/.snapshots-{uuid.uuid4().hex[:8]}"
        self.git_dir: str | None = None
        self.top: str | None = None
        self.kept: list[str] = []
        self.log: list[dict] = []
        """Every command run and what it returned, for the record of a rollback."""

    def _run(self, command: str) -> dict:
        out = self.env.execute({"command": command}, cwd=self.repo_dir, timeout=600)
        self.log.append({"command": command[:200], "returncode": out["returncode"], "output": out["output"][:500]})
        return out

    def _private(self, command: str) -> dict:
        """Run with the private index and object directory, from the repository's top level."""
        env = (
            f"GIT_INDEX_FILE={self.state_dir}/index GIT_OBJECT_DIRECTORY={self.state_dir}/objects "
            f"GIT_ALTERNATE_OBJECT_DIRECTORIES={shlex.quote(self.git_dir)}/objects"
        )
        return self._run(f"export {env}; cd {shlex.quote(self.top)} && {command}")

    def _status(self) -> str:
        """`git status --porcelain=v2 --branch`, keeping only its record lines (stderr is merged into the output)."""
        out = self._run(f"cd {shlex.quote(self.top)} && {STATUS}")["output"]
        return "".join(line + "\n" for line in out.splitlines() if line[:2] in ("# ", "1 ", "2 ", "u ", "? ", "! "))

    def available(self) -> bool:
        if self.git_dir is None:
            out = self._run(f"{GIT} rev-parse --absolute-git-dir --show-toplevel")
            if out["returncode"] != 0 or len(lines := out["output"].split()) != 2:
                return False
            self.git_dir, self.top = lines
            self._run(f"mkdir -p {self.state_dir}/objects")
        return True

    def _write_tree(self) -> str | None:
        out = self._private(f"{GIT} add -A . && {GIT} write-tree")
        return out["output"].split()[-1] if out["returncode"] == 0 and out["output"].split() else None

    def take(self) -> Snapshot | None:
        """Snapshot the working tree now; None if the directory is not inside a git repository."""
        if not self.available() or (tree := self._write_tree()) is None:
            return None
        d, git_dir = self.state_dir, shlex.quote(self.git_dir)
        saved = self._private(
            f"{MTIMES} > {d}/mtimes-{tree} && tar -C {git_dir} --exclude=./objects -cf {d}/meta-{tree}.tar ."
        )
        for old in self.kept[:-1]:  # keep the previous snapshot's files for a rollback, drop older ones
            self._run(f"rm -f {d}/mtimes-{old} {d}/meta-{old}.tar")
        self.kept = [t for t in self.kept[-1:] if t != tree] + [tree]
        return Snapshot(tree, self._status(), complete=saved["returncode"] == 0)

    def rollback(self, target: Snapshot) -> tuple[bool, dict]:
        """Restore the working tree, mtimes and `.git` metadata of `target`; checked by tree and status.

        The details name what was restored and, on failure, both status outputs and every command run.
        """
        details: dict = {"attempts": 0, "target_complete": target.complete}
        self.log = []
        for _ in range(2):
            details["attempts"] += 1
            current = self._write_tree()
            if current is None:
                break
            pairs = self._private(f"{GIT} diff-tree -r -z --name-status {current} {target.tree}")["output"].split("\0")
            # relative to the current tree: D = the step added it; A, M, T (type change) = restore it
            restore = [p for s, p in zip(pairs[::2], pairs[1::2]) if s != "D"]
            remove = [p for s, p in zip(pairs[::2], pairs[1::2]) if s == "D"]
            mtimes = self._mtimes(target.tree)
            dirs = {p for p in mtimes if p.endswith("/")}
            # the step's files and directories go first (a directory may stand where a file belongs)
            for chunk in _chunks(remove):
                self._private(f"rm -f -- {shlex.join(chunk)}")
            now = {_dirkey(p) for p in self._private(DIRS)["output"].splitlines()}
            for path in sorted(now - dirs, key=len, reverse=True):
                self._private(f"rmdir {shlex.quote(path)} 2>/dev/null; true")
            self._private(f"{GIT} read-tree {target.tree}")
            for chunk in _chunks(restore):
                self._private(f"{GIT} checkout-index -f -- {shlex.join(chunk)}")
            for chunk in _chunks(sorted(dirs - now, key=len)):
                self._private(f"mkdir -p -- {shlex.join(chunk)}")
            # every path whose mtime differs from the record is touched: restored files, directories
            # whose entries changed, and files the step rewrote with the same content
            current = self._parse_mtimes(self._private(MTIMES)["output"])
            stale = [p for p, ts in mtimes.items() if current.get(p) != ts]
            for chunk in _chunks([f"touch -h -m -d @{mtimes[p]} -- {shlex.quote(p)}" for p in sorted(stale)]):
                self._private("\n".join(chunk))
            self._restore_metadata(target.tree)
            after = Snapshot(self._write_tree() or "", self._status())
            details |= {
                "restored": len(restore),
                "removed": len(remove),
                "tree_match": after.tree == target.tree,
                "status_match": after.status == target.status,
            }
            if details["tree_match"] and details["status_match"]:
                self.kept = [t for t in self.kept if t != after.tree] + [after.tree]
                return True, details
        details |= {"target_status": target.status, "status_after": self._status(), "commands": self.log}
        return False, details

    def _restore_metadata(self, tree: str) -> None:
        """Unpack the saved `.git` metadata and drop files git wrote since (new refs, stash, locks)."""
        git_dir, tar = shlex.quote(self.git_dir), f"{self.state_dir}/meta-{tree}.tar"
        listing = self._run(f"tar -tf {tar}")
        if listing["returncode"] != 0 or self._run(f"tar -C {git_dir} -xf {tar}")["returncode"] != 0:
            return  # nothing to prune against: leave the metadata as it is
        saved = {line.strip() for line in listing["output"].splitlines()}
        present = self._run(f"cd {git_dir} && find . -type f ! -path './objects/*'")["output"].splitlines()
        for chunk in _chunks([p for p in present if p and p not in saved]):
            self._run(f"cd {git_dir} && rm -f -- {shlex.join(chunk)}")

    def _mtimes(self, tree: str) -> dict[str, str]:
        """path -> mtime (seconds, 9 decimals) as recorded with `tree`; directories end with `/`, the top is `./`."""
        return self._parse_mtimes(self._private(f"cat {self.state_dir}/mtimes-{tree}")["output"])

    @staticmethod
    def _parse_mtimes(listing: str) -> dict[str, str]:
        mtimes = {}
        for line in listing.splitlines():
            if "\t" in line:
                ts, path = line.split("\t", 1)
                mtimes[_dirkey(path) if path == "." or path.startswith("./") else path] = ts
        return mtimes


def _dirkey(path: str) -> str:
    """Key of a directory from `find .` output: `./`, `a/`, `a/b/`."""
    return "./" if path == "." else path.removeprefix("./") + "/"


def _chunks(items: list[str], size: int = CHUNK) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]

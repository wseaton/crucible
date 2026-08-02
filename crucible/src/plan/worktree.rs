//! Workspace isolation for plan tasks: a private clone per task, and the diff plumbing to
//! carry work out of one. An isolated task's edits never touch the shared workspace: what
//! leaves is its structured output (and, where the runner asks for it, a captured diff).
//!
//! Used by the wide tournament's parallel proposers and by any plan task marked
//! `isolation = "worktree"`.

use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Output};

use anyhow::{Context, Result};

#[derive(Debug)]
struct GitFailure {
    operation: String,
    status: ExitStatus,
    stderr: String,
}

impl std::fmt::Display for GitFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} failed with {}: {}",
            self.operation,
            self.status,
            self.stderr.trim()
        )
    }
}

impl std::error::Error for GitFailure {}

/// Capture uncommitted state once for reuse across a fan-out.
pub(crate) fn snapshot(workspace: &Path) -> Result<String> {
    capture_diff(workspace)
}

/// Create a task worktree as a shallow copy of `workspace`. Uses `git clone --local`, which
/// hard-links objects, so a fan-out of N candidates costs ~one checkout each rather than N
/// full copies.
///
/// Concurrent callers should share a snapshot via [`setup_with`].
pub(crate) fn setup(workspace: &Path, dest: &Path) -> Result<()> {
    let snapshot = snapshot(workspace).context("capturing the workspace before isolation")?;
    setup_with(workspace, dest, &snapshot)
}

/// Create a worktree from a caller-provided snapshot.
pub(crate) fn setup_with(workspace: &Path, dest: &Path, snapshot: &str) -> Result<()> {
    if dest.exists() {
        std::fs::remove_dir_all(dest)?;
    }
    let status = std::process::Command::new("git")
        .args([
            "clone",
            "--local",
            "--no-checkout",
            &workspace.to_string_lossy(),
            &dest.to_string_lossy(),
        ])
        .output()
        .context("git clone --local for a task worktree")?;
    require_success("git clone --local", &status)?;
    // Check out HEAD so the task has a working tree.
    let checkout = std::process::Command::new("git")
        .args(["-C", &dest.to_string_lossy(), "checkout", "HEAD"])
        .output()
        .context("git checkout HEAD in a task worktree")?;
    require_success("git checkout HEAD", &checkout)?;
    // A clone omits the upstream task's uncommitted state; replay it as a patch.
    apply(dest, snapshot).context("carrying the workspace's uncommitted state into a task worktree")
}

/// Capture staged, unstaged, and untracked state as a binary patch.
/// Uses a throwaway index to avoid staging or `.git/index.lock` contention.
pub(crate) fn capture_diff(worktree: &Path) -> Result<String> {
    let scratch = tempfile::tempdir().context("creating a temporary Git index directory")?;
    let index = scratch.path().join("index");
    // An unborn repository has no index.
    let real = git_path(worktree, "index")?;
    match std::fs::copy(&real, &index) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error)
                .with_context(|| format!("copying Git index from {}", real.display()));
        }
    }
    git_with_index(worktree, &index, &["add", "-A"])?;
    let output = git_with_index(worktree, &index, &["diff", "--cached", "--binary"])?;
    String::from_utf8(output.stdout).context("captured Git diff is not UTF-8")
}

fn git_with_index(worktree: &Path, index: &Path, args: &[&str]) -> Result<Output> {
    let operation = format!("git {}", args.join(" "));
    let output = std::process::Command::new("git")
        .args(["-C", &worktree.to_string_lossy()])
        .args(args)
        .env("GIT_INDEX_FILE", index)
        .output()
        .with_context(|| format!("spawning {operation}"))?;
    require_success(&operation, &output)?;
    Ok(output)
}

fn require_success(operation: &str, output: &Output) -> Result<()> {
    if output.status.success() {
        return Ok(());
    }
    Err(GitFailure {
        operation: operation.to_string(),
        status: output.status,
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
    .into())
}

/// Resolve a path inside the git dir; a linked worktree's is not `<worktree>/.git`.
fn git_path(worktree: &Path, name: &str) -> Result<PathBuf> {
    let out = std::process::Command::new("git")
        .args(["-C", &worktree.to_string_lossy(), "rev-parse", "--git-path"])
        .arg(name)
        .output()
        .context("git rev-parse --git-path")?;
    require_success("git rev-parse --git-path", &out)?;
    let path = PathBuf::from(
        String::from_utf8(out.stdout)
            .context("git rev-parse --git-path returned non-UTF-8")?
            .trim(),
    );
    Ok(if path.is_absolute() {
        path
    } else {
        worktree.join(path)
    })
}

/// Apply a captured diff to a workspace via `git apply` on stdin. An empty diff is a no-op.
pub(crate) fn apply(main_ws: &Path, diff: &str) -> Result<()> {
    if diff.trim().is_empty() {
        return Ok(());
    }
    let mut apply = std::process::Command::new("git")
        .args([
            "-C",
            &main_ws.to_string_lossy(),
            "apply",
            "--allow-empty",
            "-",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context("git apply in the main workspace")?;

    if let Some(mut stdin) = apply.stdin.take() {
        use std::io::Write;
        stdin.write_all(diff.as_bytes())?;
    }

    let output = apply.wait_with_output()?;
    require_success("git apply", &output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn git(dir: &Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .expect("git");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn capturing_a_diff_leaves_the_repository_alone() {
        let root =
            std::env::temp_dir().join(format!("crucible-wt-readonly-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        git(&root, &["init", "-q"]);
        std::fs::write(root.join("committed.txt"), "base\n").unwrap();
        git(&root, &["add", "-A"]);
        git(
            &root,
            &[
                "-c",
                "user.email=c@l",
                "-c",
                "user.name=c",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "base",
            ],
        );
        std::fs::write(root.join("untracked.txt"), "new\n").unwrap();
        std::fs::write(root.join("committed.txt"), "changed\n").unwrap();

        let staged_before = staged(&root);
        let diff = capture_diff(&root).unwrap();

        assert!(diff.contains("untracked.txt"), "untracked file: {diff}");
        assert!(diff.contains("committed.txt"), "modified file: {diff}");
        assert_eq!(
            staged(&root),
            staged_before,
            "capture staged the caller's workspace"
        );
        assert!(staged_before.is_empty(), "nothing was staged to begin with");
        let _ = std::fs::remove_dir_all(&root);
    }

    fn staged(dir: &Path) -> String {
        let out = std::process::Command::new("git")
            .args([
                "-C",
                &dir.to_string_lossy(),
                "diff",
                "--cached",
                "--name-only",
            ])
            .output()
            .expect("git");
        String::from_utf8_lossy(&out.stdout).to_string()
    }

    #[test]
    fn one_snapshot_feeds_concurrent_worktrees_identically() {
        const WORKTREES: usize = 8;
        let root =
            std::env::temp_dir().join(format!("crucible-wt-snapshot-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let workspace = root.join("workspace");
        std::fs::create_dir_all(&workspace).unwrap();
        git(&workspace, &["init", "-q"]);
        std::fs::write(workspace.join("committed.txt"), "base\n").unwrap();
        git(&workspace, &["add", "-A"]);
        git(
            &workspace,
            &[
                "-c",
                "user.email=c@l",
                "-c",
                "user.name=c",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "base",
            ],
        );
        std::fs::write(workspace.join("pending.txt"), "uncommitted-evidence\n").unwrap();

        let snapshot = snapshot(&workspace).unwrap();
        assert!(
            !snapshot.is_empty(),
            "the snapshot carries the pending file"
        );
        std::thread::scope(|scope| {
            for i in 0..WORKTREES {
                let workspace = workspace.clone();
                let dest = root.join(format!("wt-{i}"));
                let snapshot = snapshot.as_str();
                scope.spawn(move || setup_with(&workspace, &dest, snapshot).expect("setup_with"));
            }
        });

        for i in 0..WORKTREES {
            let pending = root.join(format!("wt-{i}")).join("pending.txt");
            assert_eq!(
                std::fs::read_to_string(&pending).unwrap_or_default(),
                "uncommitted-evidence\n",
                "worktree {i} is missing the candidate's uncommitted state"
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn capture_failure_is_reported_instead_of_becoming_an_empty_diff() {
        let root =
            std::env::temp_dir().join(format!("crucible-wt-not-a-repo-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let error = capture_diff(&root).unwrap_err().to_string();

        assert!(error.contains("git rev-parse --git-path"), "{error}");
        assert!(error.contains("failed with"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }
}

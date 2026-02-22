Changelog
=========

0.5.2 (2026-02-21)
-------------------
* **Fix planner agents running without system prompts or tool restrictions**:
  ``driver.py:_run_agent()`` now loads prompt templates as ``--system-prompt``
  and passes ``--allowedTools`` to the Claude CLI. Previously, all planner
  agents (architects, critics, refiners, judge) ran as generic Claude Code
  sessions with no behavioral instructions or tool restrictions. The judge
  agent was especially affected — with access to Bash/Edit and no system
  prompt constraining it to planning, it would implement code instead of
  writing the final plan.

0.5.1 (2026-02-21)
-------------------
* **Fix agent uses preset Bazel commands**: ``fix_agent.py`` now uses
  ``_select_command(step)`` instead of hardcoded ``step.run_command``, so
  fix agents run ``bazel run //pkg:lint.fix`` in Bazel repos instead of
  the nonexistent ``npm run lint:fix``. Root cause of lint permanent failure.
* **Fix false-success on blocked pipeline**: ``get_next_steps()`` returned
  "all complete" when steps were blocked by permanently failed upstream,
  causing ``dispatch_loop`` to report success. Now returns the blocked list,
  and ``dispatch_loop`` detects this as a failure.
* **Non-zero exit on pipeline failure**: ``main()`` now calls ``sys.exit(1)``
  when the pipeline fails, instead of always exiting 0.
* **Per-attempt transcript naming**: Retry attempts now get distinct transcript
  filenames (``lint_attempt1-transcript.log``, etc.) instead of overwriting.
* **Build result as required code agent output**: The code agent prompt now
  requires writing ``build-result.json`` with build command, exit code, and
  pass/fail status. Evidence rules on the code step verify this file exists
  and shows ``passed: true``. The judge rejects the step if the file is missing
  or shows a failed build — same pattern as visual test screenshots. Configurable
  build command via ``pass_command`` / ``bazel_pass_command`` on step definitions.
* **7 regression tests** for all fixes above.

0.5.0 (2026-02-21)
-------------------
* **Fix orphaned IN_PROGRESS on resume**: Steps stuck in ``IN_PROGRESS`` after
  SIGTERM are now automatically reset to ``PENDING`` on ``--resume``, using
  ``no_retry_inc=True`` so retry counts are preserved.
* **Remove fail-forward in dep_is_satisfied**: Permanently failed steps no
  longer satisfy dependencies. The pipeline halts instead of silently
  continuing. Use ``--skip <step>`` on resume to explicitly bypass.
* **Pre-PR gate rejects manual_skip**: Critical steps (``test``,
  ``visual_test``) with ``manual_skip=true`` checkpoints now fail the gate.
  Non-critical skips produce a warning.
* **Bazel command support**: Manifest steps can now define ``bazel_run_command``
  (for lint/test) and ``bazel_command`` (for dev server). When a ``WORKSPACE``
  or ``WORKSPACE.bazel`` file exists in the repo root, Bazel commands are
  selected automatically.
* **Code review scoped to diff**: The code review skill prompt now explicitly
  instructs reviewers to only flag code in the git diff, not pre-existing
  patterns.
* **21 regression tests** for all fixes above.

0.4.0 (2026-02-21)
-------------------
* **Fix session naming mismatch**: ``_session_name()`` now always prefers the
  ticket pattern from the git branch (e.g. ``DVAWV-19760``) over the planner
  slug. This ensures ``find_active_session()`` can locate sessions on ``--resume``
  and when agents call ``pipeline_cli.py pass/fail``.
* **Add pipeline_cli.py entry point**: AI agents are instructed to call
  ``python3 <path>/pipeline_cli.py pass <step>``, but the file was missing.
  Added ``executor/pipeline_cli.py`` delegating to ``commands.main()``.
* **Add concurrent driver guard**: ``_set_driver_pid()`` now checks if an
  existing ``driver_pid`` is still alive before claiming ownership. Raises
  ``RuntimeError`` if another driver is running, preventing state corruption
  from concurrent drivers.
* **Add regression tests**: ``${PRESET_DIR}`` resolution in skill paths,
  ``PIPELINE_CLI`` file existence, session name/prefix consistency,
  end-to-end session name → find_active_session round-trip, concurrent
  driver guard (alive/dead/absent PID scenarios).

0.3.0 (2026-02-21)
-------------------
* LLM-generated session slugs: session directories now get descriptive names
  (e.g. ``health-endpoint-api_2026-02-21_...``) instead of generic ``plan_...``
  or branch-derived names. Uses a fast Haiku call with sanitized fallback.
* Planner ``cmd_drive``: auto-generates slug from problem statement when ``--slug`` is omitted.
* Executor driver: reuses planner slug from ``.planner-state.json`` when ``--plan-dir``
  points to a planner session; falls back to LLM-generated slug from plan content.
* ``_session_name()`` in executor accepts optional ``slug`` parameter.
* Add ``generate_slug()`` and ``_sanitize_slug()`` to ``core/session.py``.
* Add ``_read_planner_slug()`` to ``executor/driver.py`` for cross-subsystem slug reuse.

0.2.0 (2026-02-20)
-------------------
* Consolidate planner + executor shared infrastructure into ``architect.core``.
* Extract ``core/state.py``: generic ``LockedStateManager[T]`` with fcntl locking and atomic writes.
* Extract ``core/session.py``: unified session creation, listing, cleanup, and active-session lookup.
* Extract ``core/logging.py``: UTC-standardized ``log_activity()``, ``atomic_write_file()``, ``StatusWriter``.
* Move ``AgentRunner`` to ``core/runner.py``; delete backward-compat re-export shim.
* Planner state: add ``repo_dir``, ``killed``, ``kill_reason``, ``driver_pid``; remove ``artifact_hashes``.
* Planner parallel dispatch: ``ThreadPoolExecutor`` for multi-agent phases (architects, critics, refiners).
* Planner driver reliability: signal handling (SIGINT/SIGTERM), killed-flag check, phase duration logging, repo_dir cwd fix.
* CLI additions: ``architect plan drive`` (full 6-phase orchestrator), ``architect plan kill``, ``--no-worktree`` flag.
* Config-driven evidence validation: replace hardcoded rules with ``evidence.json``.
* Delete subsystem cleanup wrappers; callers use ``core.session`` directly.

0.1.0 (2026-02-20)
-------------------
* Initial scaffold — project structure, pyproject.toml, Apache 2.0 license.
* Port planner engine: state management, evidence validation, cleanup, prompts.
* Port planner CLI (commands.py): all argparse commands with architect.planner.* imports.
* Port executor engine (PURE modules): state, registry, checkpoint, evidence, runner, cleanup, utils, templates.
* Provider protocol + Claude provider: abstract Provider interface, ClaudeProvider with stream-JSON parsing.
* Port executor NEEDS_REFACTOR modules: runner.py, judge.py, code_agent.py, fix_agent.py.
* AgentRunner shim: backward-compatible wrapper over ClaudeProvider for code/fix agents.
* Port visual_test_agent.py: extract hardcoded paths into configurable VisualTestConfig dataclass.
* Port executor CLI (commands.py): all pipeline management commands with direct imports.
* Port executor driver (driver.py): replace ~30 cli() subprocess calls with direct pipeline_ops function calls.
* Port pipeline_ops.py: shared state logic for CLI and driver.
* Add pre_pr_gate.py run_gate() function for direct invocation.
* Build PlannerDriver: deterministic 6-phase orchestrator with Provider-agnostic dispatch.
* Unified CLI entry point: ``architect plan``, ``architect execute``, ``architect drive``.
* Hz-web preset: manifest.json with DAG pipelines, evidence rules, model config, generic skill stubs.
* Claude Code skin: plugin.json + SKILL.md files for plan and execute.
* Port all test files (632 tests): test_pipeline_cli, test_revalidation, test_driver, test_pre_pr_gate.
* Remove all forbidden strings (adobe, squirrel) from source and test files.
* Replace hardcoded localhost.adobe.com URL with generic localhost URL.

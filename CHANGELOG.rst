Changelog
=========

0.7.2 (2026-03-02)
-------------------
* **``forge setup`` command**: New CLI subcommand installs required global
  tools (eslint, prettier) via ``npm install -g``. Persists state to
  ``~/.forge/setup.json``. Supports ``--force`` to re-run and ``--preset``
  for preset-specific setup (e.g. Playwright dependencies).
  ``run_preset_setup()`` reads ``setup`` entries from ``manifest.json``
  with ``${PRESET_DIR}`` substitution and idempotent ``check`` files.
* **Auto-run setup on first pipeline execution**: The executor driver now
  calls ``check_setup()`` before the dispatch loop. If tools are missing,
  ``run_setup()`` installs them automatically. Preflight hooks are now a
  hard gate — missing eslint/prettier causes ``sys.exit(1)`` instead of
  a warning.
* **Stall detection in ClaudeProvider**: Tracks ``last_activity`` timestamp,
  updated on every output line. Heartbeat thread monitors idle time and
  kills the subprocess after 15 minutes of no output (``_STALL_TIMEOUT_S``).
  Stall-killed processes (exit code -9) are treated as timeouts.
* **Phase timeouts for planner**: ``PHASE_TIMEOUT_S`` dict provides per-phase
  timeouts (900s for recon/architects, 600s for critics/refiners/judge/enrichment).
  Passed to provider via ``timeout_s`` kwarg. Previously agents could run
  indefinitely.
* **Skip already-complete agents on retry**: When a planner phase retries,
  agents whose output files already exist are skipped. Saves time and cost
  on partial failures.
* **Remove planner tool restrictions**: All planner agent types now run with
  full tool access (empty ``allowed_tools``). Previously restricted to
  Read/Write/Glob/Grep subsets which limited agent effectiveness.
* **Null plan error handling**: ``cmd_drive`` now checks for ``None`` result
  from the planner and exits with a clear error message instead of silently
  printing ``Plan: None``.
* **Multi-tier skill resolution**: New ``resolve_skill_dir()`` function in
  ``planner/commands.py`` resolves skills via 3-tier lookup: user override
  (``~/.claude/skills/``) > bundled in preset (``bundled-skills/``) > env
  var fallback (``FORGE_SKILLS_BASE``). Enrichment prompts now use
  ``{{SKILL_DIR}}`` template variable for flexible path resolution.
* **Write-incrementally prompt fix**: Added "Write the document incrementally"
  instruction to ``architect-topdown.md`` and ``architect-bottomup.md`` where
  large design documents are produced. Previously only in ``judge.md``.
* **Web dashboard**: New ``forge dashboard`` command starts an HTTP server
  for monitoring planner/executor sessions. Supports session listing,
  detail views, project grouping, and session deletion.

0.7.1 (2026-02-24)
-------------------
* **Detached HEAD session naming**: ``_session_name()`` and ``session_prefix()``
  now fall back to the short SHA when ``git rev-parse --abbrev-ref HEAD``
  returns literal ``HEAD`` (detached state), preventing broken session
  directory names and lookup failures.
* **Formula verification in code agent**: The code step prompt now instructs
  agents to verify mathematical invariants with concrete examples before
  running the build, catching formula direction errors earlier.

0.7.0 (2026-02-23)
-------------------
* **Structured event logging**: New ``forge.core.events`` module emits
  JSONL events for pipeline and planner lifecycle. Per-session ``events.jsonl``
  captures the full timeline; a global ``registry.jsonl`` indexes only lifecycle
  events (started/completed/failed/killed) for fast session discovery. Uses
  ``O_APPEND`` for atomic writes — no file locking. Schema-versioned (``v: 1``).
* **Registry rotation**: ``registry.jsonl`` rotates at 512 KB, renaming the old
  file to ``registry.<timestamp>.jsonl``. Rotated files are never deleted.
* **``forge status`` command**: New CLI subcommand reads executor and planner
  state files and outputs JSON. Supports ``--active`` (running only) and
  ``--limit`` flags.
* **Registry compaction on cleanup**: ``cleanup_sessions()`` now removes registry
  entries whose session directories no longer exist on disk.

1.0.0 (2026-02-22)
-------------------
* **Rename to Forge**: Package renamed from ``architect-ai`` to ``forgeai``.
  CLI command changed from ``architect`` to ``forge``. All internal imports
  updated from ``architect.*`` to ``forge.*``. Environment variables renamed
  from ``ARCHITECT_*`` to ``FORGE_*``. Default session directory changed
  from ``~/.architect/`` to ``~/.forge/``.
* **Open-source release**: Stripped all proprietary references. Removed
  bundled hz-web preset (presets are now user-provided via ``--preset``).
  Removed hardcoded build script references from prompt templates.
* **Preset is now required**: ``--preset`` must be explicitly provided when
  starting a new pipeline. No default preset is assumed.
* **Token cost estimation**: Planning and execution now display estimated
  token counts before starting, helping users understand cost implications.

0.5.3 (2026-02-21)
-------------------
* **Code agent now verifies build succeeds**: ``code_agent.py:run()`` now runs
  the build command after the agent completes and verifies it passes (exit code 0)
  before reporting success. Previously, the agent only checked if code changes
  were produced, allowing broken TypeScript code with compilation errors to pass
  as "success" and flow downstream to lint/test steps. Failed builds now save
  output to ``build-errors.txt`` for the next retry attempt, providing context
  for the agent to fix compilation errors. The build verification mirrors the
  same logic as ``generate_prompt()`` — selecting Bazel or npm commands based on
  repo type and affected packages.

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
* **Claude Code hooks for lint-on-edit and build verification**: PostToolUse
  hook runs eslint+prettier on every file edit/write. Stop hook runs the
  preset's build command when any agent finishes and blocks (exit 2) if it
  fails — catches build regressions from code_review and lint fix agents.
  Hook scripts in ``skins/claude-code/hooks/`` and ``scripts/``. The driver
  sets ``ARCHITECT_BUILD_CMD`` env var from top-level preset fields
  ``build_command`` / ``bazel_build_command``. Replaces the previous
  ``build-result.json`` evidence approach which relied on agent self-reporting.
* **Auto-load hooks via --plugin-dir**: ``ClaudeProvider`` reads
  ``ARCHITECT_PLUGIN_DIR`` env var and passes ``--plugin-dir`` to all
  ``claude -p`` subprocesses. The driver sets this at startup via
  ``_set_plugin_dir()``, pointing to ``skins/claude-code/``. Hooks are now
  automatically available to all executor agents without worktree changes.
* **Preset-driven eslint config**: ``eslint_config`` field in the preset
  manifest (e.g. ``"eslint_config": "tools/lint/eslint.config.mjs"``).
  The driver resolves the path and sets ``ARCHITECT_ESLINT_CONFIG`` env
  var so the lint-on-edit hook passes ``--config`` to eslint. Preflight
  warns if eslint can't find a config and suggests adding the field.
* **Preflight diagnostics at startup**: ``_preflight_hooks()`` runs before
  ``dispatch_loop`` and warns if eslint/prettier are missing or eslint
  cannot find a config for the repo. Issues are printed to stdout and
  written to the activity log for visibility when running headlessly.
* **BUILD_TARGETS fallback fix**: ``_set_hook_build_cmd()`` and
  ``code_agent.generate_prompt()`` no longer default to ``:tsc`` when
  no affected packages are known. If the template contains
  ``{{BUILD_TARGETS}}`` and no packages are available, the build command
  is skipped entirely rather than rendered with a bad fallback.
* **Hash-based build hook**: Stop hook computes a hash of repo state
  (``git diff HEAD`` + ``git status``) and only runs the build when
  state changed since the last successful check. Hash stored per session
  in ``ARCHITECT_SESSION_DIR``. Non-modifying agents automatically skip
  the build without needing special-case logic.
* **18 regression tests** for all fixes above.

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

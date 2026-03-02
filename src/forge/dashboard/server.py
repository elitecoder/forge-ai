#!/usr/bin/env python3
"""
Enhanced multi-session HTTP server for Forge with session management.
Supports multiple concurrent sessions with planner-executor linking and deletion.
"""

import json
import os
import shutil
import glob
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, parse_qs
import subprocess

ARCHITECT_DIR = Path.home() / '.forge' / 'sessions'
PLANNER_DIR = ARCHITECT_DIR / 'planner'
EXECUTOR_DIR = ARCHITECT_DIR / 'executor'
REGISTRY_FILE = ARCHITECT_DIR / 'registry.jsonl'

class EnhancedSessionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests for different endpoints."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/sessions':
            self.handle_sessions()
        elif parsed_path.path.startswith('/session/'):
            self.handle_session_detail()
        elif parsed_path.path == '/project':
            # Get project details (planner + linked executors)
            self.handle_project_detail()
        elif parsed_path.path == '/status':
            # Legacy endpoint for backward compatibility
            self.handle_legacy_status()
        else:
            self.send_error(404, "Endpoint not found")

    def do_DELETE(self):
        """Handle DELETE requests for session removal."""
        if self.path.startswith('/session/'):
            self.handle_delete_session()
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        """Handle OPTIONS requests for CORS."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def handle_sessions(self):
        """Return all sessions grouped by project."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            sessions = self.get_all_sessions()
            self.wfile.write(json.dumps(sessions, indent=2).encode())
        except Exception as e:
            error_data = {'error': str(e)}
            self.wfile.write(json.dumps(error_data).encode())

    def handle_project_detail(self):
        """Return planner + all linked executors for a project."""
        parsed_url = urlparse(self.path)
        params = parse_qs(parsed_url.query)
        planner_id = params.get('planner', [None])[0]

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            project_data = self.get_project_detail(planner_id)
            self.wfile.write(json.dumps(project_data, indent=2).encode())
        except Exception as e:
            error_data = {'error': str(e)}
            self.wfile.write(json.dumps(error_data).encode())

    def handle_session_detail(self):
        """Return detailed info for a specific session."""
        session_id = self.path.split('/')[-1]

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            session_detail = self.get_session_detail(session_id)
            self.wfile.write(json.dumps(session_detail, indent=2).encode())
        except Exception as e:
            error_data = {'error': str(e)}
            self.wfile.write(json.dumps(error_data).encode())

    def handle_delete_session(self):
        """Delete a session directory."""
        session_id = self.path.split('/')[-1]

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'DELETE')
        self.end_headers()

        try:
            result = self.delete_session(session_id)
            self.wfile.write(json.dumps(result).encode())
        except Exception as e:
            error_data = {'error': str(e), 'success': False}
            self.wfile.write(json.dumps(error_data).encode())

    def handle_legacy_status(self):
        """Legacy endpoint - returns most recent session."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            status_data = self.get_legacy_status()
            self.wfile.write(json.dumps(status_data, indent=2).encode())
        except Exception as e:
            error_data = {'error': str(e)}
            self.wfile.write(json.dumps(error_data).encode())

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session directory and return status."""
        deleted_paths = []

        # Check planner directory
        planner_path = PLANNER_DIR / session_id
        if planner_path.exists():
            # Kill any running process first
            state_file = planner_path / '.planner-state.json'
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    pid = state.get('driver_pid', 0)
                    if pid:
                        try:
                            os.kill(pid, 15)  # SIGTERM
                        except:
                            pass

            shutil.rmtree(planner_path)
            deleted_paths.append(str(planner_path))

        # Check executor directory
        executor_path = EXECUTOR_DIR / session_id
        if executor_path.exists():
            # Kill any running process first
            state_file = executor_path / 'agent-state.json'
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    pid = state.get('driver_pid', 0)
                    if pid:
                        try:
                            os.kill(pid, 15)  # SIGTERM
                        except:
                            pass

            shutil.rmtree(executor_path)
            deleted_paths.append(str(executor_path))

        if deleted_paths:
            return {
                'success': True,
                'deleted': deleted_paths,
                'message': f'Successfully deleted session {session_id}'
            }
        else:
            return {
                'success': False,
                'message': f'Session {session_id} not found'
            }

    def get_project_detail(self, planner_id: str) -> Dict[str, Any]:
        """Get planner and all linked executors."""
        project = {
            'planner': None,
            'executors': [],
            'linked': False
        }

        # Get planner details
        if planner_id:
            planner_path = PLANNER_DIR / planner_id
            if planner_path.exists():
                project['planner'] = self.get_planner_detail(planner_path)

                # Find linked executors
                if EXECUTOR_DIR.exists():
                    for exec_dir in EXECUTOR_DIR.glob('*'):
                        exec_info = self.get_executor_info(exec_dir)
                        if exec_info:
                            plan_file = exec_info.get('plan_file', '')
                            if planner_id in plan_file:
                                exec_detail = self.get_executor_detail(exec_dir)
                                project['executors'].append(exec_detail)
                                project['linked'] = True

        return project

    def get_all_sessions(self) -> Dict[str, Any]:
        """Get all sessions grouped by project."""
        sessions = {
            'projects': [],
            'total_active': 0,
            'total_sessions': 0
        }

        # Get all planner sessions
        planner_sessions = {}
        if PLANNER_DIR.exists():
            for session_dir in PLANNER_DIR.glob('*'):
                session_info = self.get_planner_info(session_dir)
                if session_info:
                    planner_sessions[str(session_dir)] = session_info

        # Get all executor sessions
        executor_sessions = {}
        if EXECUTOR_DIR.exists():
            for session_dir in EXECUTOR_DIR.glob('*'):
                session_info = self.get_executor_info(session_dir)
                if session_info:
                    executor_sessions[str(session_dir)] = session_info

        # Group sessions by project
        projects = {}

        # Process planner sessions - use session_id as unique key instead of problem statement
        for plan_path, plan_info in planner_sessions.items():
            # Create unique project key using session_id to avoid grouping similar problems
            project_key = f"{plan_info.get('session_id')}_{plan_info.get('problem_statement', 'Unknown Task')[:50]}"
            projects[project_key] = {
                'problem_statement': plan_info.get('problem_statement', 'Unknown Task'),
                'planner': plan_info,
                'executors': [],
                'created_at': plan_info.get('created_at'),
                'repo_dir': plan_info.get('repo_dir'),
                'preset': plan_info.get('preset'),
                'overall_status': 'inactive',
                'session_id': plan_info.get('session_id')
            }

            # Update overall status
            if plan_info.get('status') == 'active':
                projects[project_key]['overall_status'] = 'planning'
                sessions['total_active'] += 1

        # Process executor sessions and link to planners
        for exec_path, exec_info in executor_sessions.items():
            plan_file = exec_info.get('plan_file')
            linked = False

            if plan_file:
                # Find matching planner session by session_dir match
                plan_dir = Path(plan_file).parent
                for project_key, project in projects.items():
                    if project.get('planner') and str(plan_dir) == project['planner']['session_dir']:
                        project['executors'].append(exec_info)
                        linked = True

                        # Update overall status
                        if exec_info.get('status') == 'active':
                            project['overall_status'] = 'executing'
                            sessions['total_active'] += 1
                        break

            if not linked:
                # Orphaned executor (planner might be deleted)
                project_key = f"Executor: {Path(exec_path).name}"
                if project_key not in projects:
                    projects[project_key] = {
                        'problem_statement': 'Orphaned Executor Session',
                        'planner': None,
                        'executors': [],
                        'created_at': exec_info.get('created_at'),
                        'repo_dir': exec_info.get('repo_dir'),
                        'preset': exec_info.get('preset'),
                        'overall_status': 'orphaned'
                    }
                projects[project_key]['executors'].append(exec_info)

                if exec_info.get('status') == 'active':
                    sessions['total_active'] += 1

        # Convert to list and sort by creation time
        sessions['projects'] = sorted(
            projects.values(),
            key=lambda x: x.get('created_at', ''),
            reverse=True
        )
        sessions['total_sessions'] = len(planner_sessions) + len(executor_sessions)

        return sessions

    def get_planner_info(self, session_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract planner session info."""
        state_file = session_dir / '.planner-state.json'
        if not state_file.exists():
            return None

        with open(state_file, 'r') as f:
            state = json.load(f)

        # Determine if active
        is_active = False
        current_phase = None
        for phase_name, phase_data in state.get('phases', {}).items():
            if phase_data.get('status') == 'in_progress':
                is_active = True
                current_phase = phase_name
                break

        # Check if process is still running
        pid = state.get('driver_pid', 0)
        if pid and is_active:
            try:
                os.kill(pid, 0)  # Check if process exists
            except OSError:
                is_active = False

        return {
            'session_id': session_dir.name,
            'session_dir': str(session_dir),
            'type': 'planner',
            'status': 'active' if is_active else 'completed',
            'current_phase': current_phase,
            'phases': state.get('phases', {}),
            'problem_statement': state.get('problem_statement'),
            'preset': state.get('preset'),
            'repo_dir': state.get('repo_dir'),
            'fast_mode': state.get('fast_mode'),
            'created_at': state.get('created_at'),
            'updated_at': state.get('updated_at'),
            'pid': pid
        }

    def get_executor_info(self, session_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract executor session info."""
        state_file = session_dir / 'agent-state.json'
        if not state_file.exists():
            return None

        with open(state_file, 'r') as f:
            state = json.load(f)

        # Determine if active
        is_active = False
        current_step = state.get('current_step')
        if current_step:
            step_status = state.get('steps', {}).get(current_step, {}).get('status')
            if step_status in ['in_progress', 'running']:
                is_active = True

        # Check if process is still running
        pid = state.get('driver_pid', 0)
        if pid and is_active:
            try:
                os.kill(pid, 0)  # Check if process exists
            except OSError:
                is_active = False

        return {
            'session_id': session_dir.name,
            'session_dir': str(session_dir),
            'type': 'executor',
            'status': 'active' if is_active else 'completed',
            'current_step': current_step,
            'steps': state.get('steps', {}),
            'plan_file': state.get('plan_file'),
            'preset': state.get('preset'),
            'repo_dir': state.get('repo_dir') or state.get('worktree_path'),
            'is_worktree': state.get('is_worktree'),
            'created_at': state.get('created_at'),
            'updated_at': state.get('updated_at'),
            'pid': pid
        }

    def get_session_detail(self, session_id: str) -> Dict[str, Any]:
        """Get detailed info for a specific session including activity logs."""
        # Check planner sessions
        planner_path = PLANNER_DIR / session_id
        if planner_path.exists():
            return self.get_planner_detail(planner_path)

        # Check executor sessions
        executor_path = EXECUTOR_DIR / session_id
        if executor_path.exists():
            return self.get_executor_detail(executor_path)

        return {'error': f'Session {session_id} not found'}

    def get_planner_detail(self, session_dir: Path) -> Dict[str, Any]:
        """Get detailed planner session info."""
        info = self.get_planner_info(session_dir)
        if not info:
            return {'error': 'Session state not found'}

        # Add activity log
        activity_file = session_dir / 'planner-activity.log'
        if activity_file.exists():
            with open(activity_file, 'r') as f:
                lines = f.readlines()
                info['recent_activity'] = [line.strip() for line in lines[-50:]]

        # Add document info
        documents = []
        for doc in ['design-a.md', 'design-b.md', 'critique-a.md', 'critique-b.md',
                   'refined-a.md', 'refined-b.md', 'final-plan.md']:
            doc_path = session_dir / doc
            if doc_path.exists():
                size = doc_path.stat().st_size
                documents.append({
                    'name': doc,
                    'exists': True,
                    'size': f"{size / 1024:.1f}K"
                })
            else:
                documents.append({
                    'name': doc,
                    'exists': False
                })
        info['documents'] = documents
        info['session_type'] = 'planner'

        return info

    def get_executor_detail(self, session_dir: Path) -> Dict[str, Any]:
        """Get detailed executor session info."""
        info = self.get_executor_info(session_dir)
        if not info:
            return {'error': 'Session state not found'}

        # Add activity log
        activity_file = session_dir / 'pipeline-activity.log'
        if activity_file.exists():
            with open(activity_file, 'r') as f:
                lines = f.readlines()
                info['recent_activity'] = [line.strip() for line in lines[-50:]]

        # Parse pipeline status if available
        status_file = session_dir / 'pipeline-status.md'
        if status_file.exists():
            with open(status_file, 'r') as f:
                content = f.read()
                # Parse the table
                phases = {}
                for line in content.split('\n'):
                    if line.startswith('|') and '|' in line[1:]:
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 4 and parts[1] not in ['Step', '------', '']:
                            step_name = parts[1]
                            status = parts[2].lower() if parts[2] != '—' else 'pending'
                            duration = parts[3] if parts[3] != '—' else None

                            phases[step_name] = {
                                'status': status,
                                'duration': duration
                            }
                info['phases'] = phases

        info['session_type'] = 'executor'
        info['documents'] = []

        return info

    def get_legacy_status(self) -> Dict[str, Any]:
        """Legacy method - returns most recent session."""
        # Find most recent session
        planner_sessions = sorted(PLANNER_DIR.glob('*'), key=os.path.getmtime, reverse=True) if PLANNER_DIR.exists() else []
        executor_sessions = sorted(EXECUTOR_DIR.glob('*'), key=os.path.getmtime, reverse=True) if EXECUTOR_DIR.exists() else []

        latest_planner = planner_sessions[0] if planner_sessions else None
        latest_executor = executor_sessions[0] if executor_sessions else None

        if latest_executor and latest_planner:
            if os.path.getmtime(latest_executor) > os.path.getmtime(latest_planner):
                return self.get_executor_detail(latest_executor)
            else:
                return self.get_planner_detail(latest_planner)
        elif latest_executor:
            return self.get_executor_detail(latest_executor)
        elif latest_planner:
            return self.get_planner_detail(latest_planner)
        else:
            return {'error': 'No active sessions found'}

    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_server(port=8765):
    server_address = ('', port)
    httpd = HTTPServer(server_address, EnhancedSessionHandler)
    print(f"🚀 Forge Enhanced Session Server running on http://localhost:{port}")
    print(f"📊 Endpoints:")
    print(f"   /sessions - All sessions overview")
    print(f"   /session/<id> - Specific session detail")
    print(f"   /project?planner=<id> - Project with linked sessions")
    print(f"   DELETE /session/<id> - Delete session")
    print(f"Press Ctrl+C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")

if __name__ == '__main__':
    port = int(os.environ.get('DASHBOARD_PORT', '8765'))
    run_server(port)
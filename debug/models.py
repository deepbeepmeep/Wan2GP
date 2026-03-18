"""Data models for debug tool."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class TaskInfo:
    """Complete task information."""
    task_id: str
    state: Optional[Dict[str, Any]]
    logs: List[Dict[str, Any]]
    child_task_info: Optional['TaskInfo'] = None  # For cascaded failures
    child_tasks: Optional[List[Dict[str, Any]]] = None  # All child tasks for orchestrators
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'task_id': self.task_id,
            'state': self.state,
            'logs': self.logs
        }
        if self.child_task_info:
            result['child_task_info'] = self.child_task_info.to_dict()
        if self.child_tasks:
            result['child_tasks'] = self.child_tasks
        return result


@dataclass
class WorkerInfo:
    """Complete worker information."""
    worker_id: str
    state: Optional[Dict[str, Any]]
    logs: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'worker_id': self.worker_id,
            'state': self.state,
            'logs': self.logs,
            'tasks': self.tasks
        }


@dataclass
class WorkerDiskStatus:
    """Result of an attempted worker disk-space check."""

    available: bool
    runpod_id: str | None = None
    disk_info: str | None = None
    issues: List[str] | None = None
    error: str | None = None

    def get(self, key: str, default=None):
        return getattr(self, key, default)


@dataclass
class TasksSummary:
    """Summary of multiple tasks."""
    tasks: List[Dict[str, Any]]
    total_count: int
    status_distribution: Dict[str, int]
    task_type_distribution: Dict[str, int]
    timing_stats: Dict[str, Any]
    recent_failures: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tasks': self.tasks,
            'total_count': self.total_count,
            'status_distribution': self.status_distribution,
            'task_type_distribution': self.task_type_distribution,
            'timing_stats': self.timing_stats,
            'recent_failures': self.recent_failures
        }


@dataclass
class WorkersSummary:
    """Summary of multiple workers."""
    workers: List[Dict[str, Any]]
    total_count: int
    status_counts: Dict[str, int]
    active_healthy: int
    active_stale: int
    recent_failures: List[Dict[str, Any]]
    failure_rate: Optional[float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'workers': self.workers,
            'total_count': self.total_count,
            'status_counts': self.status_counts,
            'active_healthy': self.active_healthy,
            'active_stale': self.active_stale,
            'recent_failures': self.recent_failures,
            'failure_rate': self.failure_rate
        }


@dataclass
class SystemHealth:
    """System health information."""
    timestamp: datetime
    workers_active: int
    workers_spawning: int
    workers_healthy: int
    tasks_queued: int
    tasks_in_progress: int
    recent_errors: List[Dict[str, Any]]
    failure_rate: Optional[float]
    failure_rate_status: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'workers_active': self.workers_active,
            'workers_spawning': self.workers_spawning,
            'workers_healthy': self.workers_healthy,
            'tasks_queued': self.tasks_queued,
            'tasks_in_progress': self.tasks_in_progress,
            'recent_errors': self.recent_errors,
            'failure_rate': self.failure_rate,
            'failure_rate_status': self.failure_rate_status
        }


@dataclass
class OrchestratorStatus:
    """Orchestrator status information."""
    last_activity: Optional[datetime]
    last_cycle: Optional[int]
    status: str
    recent_cycles: List[Dict[str, Any]]
    recent_logs: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'last_cycle': self.last_cycle,
            'status': self.status,
            'recent_cycles': self.recent_cycles,
            'recent_logs': self.recent_logs
        }









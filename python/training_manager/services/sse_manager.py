"""
SSE (Server-Sent Events) Manager for real-time progress streaming.
"""

import asyncio
import json
from typing import Dict, Set, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SSEMessage:
    event: str
    data: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class SSEManager:
    """
    Manages SSE connections and message broadcasting.
    Supports multiple session types (simulation, training, benchmark, metrics).
    """

    def __init__(self):
        self._connections: Dict[str, Set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, session_id: str) -> asyncio.Queue:
        """Create a new subscription for a session."""
        async with self._lock:
            if session_id not in self._connections:
                self._connections[session_id] = set()

            queue = asyncio.Queue()
            self._connections[session_id].add(queue)
            return queue

    async def unsubscribe(self, session_id: str, queue: asyncio.Queue):
        """Remove a subscription."""
        async with self._lock:
            if session_id in self._connections:
                self._connections[session_id].discard(queue)
                if not self._connections[session_id]:
                    del self._connections[session_id]

    async def emit(self, session_id: str, event: str, data: Dict[str, Any]):
        """Emit an event to all subscribers of a session."""
        async with self._lock:
            if session_id in self._connections:
                message = SSEMessage(event=event, data=data)
                for queue in self._connections[session_id]:
                    try:
                        await queue.put(message)
                    except asyncio.QueueFull:
                        pass  # Skip if queue is full

    async def broadcast(self, event: str, data: Dict[str, Any]):
        """Broadcast an event to all sessions."""
        async with self._lock:
            message = SSEMessage(event=event, data=data)
            for session_id, queues in self._connections.items():
                for queue in queues:
                    try:
                        await queue.put(message)
                    except asyncio.QueueFull:
                        pass

    def get_active_sessions(self) -> list:
        """Get list of active session IDs."""
        return list(self._connections.keys())

    def get_subscriber_count(self, session_id: str) -> int:
        """Get number of subscribers for a session."""
        return len(self._connections.get(session_id, set()))


# Global SSE manager instance
sse_manager = SSEManager()


async def stream_generator(
    session_id: str,
    sse_manager: SSEManager,
    request,
    keepalive_interval: int = 30
):
    """
    Generator function for SSE responses.

    Args:
        session_id: The session to subscribe to
        sse_manager: SSEManager instance
        request: FastAPI Request object for disconnect detection
        keepalive_interval: Seconds between keepalive pings
    """
    queue = await sse_manager.subscribe(session_id)

    try:
        while True:
            if await request.is_disconnected():
                break

            try:
                message = await asyncio.wait_for(
                    queue.get(),
                    timeout=keepalive_interval
                )
                yield {
                    "event": message.event,
                    "data": json.dumps(message.data)
                }
            except asyncio.TimeoutError:
                # Send keepalive
                yield {
                    "event": "ping",
                    "data": json.dumps({"timestamp": datetime.utcnow().isoformat()})
                }
    finally:
        await sse_manager.unsubscribe(session_id, queue)

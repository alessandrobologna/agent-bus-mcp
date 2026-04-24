from __future__ import annotations

import sys

from agent_bus.db import AgentBusDB


def send(db: AgentBusDB, *, topic_id: str, sender: str, content: str) -> None:
    db.sync_once(
        topic_id=topic_id,
        agent_name=sender,
        outbox=[
            {
                "content_markdown": content,
                "message_type": "message",
                "reply_to": None,
                "metadata": None,
                "client_message_id": None,
            }
        ],
        max_items=20,
        include_self=True,
        auto_advance=True,
        ack_through=None,
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: seed_web_ui_db.py <db-path>")

    db = AgentBusDB(path=sys.argv[1])
    alpha = db.topic_create(name="Alpha review", metadata=None, mode="new")
    beta = db.topic_create(name="Beta thread", metadata=None, mode="new")

    send(db, topic_id=alpha.topic_id, sender="reviewer", content="hello from alpha")
    for index in range(1, 21):
        sender = "reviewer" if index % 2 == 0 else "architect"
        send(
            db,
            topic_id=alpha.topic_id,
            sender=sender,
            content=f"alpha checkpoint {index}: thread-map fixture message with enough text to wrap in the desktop topic view.",
        )
    send(db, topic_id=beta.topic_id, sender="architect", content="beta handoff summary")


if __name__ == "__main__":
    main()

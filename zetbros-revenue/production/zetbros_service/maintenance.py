"""Operator-owned backup/restore. Restores always enter irreversible send quarantine."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
import uuid
from contextlib import closing
from pathlib import Path

from .auth import Principal
from .config import load_settings
from .models import ReplyAction, canonical
from .store import Store, StoreError


def backup(store: Store, destination: Path):
    if destination.exists() or not destination.parent.is_dir():
        raise StoreError("backup_destination_must_be_new",400)
    started=time.monotonic()
    def bounded_progress(status, remaining, total):
        if time.monotonic()-started > 30:
            raise StoreError("backup_timeout",503)
    with store.connection() as source, closing(sqlite3.connect(destination)) as target:
        source.backup(target,pages=64,progress=bounded_progress,sleep=0.01)
        if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise StoreError("backup_integrity_failure",503)
    return {"backup":"complete","file":str(destination),"includes":"proposals_decisions_execution_ledger_audit_identity"}


def fsync_directory(path: Path):
    fd=os.open(path,os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def restore(settings, source_path: Path):
    destination=Path(settings.database_path)
    if not source_path.is_file() or source_path.is_symlink() or destination.exists() or not destination.parent.is_dir():
        raise StoreError("restore_requires_backup_and_new_database_path",400)
    # Exclusive, private, same-filesystem candidate. Never stage a runnable old
    # ledger at the configured final path before quarantine has been committed.
    fd,name=tempfile.mkstemp(prefix=".zetbros-restore-",suffix=".sqlite3",dir=destination.parent)
    os.close(fd)
    candidate=Path(name)
    published=False
    try:
        with closing(sqlite3.connect(f"file:{source_path}?mode=ro",uri=True)) as source, closing(sqlite3.connect(candidate)) as target:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise StoreError("restore_integrity_failure",503)
            identity=target.execute("SELECT value FROM metadata WHERE key='deployment_identity'").fetchone()
            if not identity or identity[0] != settings.identity_digest:
                raise StoreError("restore_identity_mismatch",409)
        staged=Store(settings.model_copy(update={"database_path":str(candidate)}))
        with staged.transaction() as conn:
            conn.execute("UPDATE metadata SET value='true' WHERE key='execution_quarantine'")
            rows=conn.execute("SELECT e.proposal_id,e.digest,e.state,p.payload FROM executions e JOIN proposals p ON p.id=e.proposal_id WHERE e.state IN ('claimed','queued')").fetchall()
            for row in rows:
                now=int(time.time()); action=ReplyAction.model_validate_json(row["payload"])
                state="uncertain" if row["state"]=="claimed" else "blocked"
                result={"submission":"uncertain" if state=="uncertain" else "not_attempted","sent_copy":"unknown" if state=="uncertain" else "not_attempted","delivery":"unverified","error_class":"restore_quarantine"}
                conn.execute("UPDATE executions SET state=?,completed_at=?,result=? WHERE proposal_id=?",(state,now,canonical(result),row["proposal_id"]))
                staged.event(conn,str(uuid.uuid4()),Principal("restore-operator","maintenance-cli","operator"),row["proposal_id"],row["digest"],action.policy_version,"uncertainty" if state=="uncertain" else "result",state,"restore_quarantine")
        with staged.connection() as conn:
            if conn.execute("SELECT value FROM metadata WHERE key='execution_quarantine'").fetchone()[0] != "true":
                raise StoreError("restore_quarantine_missing",503)
            if conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] != 0:
                raise StoreError("restore_checkpoint_failed",503)
            # Produce a self-contained file, not an unpublished file plus WAL.
            if conn.execute("PRAGMA journal_mode=DELETE").fetchone()[0] != "delete":
                raise StoreError("restore_checkpoint_failed",503)
        with candidate.open("rb") as handle: os.fsync(handle.fileno())
        fsync_directory(destination.parent)
        # Same-directory hard link is atomic and refuses an existing destination.
        # os.replace would overwrite a DB created during the preparation window.
        os.link(candidate,destination)
        published=True
        fsync_directory(destination.parent)
        return {"restore":"complete","execution_quarantine":True,"release":"requires_new_reviewed_release_and_ledger_reconciliation"}
    finally:
        # An abrupt process exit may leave a private hidden candidate, never an
        # unquarantined final database. The service does not discover candidates.
        if candidate.exists(): candidate.unlink()
        for suffix in ("-wal","-shm","-journal"):
            leftover=Path(str(candidate)+suffix)
            if leftover.exists(): leftover.unlink()
        if published: fsync_directory(destination.parent)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("check")
    sub.add_parser("init",help="explicit initialization for a genuinely new deployment only")
    backup_parser=sub.add_parser("backup"); backup_parser.add_argument("--destination",type=Path,required=True)
    restore_parser=sub.add_parser("restore"); restore_parser.add_argument("--source",type=Path,required=True)
    args=parser.parse_args()
    settings=load_settings()
    if args.command=="restore": result=restore(settings,args.source)
    elif args.command=="init":
        store=Store.initialize(settings)
        result={"initialization":"complete","new_deployment_only":True,"database_healthy":store.healthy()}
    else:
        store=Store(settings)
        result=backup(store,args.destination) if args.command=="backup" else {"database_healthy":store.healthy(),"schema_version":1,"deployment_profile":settings.deployment_profile}
    print(json.dumps(result,sort_keys=True))


if __name__ == "__main__":
    main()

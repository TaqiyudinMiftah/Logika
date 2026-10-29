# Laptop shutdown readiness

Training, validation and terminal audits run as persistent systemd user services on node2001/2003. Both users have Linger=yes. No training inputs, credentials or workers depend on a laptop SSH connection. Original model clock and scientific source are unchanged.

The remote service logika-fourhour-offline-20261029.service performs health checks, a hash-verified second copy under node2003:/mnt/home/faris/logika-fourhour-20261029-backup-2001, sanitized publication outbox, and terminal collection. It uses the original clock ceiling. Current receipts are LAPTOP_INDEPENDENCE.json and private/laptop-independent/{BACKUP_RECEIPT,OUTBOX_RECEIPT,FINAL_REMOTE_RECEIPT}.json. Health history is private/laptop-independent/HEALTH_HISTORY.jsonl.

Actual GitHub pushes and chat notifications require the laptop login/app to be online. They resume on reconnect. Sanitized metadata/source are safely queued at private/laptop-independent/PUBLICATION_OUTBOX with a SHA256 manifest and zero-finding secret scan. No GitHub credentials were copied to shared lab nodes.

After reconnect, first read canonical state and receipts; never relaunch a completed campaign. Run existing workstation publisher for collection/publication only. Source/PLAN/CLOCK remain sealed. No extra fits, refits or Kaggle submissions are authorized by reconnection.

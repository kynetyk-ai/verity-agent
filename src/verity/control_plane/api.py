"""The control-plane API surface — async (spec §3.4).

Configure-by-task; proposal intake (shape validation + object harvest from the outbox,
before sandbox teardown); verifier dispatch (the only path to the verifier); cycle control
(serve context; orchestration policy decides run-again / stop); extraction (final outputs,
context files, provenance/audit + rejected/superseded/revised logs).

TODO (Phase 1.4): the async service boundary + endpoints over the control-plane internals.
"""

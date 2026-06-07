"""Per-task configuration, the workspace contract, and the composed prompt (spec §3.4).

Config surface: task instructions, proposal-shape spec, data sources, contextual info,
and the §8 registrations. The invariant workspace contract (fixed roles/paths: read-only
``data``/``context``/``tools``/``spec``; writable-ephemeral ``scratch``/``outbox``). The
composed 3-layer system prompt (kernel orientation / domain / task), assembled
mechanically.

TODO (Phase 1.2): task-config model, workspace-contract provisioning, prompt composition.
"""

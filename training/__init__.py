"""Training orchestration.

Owns a training *run*: load data, validate, build features, fit both models,
evaluate, persist artifacts, register the version. Training never happens in the
request path -- ``POST /api/v1/models/train`` schedules a run and returns 202.

Phases 5 and 6 add the individual trainers; the end-to-end pipeline lands with them.
"""

__all__: list = []

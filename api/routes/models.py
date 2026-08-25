"""Model registry endpoints: what is trained, what is serving, retrain.

``POST /models/train`` runs the training pipeline **synchronously**. That is a
deliberate choice for a system of this size, and worth defending: training takes
tens of seconds on this dataset, the caller almost always wants to know whether
it worked, and a background job would need a task queue, a status endpoint and a
result store to tell them. The docstring says how long it takes, and the honest
answer for a hundred-thousand-row dataset is "long enough to wait for".

At real scale this becomes an enqueue that returns 202 with a job id. The
:class:`~training.pipeline.TrainingResult` the pipeline already returns is
exactly what such a job would store, so that change is a wrapper rather than a
rewrite.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.dependencies import get_registry_dependency, session_dependency, settings_dependency
from api.security import require_write
from api.schemas import ModelsResponse, ModelVersionSchema, TrainRequest, TrainResponse
from config import Settings
from features.feature_engineering import FEATURE_VERSION
from models.model_registry import ModelRegistry
from monitoring.logging_config import get_logger
from training.pipeline import TrainingFailed, TrainingPipeline

logger = get_logger(__name__)

router = APIRouter(tags=["models"])


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="Trained model versions",
    description=(
        "Every version present in the artifact directory, newest first, with "
        "its holdout metrics and provenance. `errors` reports anything that "
        "failed to load -- most importantly a feature-contract mismatch, which "
        "means the running code no longer produces the features a model was "
        "trained on."
    ),
)
def list_models(
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> ModelsResponse:
    """Return the model catalogue and what is currently serving."""
    loaded = registry.loaded
    return ModelsResponse(
        active_version=loaded.version,
        available=loaded.available,
        loaded_at=loaded.loaded_at.isoformat() if loaded.loaded_at else None,
        errors=loaded.errors,
        feature_version=FEATURE_VERSION,
        versions=[ModelVersionSchema(**row) for row in registry.catalogue()],
    )


@router.post(
    "/models/train",
    response_model=TrainResponse,
    summary="Retrain the demand models",
    description=(
        "Runs the full training pipeline and writes a new versioned artifact "
        "set: chronological split, Gradient Boosting, Prophet, evaluation "
        "against a predict-the-mean baseline, then a JSON training report.\n\n"
        "**This is synchronous and takes tens of seconds.** Set "
        "`backtest_folds` above zero to add Prophet's rolling-origin backtest, "
        "which roughly doubles the time.\n\n"
        "Returns 409 when there is not enough labelled data to train on."
    ),
    responses={
        409: {"description": "Not enough data, or no model could be trained"},
    },
)
def train_models(
    body: TrainRequest,
    session: Session = Depends(session_dependency),
    registry: ModelRegistry = Depends(get_registry_dependency),
    settings: Settings = Depends(settings_dependency),
    _scope: str = Depends(require_write),
) -> TrainResponse:
    """Train both models and, by default, serve the result immediately."""
    pipeline = TrainingPipeline(settings)

    try:
        result = pipeline.run(
            session,
            test_days=body.test_days,
            backtest_folds=body.backtest_folds,
            train_prophet=body.train_prophet,
            train_gbr=body.train_gradient_boosting,
        )
    except TrainingFailed as exc:
        # 409 rather than 500: the request is well-formed, the *state* is wrong,
        # and the fix is to seed and build features rather than to debug.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    reloaded = False
    if body.reload_after and result.succeeded:
        registry.reload(result.version)
        reloaded = registry.loaded.version == result.version

    logger.info(
        "Training via API produced %s (reloaded=%s)", result.version, reloaded
    )

    return TrainResponse(
        version=result.version,
        succeeded=result.succeeded,
        duration_seconds=round(result.duration_seconds, 2),
        n_train=result.n_train,
        n_test=result.n_test,
        steps=[step.as_dict() for step in result.steps],
        artifacts=result.artifacts,
        reloaded=reloaded,
        summary=result.summary(),
    )


__all__ = ["router"]

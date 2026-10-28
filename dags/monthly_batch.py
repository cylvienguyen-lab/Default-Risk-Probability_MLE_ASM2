"""Score the previous application month and refresh eligible monitoring."""

import os
from datetime import datetime, timedelta, timezone

from airflow.sdk import TaskGroup, dag, get_current_context, task

from dags import workflow


CONFIG_PATH = os.environ.get("PROJECT_CONFIG", "/opt/airflow/project/configs/project_config.json")


def _run_id(scope: dict, name: str) -> str:
    context = get_current_context()
    instance = context["ti"]
    return workflow.task_run_id(scope["run_id"], name, instance.try_number,
                                instance.map_index)


@dag(dag_id="monthly_batch", start_date=datetime(2024, 8, 1, tzinfo=timezone.utc),
     schedule="0 0 1 * *", catchup=False, max_active_runs=1,
     max_active_tasks=2, is_paused_upon_creation=True,
     tags=["default-prob", "monthly"])
def monthly_batch():
    # 1. Resolve the prior application month, explicit as-of and released model.
    @task
    def resolve_scope() -> dict:
        context = get_current_context()
        conf = context["dag_run"].conf or {}
        run_type = getattr(context["dag_run"].run_type, "value",
                           str(context["dag_run"].run_type)).lower()
        if run_type.endswith("manual") and not conf.get("monitoring_as_of"):
            raise ValueError("Manual monthly runs require monitoring_as_of in DagRun conf")
        interval_end = (conf.get("monitoring_as_of") or
                        context["data_interval_end"].date().isoformat())
        return workflow.monthly_scope(
            conf.get("config_path", CONFIG_PATH), context["dag_run"].run_id,
            interval_end, conf.get("application_month"),
            conf.get("monitoring_as_of"), conf.get("model_version"))

    @task(execution_timeout=timedelta(minutes=5))
    def resolve_released_model(scope: dict) -> dict:
        return workflow.released_model(scope["config_path"], scope["model_version"])

    # 2. Build the new feature cohort and the eligible outcome snapshot.
    @task(retries=1, retry_delay=timedelta(minutes=2),
          execution_timeout=timedelta(hours=2), max_active_tis_per_dag=1)
    def build_and_validate_features(scope: dict) -> dict:
        return workflow.build_feature(scope["application_month"],
                                      scope["config_path"],
                                      _run_id(scope, "feature"))

    @task(retries=1, retry_delay=timedelta(minutes=2),
          execution_timeout=timedelta(hours=2), max_active_tis_per_dag=1)
    def update_eligible_labels(scope: dict) -> dict:
        if scope["label_snapshot"] is None:
            return {"status": "not_eligible", "reason":
                    "No inference cohort reaches MOB-6 plus reporting lag"}
        return workflow.build_label(scope["label_snapshot"],
                                    scope["config_path"],
                                    _run_id(scope, "label"))

    # 3. Score with the frozen release and publish the checked predictions.
    @task(retries=1, retry_delay=timedelta(minutes=2),
          execution_timeout=timedelta(minutes=30))
    def score_month(scope: dict, features: dict, model: dict) -> dict:
        if features["status"] != "published":
            raise ValueError("Feature cohort is not published")
        return workflow.score(scope["application_month"], scope["config_path"],
                              _run_id(scope, "score"), model)

    @task(execution_timeout=timedelta(minutes=15))
    def publish_predictions(scope: dict, staging: dict) -> dict:
        return workflow.publish_prediction(staging, scope["config_path"])

    @task(execution_timeout=timedelta(minutes=10))
    def resolve_reference(scope: dict, model: dict) -> dict:
        return workflow.ensure_reference(scope["config_path"],
                                         _run_id(scope, "reference"), model)

    # 4. Monitor the new pending cohort and one newly mature older cohort.
    @task
    def validate_monitoring_inputs(scope: dict, prediction: dict,
                                   labels: dict, model: dict) -> list[str]:
        return workflow.ready_cohorts(scope["monitor_cohorts"], [prediction], labels,
                                      scope["config_path"], model["model_version"],
                                      [scope["label_snapshot"]] if scope["label_snapshot"] else [])

    @task(retries=1, retry_delay=timedelta(minutes=2),
          execution_timeout=timedelta(minutes=30), max_active_tis_per_dag=1)
    def compute_monitoring(application_month: str, scope: dict, model: dict,
                           reference: dict) -> dict:
        return workflow.compute_monitoring(application_month,
                                           scope["monitoring_as_of"],
                                           scope["config_path"],
                                           _run_id(scope, "monitor"), model, reference)

    @task(execution_timeout=timedelta(minutes=15), max_active_tis_per_dag=1)
    def publish_metrics(staging: dict, scope: dict) -> dict:
        return workflow.publish_monitoring(staging, scope["config_path"])

    @task(execution_timeout=timedelta(minutes=20))
    def publish_charts(scope: dict, model: dict, reference: dict,
                       metrics: list[dict]) -> dict:
        if len(metrics) != len(scope["monitor_cohorts"]):
            raise ValueError("Monthly monitoring publication count differs from scope")
        return workflow.report(scope["config_path"], _run_id(scope, "charts"),
                               scope["monitoring_as_of"], model, reference)

    scope = resolve_scope()
    model = resolve_released_model(scope)
    with TaskGroup(group_id="data_pipeline"):
        features = build_and_validate_features(scope)
        labels = update_eligible_labels(scope)
    with TaskGroup(group_id="inference"):
        staged_prediction = score_month(scope, features, model)
        prediction = publish_predictions(scope, staged_prediction)
    with TaskGroup(group_id="monitoring"):
        reference = resolve_reference(scope, model)
        cohorts = validate_monitoring_inputs(scope, prediction, labels, model)
        staged = compute_monitoring.partial(scope=scope, model=model,
                                            reference=reference).expand(
            application_month=cohorts)
        metrics = publish_metrics.partial(scope=scope).expand(staging=staged)
        publish_charts(scope, model, reference, metrics)


monthly_batch()

"""Build the historical stores, release one model, then publish monitoring."""

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


@dag(dag_id="bootstrap_e2e", start_date=datetime(2024, 7, 1, tzinfo=timezone.utc),
     schedule=None, catchup=False, max_active_runs=1, max_active_tasks=2,
     is_paused_upon_creation=True, tags=["default-prob", "bootstrap"])
def bootstrap_e2e():
    # 1. Resolve historical scope without reading data during DAG parsing.
    @task
    def resolve_scope() -> dict:
        context = get_current_context()
        conf = context["dag_run"].conf or {}
        return workflow.bootstrap_scope(conf.get("config_path", CONFIG_PATH),
                                        context["dag_run"].run_id,
                                        conf.get("monitoring_as_of"))

    @task
    def select_months(scope: dict, key: str) -> list[str]:
        allowed = {"feature_months", "training_label_snapshots",
                   "inference_months", "monitoring_label_snapshots"}
        if key not in allowed or not scope[key]:
            raise ValueError(f"Bootstrap scope has no valid months for {key}")
        return scope[key]

    # 2. Build both Gold inputs and validate the complete training membership.
    @task(retries=1, retry_delay=timedelta(minutes=2),
          execution_timeout=timedelta(hours=2), max_active_tis_per_dag=1)
    def build_feature(application_month: str, scope: dict) -> dict:
        return workflow.build_feature(application_month, scope["config_path"],
                                      _run_id(scope, "feature"))

    @task(retries=1, retry_delay=timedelta(minutes=2),
          execution_timeout=timedelta(hours=2), max_active_tis_per_dag=1)
    def build_training_label(snapshot_month: str, scope: dict) -> dict:
        return workflow.build_label(snapshot_month, scope["config_path"],
                                    _run_id(scope, "training_label"))

    @task(execution_timeout=timedelta(minutes=15))
    def validate_gold(scope: dict, features: list[dict],
                      labels: list[dict]) -> dict:
        if not features or not labels:
            raise ValueError("Bootstrap Gold inputs are empty")
        return workflow.validate_gold(scope["config_path"],
                                      _run_id(scope, "validate_gold"))

    # 3. Fit the frozen candidate, validate it, and release one immutable version.
    @task(execution_timeout=timedelta(hours=2))
    def train_and_evaluate(scope: dict, gold: dict) -> dict:
        if gold["status"] != "validated":
            raise ValueError("Gold validation did not pass")
        return workflow.train_candidate(scope["config_path"],
                                        _run_id(scope, "train"), scope["fit_as_of"])

    @task(execution_timeout=timedelta(minutes=15))
    def validate_and_publish_model(scope: dict, candidate: dict) -> dict:
        if candidate["status"] != "candidate":
            raise ValueError("Training candidate is not validated")
        return workflow.publish_model(candidate, scope["config_path"],
                                      _run_id(scope, "release"))

    @task
    def resolve_published_model(scope: dict, publication: dict) -> dict:
        if publication["status"] != "released":
            raise ValueError("Model release did not complete")
        return workflow.released_from_publication(publication, scope["config_path"])

    # 4. Score each inference cohort and build mature monitoring labels.
    @task(retries=1, retry_delay=timedelta(minutes=2),
          execution_timeout=timedelta(minutes=30), max_active_tis_per_dag=1)
    def score_month(application_month: str, scope: dict, model: dict) -> dict:
        return workflow.score(application_month, scope["config_path"],
                              _run_id(scope, "score"), model)

    @task(execution_timeout=timedelta(minutes=15), max_active_tis_per_dag=1)
    def publish_prediction(staging: dict, scope: dict) -> dict:
        return workflow.publish_prediction(staging, scope["config_path"])

    @task(retries=1, retry_delay=timedelta(minutes=2),
          execution_timeout=timedelta(hours=2), max_active_tis_per_dag=1)
    def build_monitoring_label(snapshot_month: str, scope: dict) -> dict:
        return workflow.build_label(snapshot_month, scope["config_path"],
                                    _run_id(scope, "monitoring_label"))

    @task(execution_timeout=timedelta(minutes=20))
    def freeze_reference(scope: dict, model: dict) -> dict:
        return workflow.ensure_reference(scope["config_path"],
                                         _run_id(scope, "reference"), model)

    # 5. Monitor only after scores, labels and the fixed reference are available.
    @task
    def validate_monitoring_inputs(scope: dict, predictions: list[dict],
                                   labels: list[dict]) -> list[str]:
        return workflow.ready_cohorts(scope["inference_months"],
                                      predictions, labels,
                                      expected_label_snapshots=scope["monitoring_label_snapshots"])

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
        if len(metrics) != len(scope["inference_months"]):
            raise ValueError("Monitoring publication count differs from inference scope")
        return workflow.report(scope["config_path"], _run_id(scope, "charts"),
                               scope["monitoring_as_of"], model, reference)

    scope = resolve_scope()
    feature_months = select_months.override(task_id="feature_months")(
        scope, "feature_months")
    training_snapshots = select_months.override(task_id="training_label_snapshots")(
        scope, "training_label_snapshots")
    inference_months = select_months.override(task_id="inference_months")(
        scope, "inference_months")
    monitoring_snapshots = select_months.override(task_id="monitoring_label_snapshots")(
        scope, "monitoring_label_snapshots")
    with TaskGroup(group_id="data_pipeline"):
        features = build_feature.partial(scope=scope).expand(
            application_month=feature_months)
        training_labels = build_training_label.partial(scope=scope).expand(
            snapshot_month=training_snapshots)
        gold = validate_gold(scope, features, training_labels)
    with TaskGroup(group_id="training"):
        candidate = train_and_evaluate(scope, gold)
        publication = validate_and_publish_model(scope, candidate)
        model = resolve_published_model(scope, publication)
    with TaskGroup(group_id="inference"):
        scored = score_month.partial(scope=scope, model=model).expand(
            application_month=inference_months)
        predictions = publish_prediction.partial(scope=scope).expand(staging=scored)
    with TaskGroup(group_id="monitoring"):
        monitoring_labels = build_monitoring_label.partial(scope=scope).expand(
            snapshot_month=monitoring_snapshots)
        publication >> monitoring_labels
        reference = freeze_reference(scope, model)
        cohorts = validate_monitoring_inputs(scope, predictions, monitoring_labels)
        staged = compute_monitoring.partial(scope=scope, model=model,
                                            reference=reference).expand(
            application_month=cohorts)
        metrics = publish_metrics.partial(scope=scope).expand(staging=staged)
        publish_charts(scope, model, reference, metrics)


bootstrap_e2e()

"""Write run metadata and publish one local partition safely."""

import hashlib
import json
import os
import shutil
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash the logical files of one Parquet partition directory."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file()
                   and not item.name.startswith(".") and item.name != "_SUCCESS")
    if not files:
        raise ValueError(f"Partition has no data files: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(sha256(item).encode("ascii"))
    return digest.hexdigest()


def write_manifest(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def run_record(config: dict, run_id: str, job_name: str, scope: str,
               status: str, **details: object) -> Path:
    from pipelines.utils.config import location

    safe_run_id = "".join(c for c in run_id if c.isalnum() or c in "_-" )
    if not safe_run_id or safe_run_id != run_id:
        raise ValueError("run_id must contain only letters, digits, _ or -")
    path = location(config, "runs") / (
        f"{safe_run_id}_{job_name}_{scope}_{uuid.uuid4().hex[:8]}.json")
    config_dir = config["config_path"].parent
    config_files = {"project": config["config_path"],
                    "feature": config_dir / config["project"]["feature_config"],
                    "model": config_dir / config["project"]["model_config"]}
    if "monitoring_config" in config["project"]:
        config_files["monitoring"] = config_dir / config["project"]["monitoring_config"]
    config_refs = {name: {"path": str(value), "sha256": sha256(value)}
                   for name, value in config_files.items()}
    code_refs = {path.relative_to(config["root"]).as_posix(): sha256(path)
                 for path in sorted((config["root"] / "pipelines").rglob("*.py"))}
    snapshot = location(config, "runs") / "snapshots" / f"{safe_run_id}.zip"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        temporary_snapshot = snapshot.with_name(f".{snapshot.name}.{uuid.uuid4().hex}.tmp")
        with zipfile.ZipFile(temporary_snapshot, "w", zipfile.ZIP_DEFLATED) as archive:
            for relative in [*code_refs, "main.py", "requirements.txt"]:
                archive.write(config["root"] / relative, arcname=relative)
            for role, source in config_files.items():
                archive.write(source, arcname=f"configs/{role}_config.json")
        os.replace(temporary_snapshot, snapshot)
    with zipfile.ZipFile(snapshot) as archive:
        for relative, expected in code_refs.items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                raise RuntimeError(f"Run {run_id}: code changed after its source snapshot")
    return write_manifest(path, {
        "run_id": run_id, "job_name": job_name, "scope": scope,
        "status": status, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config["config_path"]),
        "config_refs": config_refs,
        "code_refs": code_refs,
        "code_snapshot_ref": str(snapshot.relative_to(config["root"])),
        "code_snapshot_sha256": sha256(snapshot),
        "feature_schema_version": config["feature"]["schema_version"],
        **details,
    })


def previous_publication(config: dict, job_name: str, scope: str,
                         input_refs: dict) -> dict | None:
    """Describe the prior published revision and whether retained inputs changed."""
    from pipelines.utils.config import location

    candidates = sorted(location(config, "runs").glob(f"*_{job_name}_{scope}_*.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "published":
            return {"previous_manifest": str(path),
                    "input_refs_changed": record.get("input_refs") != input_refs}
    return None


def publish_partition(staging: Path, target: Path) -> Path:
    """Replace one local partition while retaining the previous one on failure."""
    if not staging.exists():
        raise FileNotFoundError(f"Staging partition absent: {staging}")
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_name(f".{target.name}.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Another publisher holds {lock_path}") from error
    os.close(lock_fd)
    backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.backup")

    def replace_with_retry(source: Path, destination: Path) -> None:
        for attempt in range(6):
            try:
                os.replace(source, destination)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.2 * (attempt + 1))

    try:
        if target.exists():
            replace_with_retry(target, backup)
        try:
            replace_with_retry(staging, target)
        except Exception:
            if backup.exists():
                replace_with_retry(backup, target)
            raise
    except Exception:
        raise
    else:
        if backup.exists():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
    finally:
        lock_path.unlink(missing_ok=True)
    return target


def write_spark_parquet_local(frame, target: Path, max_rows: int = 1000000) -> Path:
    """Collect a bounded Spark frame and retain its schema in local Parquet.

    This small-dataset writer avoids Hadoop's Windows-only winutils dependency.
    It is deliberately unsuitable for unbounded production data.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from pyspark.sql import types as T

    def arrow_type(data_type):
        if isinstance(data_type, T.StringType):
            return pa.string()
        if isinstance(data_type, T.IntegerType):
            return pa.int32()
        if isinstance(data_type, T.LongType):
            return pa.int64()
        if isinstance(data_type, T.FloatType):
            return pa.float32()
        if isinstance(data_type, T.DoubleType):
            return pa.float64()
        if isinstance(data_type, T.DateType):
            return pa.date32()
        if isinstance(data_type, T.ArrayType):
            return pa.list_(arrow_type(data_type.elementType))
        raise TypeError(f"Unsupported local Parquet field type: {data_type}")

    count = frame.count()
    if count > max_rows:
        raise ValueError(f"Local Parquet driver limit exceeded: {count} > {max_rows}")
    # Spark's writer marks these two derived quality flags nullable in A1,
    # despite their non-null expression values. Preserve that baseline schema.
    schema = pa.schema([pa.field(field.name, arrow_type(field.dataType),
                                 nullable=(field.nullable or field.name in {
                                     "unusual_discrete_value_flag",
                                     "loan_info_inconsistent_flag"}))
                        for field in frame.schema.fields])
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        table = pa.Table.from_pandas(frame.toPandas(), schema=schema,
                                     preserve_index=False, safe=True)
        pq.write_table(table, temporary / "part-00000.parquet")
        (temporary / "_SUCCESS").touch()
        publish_partition(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def read_spark_parquet_local(spark, *paths: Path):
    """Read bounded local Parquet paths without Hadoop's Windows native layer."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyspark.sql import types as T

    def spark_type(data_type):
        if pa.types.is_string(data_type):
            return T.StringType()
        if pa.types.is_int32(data_type):
            return T.IntegerType()
        if pa.types.is_int64(data_type):
            return T.LongType()
        if pa.types.is_float32(data_type):
            return T.FloatType()
        if pa.types.is_float64(data_type):
            return T.DoubleType()
        if pa.types.is_date32(data_type):
            return T.DateType()
        if pa.types.is_list(data_type):
            return T.ArrayType(spark_type(data_type.value_type))
        raise TypeError(f"Unsupported local Parquet field type: {data_type}")

    tables = [pq.read_table(path) for path in paths]
    if not tables:
        raise ValueError("No Parquet paths supplied")
    schema = T.StructType([T.StructField(field.name, spark_type(field.type),
                                         field.nullable) for field in tables[0].schema])
    if any(table.schema.names != tables[0].schema.names for table in tables[1:]):
        raise ValueError("Parquet partitions have incompatible columns")
    table = pa.concat_tables(tables)
    if table.num_rows > 1000000:
        raise ValueError("Local Parquet driver limit exceeded")
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "false")
    return spark.createDataFrame(table.to_pandas(), schema)


def release_model(candidate: Path, version: str, model_bank: Path,
                  manifest: dict) -> dict:
    """Release a checked candidate under an immutable version."""
    from pipelines.utils.validation import validate_model_bundle

    import joblib

    bundle = joblib.load(candidate)
    validate_model_bundle(bundle)
    if bundle["model_version"] != version:
        raise ValueError("Model version differs from candidate bundle")
    model_bank.mkdir(parents=True, exist_ok=True)
    lock_path = model_bank / ".release.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("Another model release holds the release lock") from error
    os.close(lock_fd)
    target_dir = model_bank / version
    created = False
    try:
        if target_dir.exists():
            raise FileExistsError(f"Released model version is immutable: {target_dir}")
        target_dir.mkdir()
        created = True
        target = target_dir / "model.joblib"
        shutil.copy2(candidate, target)
        record = {**manifest, "model_version": version,
                  "artifact_path": (Path(version) / "model.joblib").as_posix(),
                  "sha256": sha256(target)}
        manifest_path = write_manifest(target_dir / "manifest.json", record)
        pointer = {"model_version": version,
                   "artifact_path": (Path(version) / "model.joblib").as_posix(),
                   "manifest_path": (Path(version) / "manifest.json").as_posix(),
                   "sha256": record["sha256"]}
        write_manifest(model_bank / "released_model.json", pointer)
        return {**pointer, "artifact_path": str(target),
                "manifest_path": str(manifest_path)}
    except Exception:
        if created and target_dir.exists() and not (target_dir / "manifest.json").exists():
            shutil.rmtree(target_dir)
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def read_released_model(model_bank: Path) -> dict:
    pointer_path = model_bank / "released_model.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    artifact = Path(pointer["artifact_path"].replace("\\", "/"))
    if not artifact.is_absolute():
        artifact = model_bank / artifact
    if sha256(artifact) != pointer["sha256"]:
        raise ValueError(f"Released model checksum mismatch: {artifact}")
    manifest = Path(pointer["manifest_path"].replace("\\", "/"))
    if not manifest.is_absolute():
        manifest = model_bank / manifest
    return {**pointer, "artifact_path": str(artifact),
            "manifest_path": str(manifest)}

FROM apache/airflow:3.3.2-python3.12

USER root
ENV DEBIAN_FRONTEND=noninteractive JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements-airflow.txt /tmp/requirements-airflow.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" \
    -r /tmp/requirements-airflow.txt && pip check

WORKDIR /opt/airflow/project

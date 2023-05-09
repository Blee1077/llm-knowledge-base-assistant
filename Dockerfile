FROM public.ecr.aws/docker/library/python:3.8.12-slim-buster
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.7.0 /lambda-adapter /opt/extensions/lambda-adapter

ENV NUMBA_CACHE_DIR=/tmp/numba_cache
ENV PORT="8080"
ENV READINESS_CHECK_PROTOCOL="tcp"
ENV ASYNC_INIT="false"

WORKDIR /var/task
COPY demo/app.py demo/utils.py demo/requirements.txt ./s3_config.json  ./
COPY demo/doc_stores/  ./doc_stores/
RUN python3 -m pip install -r requirements.txt -t .

CMD ["python3", "app.py"]

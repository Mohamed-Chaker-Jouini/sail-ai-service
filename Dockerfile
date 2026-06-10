FROM python:3.12-slim

WORKDIR /app

# proxy args (only used at build time)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV HTTP_PROXY=$HTTP_PROXY
ENV HTTPS_PROXY=$HTTPS_PROXY
ENV NO_PROXY=$NO_PROXY

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# remove proxy from final image (important)
ENV HTTP_PROXY=
ENV HTTPS_PROXY=
ENV NO_PROXY=

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
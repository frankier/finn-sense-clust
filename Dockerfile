FROM registry.gitlab.com/frankier/finntk/full-deb:latest

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Apt-get requirements
RUN apt-get update && apt-get install -y \
    # Python/Poetry
        python3 python3.7 python3-pip curl wget \
    # Python build requirements
        python3-dev python3.7-dev build-essential libffi-dev

# Poetry
RUN set -ex && curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python3
RUN ~/.poetry/bin/poetry config virtualenvs.create false

# Fixup Python
RUN ln -sf /usr/bin/python3.7 /usr/bin/python

WORKDIR /app

# Package installation
COPY ./pyproject.toml /app/
COPY ./poetry.lock /app/
RUN pip3 install --upgrade pip==19.0.3
RUN ~/.poetry/bin/poetry install --no-interaction

# NLTK resources
RUN python -c "from nltk import download as d; d('wordnet'); d('omw'); d('punkt')"

# FinnTK resources
RUN python -m finntk.scripts.bootstrap_all

# STIFF post_install
RUN python -m stiff.scripts.post_install

# Evaluation framework setup
COPY . /app

# Set up Python path
RUN echo "/app/" > "/usr/local/lib/python3.7/dist-packages/senseclust.pth"

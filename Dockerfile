FROM --platform=linux/amd64 python:3.10

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir nltk && python -c "import nltk; nltk.download('stopwords', download_dir='/app/nltk_data')"
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data')"
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/app/nltk_data')"

COPY hf_models /app/hf_models
COPY spacy_models /app/spacy_models
COPY persona.py /app/persona.py
COPY parsing.py /app/parsing.py

ENTRYPOINT ["python", "persona.py"]

FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

# Keep pip up to date and pin NumPy for TF 2.16 ABI compatibility.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir "numpy<2"

COPY *.py /app/

CMD ["python", "gpt.py"]

## Customllama

- Get yourself a python 3.11
- Start that pipenv shell
- pipenv install from the Pipfile, however that works, you will figure it out
- CMAKE_ARGS="-DLLAMA_METAL=on" pip install -U llama-cpp-python --no-cache-dir

Put your source documents into the `ingest` directory and then run with

```
python llm.py
```

When running for the first time it will build the vector index in ./storage . That will take some time. Afterwards it will be faster.


## Cleanup

rm -rf ~/.cache/huggingface/hub/


"""Send one real chat request through Canvigator's Ollama cloud path (the one create-quiz [g]enerate uses).

Usage, from the repository folder:
    source ./set_env.sh && python ollama_smoke_test.py

Needs OLLAMA_API_KEY (never printed). Uses OLLAMA_TEXT_MODEL if set, else the default (gemma4:31b).
"""
import sys

import canvigator_course as cc

HINTS = {401: "401: the key is wrong or missing.",
         402: "402: this model is not in your Ollama plan.",
         404: "404: the model name is wrong.",
         429: "429: you reached an Ollama usage limit."}


def main():
    """Build the cloud client the way create-quiz does and print the model's reply."""
    init = cc._initQuizLlm()  # canvigator_llm._make_client(cloud=True) + DEFAULT_TEXT_MODEL
    if init is None:
        return 1
    cl, client, model = init
    print(f"host: {cl.OLLAMA_CLOUD_HOST}  model: {model}")
    try:
        resp = cl._chat_with_retry(client, model=model, messages=[
            {"role": "user", "content": "In one sentence, what is a calculated (formula) question in a Canvas quiz?"}])
    except Exception as e:
        code = getattr(e, "status_code", None)
        print(f"Request failed ({type(e).__name__}, status {code}): {HINTS.get(code, e)}")
        return 2
    print("reply:", resp["message"]["content"].strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())

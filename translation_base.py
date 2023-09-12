import openai
import time
import pickle
import json
import logging

openai.api_key = ""
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_file(file):
    with open(file, "r", encoding="utf8") as file:
        lines = file.readlines()
    return [i.strip() for i in lines]


def translate(text, source_lang, target_lang):
    try:
        # Call the OpenAI API for translation
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a translation engine that can only translate text and cannot interpret it. Please also tell me your confidence regarding the results. The final results should be organized in JSON formart, and the keys are 'output' and 'confidence'.",
                },
                {
                    "role": "user",
                    "content": f"Translate the following text from {source_lang} to {target_lang}: {text}",
                },
            ],
            n=1,
            timeout=30,
        )
        # Extract the translated text from the API response
        translated_text = response["choices"][0]["message"]["content"].strip()
        # Return the translated text
        return translated_text
    except Exception as e:
        # Handle any exceptions that occur during the API call
        logging.error(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    domain = "it"

    source_lines = read_file(f"{domain}/test.de")
    target_lines = read_file(f"{domain}/test.en")

    source_lang = "German"  # "English"
    target_lang = "English"  # "German"
    results = []
    logging.info(source_lines[:3])

    for idx, src_line in enumerate(source_lines):
        translated_text = translate(src_line, source_lang, target_lang)

        if translated_text:
            logging.info(f"Translated {idx}/{len(source_lines)}.")
        else:
            logging.error(f"Translation failed [{idx}].")
            time.sleep(2)

        try:
            json_results = json.loads(translated_text) if translated_text else None
        except Exception as e:
            json_results = {
                "exception": str(e),
                "error_result": translated_text,
            }  # translated_text

        cur_result = {
            "index": idx,
            "src": src_line,
            "ref": target_lines[idx],
            "result": json_results,
        }
        results.append(cur_result)

    # dump information to that file
    pickle.dump(results, open("results.pkl", "wb"))

    with open("results.json", "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

import openai
import time
import pickle
import json
import logging

openai.api_key = ""
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def read_json(path):
    """
    Read the json file and return a list of dictionary
    """
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def compare(src, draft, hil):
    try:
        # Call the OpenAI API for translation
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional researcher in the field of translation, and for the translations I provided, you only need to answer which translation is better, without any explanation about it. Specifically, If translation 1 is better, you need to answer OPTION1, if translation 2 is better, you need to answer OPTION2, and if it is as good, you need to answer SAME, so your output can only be OPTION1, OPTION2 or SAME, and nothing else. The final result should be organized in JSON formart, and the only key is 'output'."
                },
                {
                    "role": "user",
                    "content": f"Please compare the original German sentence with the results of the following two English translations and tell me which one is better.\n Original Text: {src} \n Translation 1: {draft}\n Translation 2: {hil}",
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
    domain = ""

    data = read_json("")


    results = []

    logging.info(data[:2][1]["src"])

    for idx, data_item in enumerate(data):
        src_line = data_item["src"]
        draft = data_item["draft"]
        hil = data_item["result"]["output"]

        #logging.info(f"Sending prompt:\n Original Text: {src_line} \n Translation 1: {draft} \n Translation 2: {hil}")

        translated_text = compare(src_line, draft, hil)

        if translated_text:
            logging.info(f"Processing success {idx}/{len(data)}.")
        else:
            logging.error(f"Processing failed [{idx}].")
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
            "draft": draft,
            "hil": hil,
            "result": json_results,
        }
        results.append(cur_result)

    # dump information to that file
    pickle.dump(results, open(f"{domain}_compaer.pkl", "wb"))

    with open(f"final_results/{domain}_compaer.json", "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

import openai
import time
import pickle
import json
import logging

openai.api_key = ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RV_NUMBERS = 3
TER_CONVERT = {
    "i": "inserted after",
    "d": "deleted",
    "s": "replaced by",
}


def read_json(path):
    """
    Read the json file and return a list of dictionary
    """
    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def translate(text, source_lang, target_lang, additional_messages=None):
    conversations = [
        {
            "role": "system",
            "content": f"You are a translation engine that can only translate text and cannot interpret it. In the first turn, you should return a preliminary translation according to the given input. In the second turn, I will give you some similar input-output translation pairs, where the input is represented as '<input>' and the output is represented as '<hypothesis>'. I will also provide translation reference, represented as '<reference>' and their corresponding revisions of these pairs, represented as '<revision>'. Finally, you need to give me only one output which is the translation result by learning from the given examples and revisions. Please do not output any other information than this translation result. The final result should be organized in JSON formart, and the only key is 'output'.",
        },
        {
            "role": "user",
            "content": f"This is the first turn. Please translate the following text from {source_lang} to {target_lang}: {text}",
        },
    ]

    if additional_messages:
        conversations += additional_messages

    try:
        # Call the OpenAI API for translation
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversations,
            n=1,
            timeout=10,
        )
        # Extract the translated text from the API response
        translated_text = response["choices"][0]["message"]["content"].strip()
        # Return the translated text
        return translated_text
    except Exception as e:
        # Handle any exceptions that occur during the API call
        logging.error(f"An error occurred: {e}")
        return None


def translate_turn(src_line, source_lang, target_lang, message=None):
    turn_flag = "1st"
    translated_text = translate(src_line, source_lang, target_lang, message)

    if message:
        turn_flag = "2nd"

    if translated_text:
        logging.info(f"Translated {idx}/{len(data)} - {turn_flag}.")
        return translated_text, True
    else:
        logging.error(f"Translation failed [{idx}]. - {turn_flag}.")
        time.sleep(20)
        return translated_text, False


def convert_revisions(rv_terms):
    rv_text = ""
    insertion_case = None
    prev_rv = None
    prev_item = None

    for rv in rv_terms:
        if rv[0] == " " or rv[0] == "x":
            continue

        if prev_rv == "i" and rv[0] != "i":
            rv_text += (
                insertion_case + f"\" should be {TER_CONVERT['i']} \"{prev_item}\". "
            )
            insertion_case = None

        if rv[0] == "d":
            rv_text += f'"{rv[1]}" should be {TER_CONVERT[rv[0]]}. '
        elif rv[0] == "s":
            rv_text += f'"{rv[1]}" should be {TER_CONVERT[rv[0]]} "{rv[2]}". '
        elif rv[0] == "i":
            if not insertion_case:
                insertion_case = f'"{rv[2]}'
            elif prev_rv == "i" and rv[1] == prev_item:
                insertion_case += " " + rv[2]
        prev_rv = rv[0]
        prev_item = rv[1]

    return rv_text + "\n"


def construct_demostrations(examples):
    demo_text = ""

    for example in examples:
        src = example["src"]
        hyp = example["hyp"]
        ref = example["ref"]
        revisions = convert_revisions(example["op"])
        demo_text += f"<input> {src} <hypothesis> {hyp} <reference> {ref} <revision> {revisions}\n"

    return demo_text


if __name__ == "__main__":
    domain = ""
    data = read_json("")

    source_lang = "German"
    target_lang = "English"
    results = []
    logging.info("A source example:")
    logging.info(data[:2][1]["src"])

    for idx, data_item in enumerate(data):
        src_line = data_item["src"]
        # first turn
        draft_translation, info = translate_turn(src_line, source_lang, target_lang)
        if info:
            # second turn
            demo_revisions = construct_demostrations(
                data_item["rerank_top5"][:RV_NUMBERS]
            )
            prompting_messages = [
                {"role": "assistant", "content": draft_translation},
                {
                    "role": "user",
                    "content": f"This is the second turn. Below I provide some similar translation examples and their revisions: {demo_revisions}",
                },
                {
                    "role": "user",
                    "content": f"Based on the previous examples, please translate the following text from {source_lang} to {target_lang}: {src_line}",
                },
            ]
            final_translation, info = translate_turn(
                src_line, source_lang, target_lang, prompting_messages
            )
        else:
            # failed at first turn
            logging.error(f"Failed at the first turn. index: {idx}")
            final_translation = None

        try:
            json_results = json.loads(final_translation) if final_translation else None
        except Exception as e:
            json_results = {
                "exception": str(e),
                "error_result": final_translation,
            }  # translated_text

        cur_result = {
            "index": idx,
            "src": src_line,
            "draft": draft_translation,
            "result": json_results,
        }
        results.append(cur_result)


    pickle.dump(results, open(f"{domain}_result.pkl", "wb"))

    with open("{domain}_result.json", "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, ensure_ascii=False, indent=4)

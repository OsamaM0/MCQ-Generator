from flask import Flask, request, jsonify
from transformers import pipeline

# Flask app
app = Flask(__name__)

import re
import random
from transformers import pipeline
from datetime import datetime
from PyMultiDictionary import MultiDictionary

pipe = pipeline("text2text-generation", "lmqg/t5-small-squad-qag")
dictionary = MultiDictionary()



# Auxiliary verb negation dictionary
AUXILIARY_VERBS = {
    "is": "isn't", "are": "aren't", "was": "wasn't", "were": "weren't",
    "will": "won't", "would": "wouldn't", "can": "can't", "could": "couldn't",
    "shall": "shan't", "should": "shouldn't", "do": "don't", "does": "doesn't",
    "did": "didn't", "has": "hasn't", "have": "haven't", "had": "hadn't"
}

QUESTION_WORDS_PATTERN = re.compile(
    r"\b(what|where|why|how|when|who|whom|which|how (many|much|often))\b",
    re.IGNORECASE
)

def negate_auxiliary(sentence):
    """
    Negates auxiliary verbs randomly within a sentence.

    :param sentence: The input sentence (usually a question).
    :return: Tuple with modified sentence and a flag indicating if negation was applied.
    """
    sentence = re.sub(QUESTION_WORDS_PATTERN, "", sentence).strip()
    if random.choice([True, False]):
        correct_options, wrong_options = make_options_neg([sentence])
        if not wrong_options:
            words = sentence.split()
            for i, word in enumerate(words):
                if word.lower() in AUXILIARY_VERBS:
                    words[i] = AUXILIARY_VERBS[word.lower()]
                    return ' '.join(words), False
        else:
            return wrong_options[0], False

    return sentence, True


def slightly_modify_digits(s, num):
    """
    Randomly modifies digits within a string.

    :param s: The input string containing digits.
    :param num: The number of variations to generate.
    :return: List of modified strings with altered digits.
    """
    return list(set(
        re.sub(r'\d+', lambda x: str(int(x.group()) + random.randint(-100, 100)), s)
        for _ in range(10)
    ))[:num]


def has_word_intersection(str1, str2):
    """
    Checks if two strings share more than half their words in common.

    :param str1: First string.
    :param str2: Second string.
    :return: Boolean indicating significant word overlap.
    """
    words1, words2 = set(str1.split()), set(str2.split())
    return len(words1.intersection(words2)) > (min(len(words1), len(words2)) / 2)

def generate_dummy_answers(word, num=3):
    """
    Generates a list of dummy answers for a given word.

    :param word: The input word.
    :param num: Number of dummy answers to generate.
    :return: List of dummy answers.
    """
    antoyms = []
    for an in dictionary.antonym('en', word)[:num]:
      if word in dictionary.antonym('en', an):
        antoyms.append(an)

      return antoyms


def sentence_antonym(sentence):
    sentence_words = []
    is_antoym = False
    for word in sentence.split(" "):
        antoym_word = generate_dummy_answers(word,1)
        if antoym_word and not is_antoym:
            sentence_words.append(antoym_word[0])
            is_antoym = True
        else:
            sentence_words.append(word)
    return " ".join(sentence_words), is_antoym

def make_options_neg(options):
    wrong_options = []
    correct_options = []
    for option in options:
        if len(correct_options) > 1:
            return options, []
        new_option, is_antoym = sentence_antonym(option)
        if is_antoym:
            wrong_options.append(new_option)
        else:
            correct_options.append(new_option)
    return correct_options, wrong_options



def process_question(question, answer):
    """
    Processes a single question and generates a multiple-choice question.

    :param question: The question string.
    :param answer: The correct answer string.
    :return: Dictionary containing the question, options, and the correct answer.
    """
    correct_pos = random.randint(0, 1)

    if re.search(r'\d', answer):  # Numeric answer handling
        options = slightly_modify_digits(answer, 3)
        options.insert(correct_pos, answer)
        return {"question": question, "options": options, "answer": ["A", "B", "C", "D"][correct_pos]}

    else:
        correct_options, wrong_options = make_options_neg([answer])
        if not wrong_options:
            negated_question, negated = negate_auxiliary(question)
            return {"question": f"{answer} {negated_question}", "options": [True, False], "answer": "A" if negated else "B"}

        wrong_options.insert(correct_pos, answer)
        return {"question": question, "options": wrong_options, "answer": ["A", "B", "C", "D"][correct_pos]}


def question_rule_base(questions_list):
    """
    Processes a list of questions and converts them into multiple-choice format.

    :param questions_list: List of dictionaries with questions and their answers.
    :return: List of dictionaries containing MCQs with their options and answers.
    """
    question_dict_mod = {}

    for question_dict in questions_list:
        question, answer = list(question_dict.items())[0]

        if answer in question:  # Skip redundant questions
            continue

        # Initialize or append to the question's answer list
        if question not in question_dict_mod:
            question_dict_mod[question] = answer.split(' or ') if 'or' in answer else answer.split(' and ')
        else:
            if (answer in question_dict_mod[question] or
                    len(question_dict_mod[question]) > 3 or
                    any(has_word_intersection(answer.lower(), qa.lower()) for qa in question_dict_mod[question])):
                continue
            question_dict_mod[question].append(answer)


    final_question_list = []
    for question, answers in question_dict_mod.items():
        if len(answers) == 1:  # Single correct answer
            final_question_list.append(process_question(question, answers[0]))
        else:
            correct_options, wrong_options =  make_options_neg(answers)
            correct_index = random.randint(0,max(len(wrong_options)-1, 0))

            if len(correct_options) == 1:
                options = wrong_options[:3]
                options.insert(correct_index, correct_options[0])
                options_answer = ["A","B","C","D"][correct_index]

            elif len(correct_options) == 0:
                options = wrong_options[:2]
                options = options + ["All of the above", "None is correct"]
                options_answer = "D"

            elif len(wrong_options) == 0:
                options = correct_options[:2]
                options = options + ["All of the above", "None is correct"]
                options_answer = "C"

            final_question_list.append({
                "question": question,
                "options": options,
                "answer": options_answer
            })

    return final_question_list


# Define the mcq_extractor function
def mcq_extractor(context:str, num_quizzes:int):
    inc = int(1000 * 1)
    qa_pairs = []
    
    # Step 1: Process the context and generate questions/answers
    for i in range(0, len(context), inc):
        _context = "Generate more than one question:" + context[i : i +inc ]
        output = pipe(_context, max_new_tokens=inc)
        
        # Parsing the output for questions and answers
        qa_pairs.extend(output[0]['generated_text'].split(' | '))
        if len(qa_pairs) >= num_quizzes:
            break

    # Step 2: Create the final structure
    qa_list = []
    for pair in qa_pairs[:num_quizzes]:  # Limit to num_quizzes
        try:
            question, answer = pair.split(", answer: ")
            question = question.replace("question: ", "")
            # Add question and options to the list
            qa_list.append({question.strip(): answer.strip()})
        except:
            pass

    return qa_list

# Define an API endpoint
@app.route('/generate_mcq', methods=['POST'])
def generate_mcq():
    # Get data from the request
    data = request.json
    context = data.get('context', '')
    num_quizzes = data.get('num_quizzes', 5)
    
    # Validate input
    if not context or not isinstance(num_quizzes, int):
        return jsonify({"error": "Invalid input"}), 400
    
    # Generate the MCQs
    qa_list = mcq_extractor(context, num_quizzes)
    
    # Return the result
    return jsonify({"quizzes": qa_list})

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

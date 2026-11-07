import random
import re

CONDITIONS = ["bulk", "multiturn", "chained"]
OPERATIONS_POOL = ["add", "add", "sub", "sub", "mul", "mul2d", "mul2d", "fdiv", "sq"]


def generateStep(rng, inputNum, pool):
    # Method returns a string instruction for a single step, the expected answer, and the chosen operation.

    # Filter out operations if input number is too large or small
    if inputNum < 30:
        pool = [p for p in pool if p != "sub"]
    if inputNum >= 1000:
        pool = [p for p in pool if p != "sq"]
    if inputNum > 5 * 10 ** 4:
        pool = [p for p in pool if p not in ("mul", "mul2d", "sq")]
    if inputNum > 10 ** 6:
        pool = ["fdiv", "mod"]
    
    c = rng.choice(pool)
    if c == "add":
        a = rng.randint(25, 400)
        return f'Add {a} to it.', inputNum + a, c
    if c == "sub":
        a = rng.randint(10, min(300, inputNum))
        return f'Subtract {a} from it.', inputNum - a, c
    if c == "mul":
        b = rng.randint(3, 9)
        return f'Multiply it by {b}.', inputNum * b, c
    if c == "mul2d":
        b = rng.randint(11, 29)
        return f'Multiply it by {b}.', inputNum * b, c
    if c == "fdiv":
        d = rng.randint(2, 9)
        return f'Divide it by {d} and round down.', inputNum // d, c
    if c == "mod":
        m = rng.randint(5, 13)
        return f'Take the remainder when it is divided by {m}.', inputNum % m, c
    return "Square it.", inputNum * inputNum, c


def makeTask(chainLength, idx, seed):
    # Generates a list, where element k stores the operation being perfomed,
    # the prompt for the step, and the expected result of the task.

    # Choose starting number randomly
    rng = random.Random(f'{seed}-math_dependent_mid-{chainLength}-{idx}')
    startingVal = rng.randint(20, 400)
    currentVal = startingVal
    steps = []
    for i in range(chainLength):
        prompt, currentVal, operation = generateStep(rng, currentVal, OPERATIONS_POOL)
        lead = f'Start with the number {startingVal}.' if i == 0 else "Take the result of the previous step."
        steps.append({"op": operation, "instr": f'{lead} {prompt}', "expected": currentVal})
    
    task = {"preamble": "", "steps": steps, "id" : f'math_dependent_mid-k{chainLength}-{idx:04d}', "k" : chainLength}
    return task


def bulkPrompt(task):
    prompt = "Complete all of the following steps in order.\n\n"
    steps = task["steps"]
    for i in range(len(steps)):
        prompt += f'Step {i + 1}: {task["steps"][i]["instr"]}\n'
    prompt += "\nGive the answer to every step on its own line, in exactly this form: `Step N: <integer>`. Do not write anything else."
    return prompt

def multiturnPrompt(task, stepNum, carry = ""):
    prompt = ""
    if(stepNum > 1) and (len(carry) > 0):
        prompt += carry + "\n\n"

    prompt += f'Step {stepNum}: {task["steps"][stepNum-1]["instr"]}\n\n'
    prompt += f'Reply with exactly one line in this form: `Step {stepNum}: <integer>`. Do not write anything else.'
    return prompt

def chainedPrompt(task, stepNum, prevResponses):
    prompt = ""
    if(stepNum > 1):
        val = parse_int(prevResponses[-1], stepNum - 1)
        prompt += f'The result of the previous step was: {val if val is not None else "unknown"}\n\n'
    
    prompt += f'Step {stepNum}: {task["steps"][stepNum-1]["instr"]}\n\n'
    prompt += f'Reply with exactly one line in this form: `Step {stepNum}: <integer>`. Do not write anything else.'

    return prompt


def runCondition(task, condition, llm):
    """llm(messages: list[{'role','content'}]) -> str. Returns list of raw responses."""
    k = task["k"]
    if condition == "bulk":
        return [llm([{"role": "user", "content": bulkPrompt(task)}])]
    if condition == "multiturn":
        msgs, out = [], []
        for i in range(1, k + 1):
            msgs.append({"role": "user", "content": multiturnPrompt(task, i)})
            r = llm(msgs)
            msgs.append({"role": "assistant", "content": r})
            out.append(r)
        return out
    if condition == "chained":
        out = []
        for i in range(1, k + 1):
            out.append(llm([{"role": "user", "content": chainedPrompt(task, i, out)}]))
        return out
    raise ValueError(condition)

# Scoring 

def parse_int(text, i):
    m = re.findall(rf"Step\s*{i}\s*:\s*\**\s*(-?\d+)", text)
    return int(m[-1]) if m else None



def score(task, responses, condition):
    steps, k = task["steps"], task["k"]
   
    if condition == "bulk":
        texts = [responses[0]] * k
    else:
        texts = responses
    got = [parse_int(texts[i], i + 1) for i in range(k)]
    correct = [g == s["expected"] for g, s in zip(got, steps)]
    fmt_ok = [g is not None for g in got]
   

    # conditional accuracy: step i correct AND all earlier steps correct (propagation-aware)
    prefix = [all(correct[: i + 1]) for i in range(k)]
    return {"id": task["id"], "family": "math_dependent_mid", "k": k, "condition": condition,
            "step_correct": correct, "format_ok": fmt_ok, "prefix_correct": prefix,
            "all_correct": all(correct)}


# ----------------------------------------------------------------------------
# Example local-model adapter (Ollama). Swap for llama.cpp / vLLM / transformers as needed.
# ----------------------------------------------------------------------------
def ollama_llm(model, think=False, temperature=0.0, top_p=None, top_k=None, seed=0,
               num_predict=None, num_ctx=None, host="http://localhost:11434"):
    """Ollama adapter. `think` is passed explicitly so reasoning is a controlled variable.
    The returned callable gives back ONLY the visible answer (thinking is stripped, never
    fed back into multi-turn history). Token usage accumulates in llm.usage."""
    import requests

    if num_predict is None:
        num_predict = 8192 if think else 1024
    if num_ctx is None:
        num_ctx = 16384 if think else 8192

    def llm(messages):
        opts = {"temperature": temperature, "seed": seed, "num_predict": num_predict, "num_ctx": num_ctx}
        if top_p is not None:
            opts["top_p"] = top_p
        if top_k is not None:
            opts["top_k"] = top_k
        r = requests.post(f'{host}/api/chat', json={
            "model": model, "messages": messages, "stream": False,
            "think": think, "options": opts}, timeout=1800)
        r.raise_for_status()
        j = r.json()
        content = j["message"].get("content", "") or ""
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        u = llm.usage
        u["calls"] += 1
        u["prompt_tokens"] += j.get("prompt_eval_count", 0)
        u["output_tokens"] += j.get("eval_count", 0)
        u["truncated"] += int(j.get("done_reason") == "length")
        return content

    llm.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "truncated": 0}
    return llm

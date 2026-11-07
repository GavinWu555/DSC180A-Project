# AI Contributions: Programming

Claude was used throughout the development of our task generator and pipeline. The task generator selects a random number and generates a sequences of random arithmetic operations ie. "Take 400. Add 20 to it". The core logic of the generator was developed by us.

However, both of us were unfamiliar with how we could pass the generated questions from our python script to the qwen LLM and get its answer back. Claude created the infrastructure that gives the tasks to the qwen model, receives its answer, and compares it to the expected correct answer we specified. Claude was also heavily used in the creation of run_experiment.py, which automates the entire process, giving the qwen agent tasks and then receiving its answers back and comparing it against the correct answers for hundreds of tasks.


# AI Contributions: Report
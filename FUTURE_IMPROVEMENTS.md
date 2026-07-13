### Future Improvements
1. Further consolidating the word list corpus. Although the model's valid crosswords are indeed somewhat valid (not total gibberish), they still contain acronyms and just very obscure words. See the examples in FINE-TUNING.md for better understanding on this point

The word list corpus is currently 10% SAT words (that have been deemed good for crossword by me taking an intersection with an SAT word list and a crossword-worthy words list) and 90% random words that are also good at crosswords. The issue with SAT words is that they are incredibly long and an actual crossword needs a lot of shorter (3-letter) words to physically fill all the spaces. In the future, I think it could be fruitful if I moved past SAT words and built this generator for little children learning english. Here, there is a mix of semi-long words and short words that children could learn.

2. Finding a way to indicate to a model what optimal words are. This week, I tried it by hardcoding a word list into the training data code samples and training the model off of this. Not only does this slow down inference significantly, it is not effective, as an SLM struggles with memorizing such a list of words. I didn't formally evaluate this point due to time constraints, but I partially ran a test and it was pretty much unable to output a comprehensible code snippet. The model for this is found in finetuned-models/hardcoded/qwen3-4b-crossword-grpo. The eval and training scripts are in rlvr/

In the future, I think it'd be cool to write the rewards function myself and better define that. I don't think Claude did a good job on the weights (see HARDCODED_MODELS.md for more info)

3. Constrain it to one particular domain. If the user asks for a crossword about biology, it should err on the easier side, whereas if a user asks for a more high-level concept, providing more details it should take this into account when selecting the word list.

4. For the best applicability, improve the fail time limit for Claude. Its programs might be good, just take a long time. Evaluate based on this.
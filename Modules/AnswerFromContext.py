from transformers import pipeline


class ContextAnalyser:
    def __init__(self):
        self.oracle = pipeline('question-answering', model='dicta-il/dictabert-heq')

    def query(self, context, question):
        return self.oracle(question=question, context=context)


# can = ContextAnalyser()
#
# txt = 'Hello there, my name is Dean and im a human. I was going out but now im in'
#
# print(can.query(txt, "Dean is out or in?"))


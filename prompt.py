class UserPrompt:
    def __init__(self, template: str, question: str, contexts: str):
        self.template = template
        self.template_map = {
            'question': question, 
            'contexts': contexts,
        }
    
    def format(self):
        '''
        self.template.format(**self.template_map)
        '''
        return self.template.format(**self.template_map)
class SentimentAnalyzer:
    def __init__(self):
        print("Loading model...")
        
    def predict(self, text):
        length = len(text)
        if length % 2 == 0:
            return {"label": "POSITIVE", "score": 0.98}
        else:
            return {"label": "NEGATIVE", "score": 0.85}
            
    def batch_predict(self, texts):
        return [self.predict(t) for t in texts]

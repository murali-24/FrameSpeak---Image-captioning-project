from pycocoevalcap.bleu.bleu import Bleu
#from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
#from pycocoevalcap.spice.spice import Spice

# gts: ground truths, res: model results
# Each dict should map image_id -> list of 1+ captions

def evaluate_metrics(gts, res):
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        #(Meteor(), "METEOR"),
        (Cider(), "CIDEr"),
        #(Spice(), "SPICE")
    ]
#for bleu we get list of scores. So , we zip the scores along with names of scores.
    results = {}

    for scorer, method in scorers:
        score, scores_per_instance = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for m, s in zip(method, score):
                results[m] = s #each bleu score is stored seperately in the dictionary
        else:
            results[method] = score #other methods and scores are stored as key-value pairs

    return results

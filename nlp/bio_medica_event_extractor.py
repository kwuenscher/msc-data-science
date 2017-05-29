import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
from IPython.display import display

##### Function that reduces the number of sentences that have a none label.

def reduceNoneMass(event_train):

    trigger_counts = defaultdict(float)
    labels = {y for _, y in event_train}

    for sentence, label in event_train:
        trigger_counts[sentence.sent.tokens[sentence.trigger_index]["word"]] +=1

    singletons = [word for word in trigger_counts if trigger_counts[word] == 1]

    event_train = [sentence for sentence in event_train if sentence[0].sent.tokens[sentence[0].trigger_index]["word"] not in singletons]


    for label in labels:
        basket = [getTriggerWord(event[0]) for event in event_train if event[1] == label]
        clean = []
        counter = 0
        for event in event_train:
            if event[1] == "None" and getTriggerWord(event[0]) in basket and (counter % 2) == 0.0:
                counter += 1.5
            elif event[1] == "None" and getTriggerWord(event[0]) in basket:
                counter += 1.5
                clean.append(event)
            else:
                clean.append(event)

        event_train = clean

    return event_train

##### Convinience function to extract the trigger word of a given event.

def getTriggerWord(event):
    return event.sent.tokens[event.trigger_index]["word"]

def listToString(l):
    string = ""
    for i in l:
        string += i + " "
    return string


##### Extracting tirgger words that occure for multiple classes.

def AmbigiousTriggers(data):
    globabl_triggers = allTriggerWords(data)
    labels = {y for _, y in data}
    label_specific_trigger_words = defaultdict(dict)
    for label in labels:
        label_specific_trigger_words[label] = Counter([getTriggerWord(sentence) for sentence, y in data if y == label])

    trigger_occurences = defaultdict(float)
    for trigger in globabl_triggers:
        for label in labels:
            if trigger in label_specific_trigger_words[label]:
                trigger_occurences[trigger] += 1

    return label_specific_trigger_words, Counter(trigger_occurences)

##### Extract all different trigger words and their counts.

def allTriggerWords(data):
    trigger_word_counter = Counter([getTriggerWord(sentence) for sentence, y in data])
    return trigger_word_counter


##### Extracting the minimum distance from the trigger word to the protein.

def addProtDist(result, event):
    distances = []
    protein_locations = event.sent.is_protein
    for pro in protein_locations:
        if protein_locations[pro]:
            distances.append(abs(pro - event.trigger_index))

    if distances == []:
        distances.append(1000)

    result["distance"] = min(1, min(distances)/10)

    return result

##### Function that checks whether a given sentence includes a protein mention.
##### If so, it further checks whether the trigger word lies between those mentions.

def addMention(result, event):

    trigger_ment = event.sent.mentions

    # Since only None events can occure with no Protein mentions we can add
    # a feature that assigns 0 to trigger_indexes with no protein
    # mention and 1 to trigger words with protein mention. This eliminates
    # approximately 18% of non-labels in a training sample of 7500

    if len(trigger_ment) > 0:
        result["Mention"] += 0.25

    for ment in trigger_ment:
        begin = ment["begin"]
        end = ment["end"]
        if event.trigger_index >= begin and event.trigger_index <= end:
            result["Inbetween"] += 0.25

    return result

##### Function that extracts the proteins within a sentence (not used in classification).

def getProteins(result, event):
    proteins = event.sent.mentions
    ment = []
    for prot in proteins:
        ment.append([pro["word"] for pro in event.sent.tokens[prot["begin"]:prot["end"]]])

    ment = [item for sublist in ment for item in sublist]

    feature_string = " ".join(ment)
    result[feature_string] +=1

    return result

##### Function that extracts the trigger word N-gram.

def addNGram(result, event, n):
    result[listToString(getNGramTrigger(event, n))] += 0.25
    return result


def getNGramTrigger(event, N, history = False):
    if history == False:
        ngram = [word["word"] for word in event.sent.tokens[event.trigger_index - N:event.trigger_index]]
    else:
        ngram = [word["word"] for word in event.sent.tokens[event.trigger_index:event.trigger_index + N]]
    return(ngram)

##### Function that extracts the skip-gram of a trigger token.

def getSkipGram(result, event, skip = 2):
    prefix = event.sent.tokens[event.trigger_index - skip-1:event.trigger_index - 1]
    suffix = event.sent.tokens[event.trigger_index + 1:event.trigger_index + skip +1]
    presuf = ""
    for word in prefix:
        presuf = " ".join([presuf, word["word"]])
    for word in suffix:
        presuf = " ".join([presuf, word["word"]])

    result[presuf] +=1
    return result

##### Function that extract the parents and their corresponding depenency.

def addParentsDependency(result, event):
    for parent in event.sent.parents[event.trigger_index]:
            result["Parent: " + str(parent[1]) + "->"+ str(event.sent.tokens[parent[0]]["word"])] += 0.25
    return result

##### Function that extracts all candidate arguments for a given sentence (not used in classification).

def getCandidateArgument(result, event):
    candidate_arg = event.argument_candidate_spans
    candidates = ""
    for arg in candidate_arg:
        candidate = event.sent.tokens[arg[0]:arg[1]]

        for cand in candidate:
            candidates = " ".join([candidates, cand["word"]])
    result[candidates] += 1

    return result

##### Function that extracts a history of words that preceed the trigger token.

def addHistory(result, event, n):
    history = ""
    padding = "[<Start>]"
    trigger = event.trigger_index
    start = trigger - n
    if start < 0:
        history = " ".join([padding, history])
        words = event.sent.tokens[0: trigger]
        for word in words:
            history = " ".join([history, word["word"]])
    else:
        words = event.sent.tokens[trigger - n: trigger]
        for word in words:
            history = " ".join([history, word["word"]])

    result[history] +=0.25
    return result

##### Function that adds the trigger word to the feature set including its weighted score.

def addTrigger(result, event):

    if getTriggerWord(event) in trigger_counter:
        result['trigger_word=' + event.sent.tokens[event.trigger_index]['word']] += min(1, (1/(trigger_counter[getTriggerWord(event)]))*100)
    else:
        result['trigger_word=' + event.sent.tokens[event.trigger_index]['word']] += 1/np.mean([trigger_counter[counts] for counts in trigger_counter])

    return result

##### Function that extracts the children of a trigger word, including their dependency.

def addChildDependency(result, event):
    for child in event.sent.children[event.trigger_index]:
        child_feature = "Child: " + str(child[1]) + "->"+ str(event.sent.tokens[child[0]]["word"])
        result[child_feature] += 0.25

    return result

##### Function that checks whether a given event is ambiguous

def addAmbiguity(result, event):
    label_score = defaultdict(float)
    trigger_word = getTriggerWord(event)
    if getTriggerWord(event) in most_common_triggers:
        for tokens in event.sent.tokens:
            for label in labels:
                if tokens["word"] in label_specific_triggers[label]:
                    label_score[label] +=1

        second = Counter(label_score)
        ambiguity_score = defaultdict(float)
        if len(second) > 1:
            for label in labels:
                if trigger_word in label_specific_triggers[label]:
                    ambiguity_score[label] += label_specific_triggers[label][trigger_word]/sum([label_specific_triggers[label][trig] for trig in label_specific_triggers[label]])


            winner = max(ambiguity_score.items(), key=lambda a: a[1])
            result["ambigious:" + winner[0]] += 1

    else:
        pass

    return result

##### Function that returns the counts of all trigger word occurences and
##### children occurences.

def relativeScores(data):
    labels = {y for x,y in data}
    label_trigger_count = defaultdict(float)
    trigger_counter = Counter()
    for label in labels:
        label_data = [sentence for sentence in data if sentence[1] == label]
        trigger_counter.update(Counter([getTriggerWord(x) for x, _ in label_data]))
        label_trigger_count[label] = len(Counter([getTriggerWord(x) for x, _ in label_data]))
    return trigger_counter


# Classifier

def event_feat(event):

    result = defaultdict(float)
    result = addTrigger(result, event)
    result = addProtDist(result, event)
    result = addHistory(result, event, 5)
    result = addChildDependency(result, event)
    result = addParentsDependency(result, event)
    result = addMention(result, event)
    result = addNGram(result, event, 3)
    result = addAmbiguity(result, event)
    return result


def predict_event_labels(event_candidates):

    event_x = vectorizer.transform([event_feat(e) for e in event_candidates])
    event_y = label_encoder.inverse_transform(lr.predict(event_x))
    return event_y



if __name__ in "__main__":
    import sys

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    import numpy as np
    from collections import defaultdict

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import LabelEncoder

    ##### converts labels into integers, and vice versa, needed by scikit-learn.
    label_encoder = LabelEncoder()

    ##### encodes feature dictionaries as numpy vectors, needed by scikit-learn.
    vectorizer = DictVectorizer()

    print("Prepare Data")
    ##### Converting the event candidates and their labels into vectors and integers, respectively.
    train_event_x = vectorizer.fit_transform([event_feat(x) for x,_ in event_train])
    train_event_y = label_encoder.fit_transform([y for _,y in event_train])

    print("Training")
    # Create and train the model. Feel free to experiment with other parameters and learners.
    lr = LogisticRegression(C=95, class_weight = "balanced")
    lr.fit(train_event_x, train_event_y)

    #lsvc = LinearSVC(C = 95, class_weight = "balanced")
    #lsvc.fit(train_event_x, train_event_y)

    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(25,), random_state=1)
    #clf.fit(train_event_x, train_event_y)

    print("Done")

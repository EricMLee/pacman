"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    If there is no noise then the agent will always end up where they intend
    so they can cross the bridge and they will never fall off
    """

    answerDiscount = 0.9
    answerNoise = 0.0

    return answerDiscount, answerNoise

def question3a():
    """
    Prefer the close exit (+1), risking the cliff (-10)
    You want no noise so it will go near the cliff
    You want a big discount so it is not worth going to the +10
    """

    answerDiscount = 0.2
    answerNoise = 0.0
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Prefer the close exit (+1), but avoiding the cliff (-10)
    Added a noise factor so it doesnt risk dying at the cliff
    """

    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Prefer the distant exit (+10), risking the cliff (-10)
    Bigger discount so it wants to finish asap
    Smaller Noise so it risks the cliff
    """

    answerDiscount = 0.8
    answerNoise = 0.1
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    Prefer the distant exit (+10), avoiding the cliff (-10)
    Higher Noise so its too risky to go near the cliff
    """

    answerDiscount = 0.9
    answerNoise = 0.4
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    Living Reward so the bot doesn't want to win
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 1

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    When epsilon is at 0 which means no random movements and no noise,
    the agent will just go back to the starting tile, because going back is
    guarenteed points while keep going on the bridge is unknown
    """
    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))

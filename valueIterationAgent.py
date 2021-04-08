from pacai.agents.learning.value import ValueEstimationAgent
from pacai.util import counter

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index)
        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = counter.Counter()  # A Counter is a dict with default 0
        copyOfValues = self.values.copy()
        for i in range(self.iters):
            states = self.mdp.getStates()
            for state in states:
                if not self.mdp.isTerminal(state):
                    move = self.getPolicy(state)
                    qValue = self.getQValue(state, move)
                    copyOfValues[state] = qValue
            for state in states:
                self.values[state] = copyOfValues[state]

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """
        return self.getPolicy(state)

    def getPolicy(self, state):
        """
        getPolicy(state) returns the best action according to computed values.
        """
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        bestAction = None
        bestValue = -10000000000
        for action in actions:
            qValue = self.getQValue(state, action)
            if qValue > bestValue:
                bestAction = action
                bestValue = qValue
        return bestAction

    def getQValue(self, state, action):
        """
        getQValue(state, action) returns the q-value of the (state, action) pair.
        """
        qValue = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        for transition in transitions:
            qValue += transition[1] * (self.mdp.getReward(state, action, transition[0])
                + (self.discountRate * self.values[transition[0]]))
        return qValue

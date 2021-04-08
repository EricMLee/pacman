import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance
from pacai.core.directions import Directions

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        score = 0
        closestFoodDistance = -1
        for food in oldFood.asList():
            distanceOfFood = distance.manhattan(food, newPosition)
            if closestFoodDistance == -1 or closestFoodDistance >= distanceOfFood:
                closestFoodDistance = distanceOfFood
        score += 10 / (1 + float(closestFoodDistance))

        if currentGameState.getPacmanPosition() == newPosition:
            score -= 5

        if newPosition in currentGameState.getCapsules():
            score += 100

        for ghostState in newGhostStates:
            distanceOfGhost = distance.manhattan(ghostState.getPosition(), newPosition)
            if distanceOfGhost <= 2:
                score -= 100
        return score + successorGameState.getScore()

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """
    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        """ returns best action using a minimax tree """
        return self.minimax(state, self.getTreeDepth())

    def minimax(self, state, depth):
        bestScore = -100000000
        bestMove = Directions.STOP
        listOfActions = state.getLegalActions(0)  # All of pacman's legal moves
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)  # Remove stop from moves
        for action in listOfActions:
            successorState = state.generateSuccessor(0, action)
            successorScore = self.minState(successorState, depth, 1)
            if bestScore <= successorScore:
                bestMove = action
                bestScore = successorScore
        return bestMove

    # def minState(self, state, depth):
    #     lowestScore = 10000000000
    #     numOfGhosts = state.getNumAgents() - 1

    #     ghostMoves = []
    #     moveLen = []
    #     permutations = 1

    #     for ghostID in range(numOfGhosts):
    #         listOfActions = state.getLegalActions(ghostID + 1) # All of pacman's legal moves
    #         if Directions.STOP in listOfActions:
    #             listOfActions.remove(Directions.STOP) # Remove stop from moves
    #         ghostMoves.append(listOfActions)
    #         moveLen.append(len(listOfActions))
    #         permutations = permutations * len(listOfActions)
    #     print(permutations)
    #     for action in range(permutations):
    #         print("NEW")
    #         temp = action
    #         counter = numOfGhosts
    #         newState = state
    #         while counter != 0:
    #             newState = newState.generateSuccessor(
    #                   counter,
    #                   ghostMoves[counter - 1][temp % moveLen[counter - 1]]
    #             )
    #             temp = int(temp / moveLen[counter-1])
    #             counter = counter - 1
    #         score = self.maxState(newState, depth - 1)
    #         if lowestScore > score:
    #             lowestScore = score
    #     return lowestScore

    def minState(self, state, depth, ghost):
        lowestScore = 1000000000
        score = 0
        if state.isLose() or state.isWin() or depth == 0:
            return self.getEvaluationFunction()(state)
        listOfActions = state.getLegalActions(ghost)  # All of pacman's legal moves
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)  # Remove stop from moves
        for action in listOfActions:
            nextState = state.generateSuccessor(ghost, action)
            if ghost == state.getNumAgents() - 1:
                score = self.maxState(nextState, depth - 1)
            else:
                score = self.minState(nextState, depth, ghost + 1)
            if lowestScore > score:
                lowestScore = score
        return lowestScore

    def maxState(self, state, depth):
        highestScore = -100000000
        if depth == 0 or state.isLose() or state.isWin():
            return self.getEvaluationFunction()(state)

        listOfActions = state.getLegalActions(0)
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)

        for action in listOfActions:
            successorState = state.generateSuccessor(0, action)
            successorScore = self.minState(successorState, depth, 1)
            if highestScore <= successorScore:
                highestScore = successorScore
        return highestScore

    # min
    #     ghosts will try to cuck you and get you low score
    #     return lowest score out of possibilities

    # max
    #     pacman will try to give you high score
    #     return highest score out of possibilities

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        """ returns best action using a minimax tree """
        return self.minimax(state, self.getTreeDepth())

    def minimax(self, state, depth):
        bestScore = -100000000
        bestMove = Directions.STOP
        listOfActions = state.getLegalActions(0)  # All of pacman's legal moves
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)  # Remove stop from moves
        for action in listOfActions:
            successorState = state.generateSuccessor(0, action)
            successorScore = self.minState(successorState, depth, 1, -1000000, 1000000)
            if bestScore <= successorScore:
                bestMove = action
                bestScore = successorScore

        return bestMove

    def minState(self, state, depth, ghost, alpha, beta):
        lowestScore = 1000000000
        score = 0
        if state.isLose() or state.isWin() or depth == 0:
            return self.getEvaluationFunction()(state)
        listOfActions = state.getLegalActions(ghost)
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)
        for action in listOfActions:
            nextState = state.generateSuccessor(ghost, action)
            if ghost == state.getNumAgents() - 1:
                score = self.maxState(nextState, depth - 1, alpha, beta)
            else:
                score = self.minState(nextState, depth, ghost + 1, alpha, beta)
            if lowestScore > score:
                lowestScore = score
            if beta > score:
                beta = score
            if beta < alpha:
                break
        return lowestScore

    def maxState(self, state, depth, alpha, beta):
        highestScore = -100000000
        if state.isWin():
            return 10000 
        if state.isLose():
            return -10000 
        if depth == 0:
            return self.getEvaluationFunction()(state)

        listOfActions = state.getLegalActions(0)
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)

        for action in listOfActions:
            successorState = state.generateSuccessor(0, action)
            successorScore = self.minState(successorState, depth, 1, alpha, beta)
            if highestScore <= successorScore:
                highestScore = successorScore
            if alpha < successorScore:
                alpha = successorScore
            if beta < alpha:
                break
        return highestScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, state):
        """ returns best action using a minimax tree """
        return self.minimax(state, self.getTreeDepth())

    def minimax(self, state, depth):
        bestScore = -100000000
        bestMove = Directions.STOP
        listOfActions = state.getLegalActions(0)  # All of pacman's legal moves
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)  # Remove stop from moves
        for action in listOfActions:
            successorState = state.generateSuccessor(0, action)
            successorScore = self.minState(successorState, depth, 1)
            if bestScore <= successorScore:
                bestMove = action
                bestScore = successorScore
        return bestMove

    def minState(self, state, depth, ghost):
        score = 0
        if state.isLose() or state.isWin() or depth == 0:
            return self.getEvaluationFunction()(state)
        listOfActions = state.getLegalActions(ghost)  # All of pacman's legal moves
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)  # Remove stop from moves
        for action in listOfActions:
            nextState = state.generateSuccessor(ghost, action)
            if ghost == state.getNumAgents() - 1:
                score = score + self.maxState(nextState, depth - 1)
            else:
                score = score + self.minState(nextState, depth, ghost + 1)
        return score

    def maxState(self, state, depth):
        highestScore = -100000000
        if depth == 0 or state.isLose() or state.isWin():
            return self.getEvaluationFunction()(state)

        listOfActions = state.getLegalActions(0)
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)

        for action in listOfActions:
            successorState = state.generateSuccessor(0, action)
            successorScore = self.minState(successorState, depth, 1)
            if highestScore <= successorScore:
                highestScore = successorScore
        return highestScore

    # def getAction(self, state):
    #     """ returns best action using a minimax tree """
    #     return self.expectimax(state, self.getTreeDepth())

    # def expectimax(self, state, depth):
    #     listOfActions = state.getLegalActions(0)
    #     if Directions.STOP in listOfActions:
    #         listOfActions.remove(Directions.STOP)
    #     possibleActions = [0, 0, 0, 0]
    #     for action in listOfActions:
    #         successorState = state.generateSuccessor(0, action)
    #         if action == Directions.NORTH:
    #             self.ghostMoves(successorState, depth, 1, possibleActions, 0)
    #         if action == Directions.EAST:
    #             self.ghostMoves(successorState, depth, 1, possibleActions, 1)
    #         if action == Directions.SOUTH:
    #             self.ghostMoves(successorState, depth, 1, possibleActions, 2)
    #         if action == Directions.WEST:
    #             self.ghostMoves(successorState, depth, 1, possibleActions, 3)
    #     print(possibleActions)
    #     if Directions.NORTH not in listOfActions:
    #         possibleActions[0] = -100000000000000
    #     if Directions.EAST not in listOfActions:
    #         possibleActions[1] = -100000000000000
    #     if Directions.SOUTH not in listOfActions:
    #         possibleActions[2] = -100000000000000
    #     if Directions.WEST not in listOfActions:
    #         possibleActions[3] = -100000000000000

    #     if possibleActions[0] >= possibleActions[1] and possibleActions[0]
    #       >= possibleActions[2] and possibleActions[0] >= possibleActions[3]:
    #         return Directions.NORTH

    #     if possibleActions[1] >= possibleActions[0] and possibleActions[1]
    #       >= possibleActions[2] and possibleActions[1] >= possibleActions[3]:
    #         return Directions.EAST

    #     if possibleActions[2] >= possibleActions[1] and possibleActions[2]
    #       >= possibleActions[0] and possibleActions[2] >= possibleActions[3]:
    #         return Directions.SOUTH

    #     if possibleActions[3] >= possibleActions[1] and possibleActions[3]
    #       >= possibleActions[2] and possibleActions[3] >= possibleActions[0]:
    #         return Directions.WEST

    #     return Directions.STOP

    # def ghostMoves(self, state, depth, ghost, possibleActions, myAction):
    #     if depth == 0 or state.isLose() or state.isWin():
    #         possibleActions[myAction] =
    #           possibleActions[myAction] + self.getEvaluationFunction()(state)
    #     else:
    #         listOfActions = state.getLegalActions(ghost)
    #         if Directions.STOP in listOfActions:
    #             listOfActions.remove(Directions.STOP)
    #         for action in listOfActions:
    #             nextState = state.generateSuccessor(ghost, action)
    #             if ghost == state.getNumAgents() - 1:
    #                 self.pacmanMoves(nextState, depth - 1, possibleActions, myAction)
    #             else:
    #                 self.ghostMoves(nextState, depth, ghost + 1, possibleActions, myAction)

    # def pacmanMoves(self, state, depth, possibleActions, myAction):
    #     if depth == 0 or state.isLose() or state.isWin():
    #         possibleActions[myAction] =
    #           possibleActions[myAction] + self.getEvaluationFunction()(state)
    #     else:
    #         listOfActions = state.getLegalActions(0)
    #         if Directions.STOP in listOfActions:
    #             listOfActions.remove(Directions.STOP)
    #         for action in listOfActions:
    #             successorState = state.generateSuccessor(0, action)
    #             self.ghostMoves(successorState, depth, 1, possibleActions, myAction)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>

    Pacman should be penalized based on distance of closest food and number of food
    Pacman should get every capsule it can

    """
    # Useful information you can extract.
    score = currentGameState.getScore()
    pacPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    n = 0
    closestFoodDistance = 10000
    for food in foodList:
        distanceOfFood = distance.manhattan(pacPosition, food)
        if closestFoodDistance > distanceOfFood:
            closestFoodDistance = distanceOfFood
        n = n + 1
    if n != 0:
        score = score + 1000 / (1 + float(n))
        score = score - (closestFoodDistance * 2)
    else:
        score = score + 10000
    capsuleList = currentGameState.getCapsules()
    score = score - (10 * len(capsuleList))
    if n == 0:
        score = score + 10000
    return score

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

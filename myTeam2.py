from pacai.util import reflection
from pacai.core.directions import Directions
from pacai.util import counter
from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.agents.capture.capture import CaptureAgent
from pacai.core import distanceCalculator
import random
import math

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.student.team_TeamHardcode',
        second = 'pacai.student.team_TeamHardcode'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = AttackAgent
    secondAgent = DefensiveAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

class AttackAgent(ReflexCaptureAgent):
    midpointTiles = []
    targetTile = 0
    badTile = 0
    maxDistance = 0
    pacmanCounter = 0
    totalFood = 0
    totalCapsules = 0

    def __init__(self, index, **kwargs):
        super().__init__(index)
        global midpointTiles2
        global targetTile
        global badTile
        badTile = 0
        global maxDistance
        global pacmanCounter
        global totalFood
        global totalCapsules


    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """
        global midpointTiles2
        global badTile
        global pacmanCounter
        global targetTile
        actions = gameState.getLegalActions(self.index)
        myState = gameState.getAgentState(self.index)
        if myState.isPacman():
            pacmanCounter += 1
        else:
            pacmanCounter = 0
            badTile += 1
        if badTile % 20 == 0:
            temp = midpointTiles2.index(random.choice(midpointTiles2))
            temp2 = midpointTiles2.index(random.choice(midpointTiles2))
            for i in range(1,3):
                if abs(targetTile - temp) < abs(targetTile - temp2):
                    temp = temp2
                temp2 = midpointTiles2.index(random.choice(midpointTiles2))
            targetTile = temp
            
        global oldDefenders
        enemies = self.getOpponents(gameState)
        oldDefenders = []
        oldAttackers = []
        for enemy in enemies:
            if gameState.getAgentState(enemy).isPacman():
                oldAttackers.append(gameState.getAgentState(enemy))
            else:
                oldDefenders.append(gameState.getAgentState(enemy))

        global defenderID        
        if len(oldDefenders) != 0:
            defenderID = oldDefenders[0]
        # *** Gamestate Ghost Distance ***
        global oldClosestDefender
        global myOldPos
        myOldPos = gameState.getAgentState(self.index).getPosition()
        oldClosestDefender = float("inf")
        for defender in oldDefenders:
            defenderDistance = self.getMazeDistance(defender.getPosition(), myOldPos)
            if defenderDistance < oldClosestDefender:
                oldClosestDefender = defenderDistance
        if len(oldDefenders) == 0:
            oldClosestDefender = 0
            
        # *** Gamestate Food ***
        global oldTotalFoodDistance
        global oldClosestFood
        global oldFoodList
        global oldGettableFood

        oldTotalFoodDistance = 0
        oldClosestFood = float("inf")
        oldFoodList = self.getFood(gameState).asList()
        oldGettableFood = float("inf")
        for food in oldFoodList:
            currentFoodDistance = self.getMazeDistance(myOldPos, food)
            oldTotalFoodDistance += currentFoodDistance
            if len(oldDefenders) != 0:
                defendFood = self.getMazeDistance(defenderID.getPosition(), food)
                if currentFoodDistance < oldClosestFood and defendFood >= currentFoodDistance:
                    oldGettableFood = currentFoodDistance
            if currentFoodDistance < oldClosestFood:
                oldGettableFood = currentFoodDistance
                oldClosestFood = currentFoodDistance

        # *** Gamestate Capsules ***
        global oldNumCapsules
        global oldClosestCapsule
        oldNumCapsules = self.getCapsules(gameState)
        oldClosestCapsule = float('inf')
        for capsule in oldNumCapsules:
            oldCapsuleDistance = self.getMazeDistance(myOldPos, capsule)
            if oldCapsuleDistance < oldClosestCapsule:
                oldClosestCapsule = oldCapsuleDistance
        if oldClosestCapsule == float('inf'):
            oldClosestCapsule = 0

        return self.minimax(gameState, 1)
       
    def legalActions(self, state, agent):
        actions = state.getLegalActions(agent)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        return actions

    def maximum(self, state, depth, alpha, beta):
        if state.isWin():
            return float("inf")
        if state.isLose():
            return -float("inf")
        if depth == 0:
            return self.evaluate(state, Directions.STOP)
        maximumValue = - float("inf")
        for action in self.legalActions(state, self.index):
            successor = state.generateSuccessor(self.index, action)
            opponents = self.getOpponents(state)
            current = self.minimum(successor, depth, alpha, beta, opponents, 0)
            maximumValue = max(maximumValue, current)
            alpha = max(alpha, current)
            if beta <= alpha:
                break
        return maximumValue

    def minimum(self, state, depth, alpha, beta, ghost, index):
        if state.isWin():
            return float("inf")
        if state.isLose():
            return -float("inf")
        if depth == 0:
            return self.evaluate(state, Directions.STOP)
        minimumValue = float("inf")
        actions = self.legalActions(state, ghost[index])
        for action in actions:
            successor = state.generateSuccessor(ghost[index], action)
            if index == len(ghost) - 1:
                current = self.maximum(successor, depth - 1, alpha, beta)
            else:
                current = self.minimum(successor, depth, alpha, beta, ghost, index + 1)
            minimumValue = min(minimumValue, current)
            beta = min(beta, current)
            if beta <= alpha:
                break
        return minimumValue

    def minimax(self, state, depth):
        # initialize the first action to STOP
        # initialize score negative infinity
        # get all the legal moves for given state
        Action = Directions.STOP
        Score = - float("inf")
        actions = self.legalActions(state, self.index)
        # go through all the legal actions
        for action in actions:
            # get the successor state for the agent after the given action is
            # taken
            successor = state.generateSuccessor(self.index, action)
            # apply the maximum function with the given successor state
            # and the depth given initially, and negative and positive infinity
            score = self.maximum(successor, depth, - float("inf"), float("inf"))
            # if this newly calculated score is larger than the initial
            # score, make the action in this loop the new Action and
            # update the new score into Score
            if score > Score:
                Action = action
                Score = score
        return Action

    def registerInitialState(self, gameState):
        self.red = gameState.isOnRedTeam(self.index)
        self.distancer = distanceCalculator.Distancer(gameState.getInitialLayout())
        self.distancer.getMazeDistances()
        
        abc = gameState.getInitialLayout()
        width = abc.getWidth()
        height = abc.getHeight()
        midpoint = int(width/2)
        global midpointTiles2
        global targetTile
        global maxDistance
        global totalCapsules
        global totalFood
        totalFood = len(self.getFood(gameState).asList())
        totalCapsules = len(self.getCapsules(gameState))
        midpointTiles2 = []
        redPenalty = 1
        redPenalty2 = -1
        if self.red:
            redPenalty = -1
            redPenalty2 = -1
        longestPath = 0
        shortestPath = float("inf")
        for i in range(1, height - 1):
            if not abc.isWall((midpoint + redPenalty2, i)) and not abc.isWall((midpoint + redPenalty2 + (redPenalty*-1), i)):
                midpointTiles2.append(((midpoint + redPenalty2), i))
                temp = self.getMazeDistance((midpoint + redPenalty2, i), gameState.getAgentState(self.index).getPosition())
                if shortestPath > temp:
                    shortestPath = temp
                    targetTile = len(midpointTiles2) - 1
                if longestPath < temp:
                    maxDistance = temp
                    
        # print("Initial Target Tile",targetTile)
        targetTile = 0

    def aStarSearch(problem, heuristic):
        """
        Search the node that has the lowest combined cost and heuristic first.
        """
    
        # *** Your Code Here ***
        visited_nodes = []
        # using priority queue again
        created_priorityQueue = PriorityQueue()
        # cost and heuristics
        created_priorityQueue.push((problem.startingState(), [], 0),
                                   heuristic(problem.startingState(), problem))
        while not created_priorityQueue.isEmpty():
            state, actions, cost = created_priorityQueue.pop()
            if problem.isGoal(state):
                return actions
            if state in visited_nodes:
                continue
            visited_nodes.append(state)
            Successors = problem.successorStates(state)
            for s in Successors:
                # taking the cost and heuristics into account while pushing
                created_priorityQueue.push((s[0], actions + [s[1]], cost + s[2]),
                                           s[2] + cost + heuristic(s[0], problem))
        return None


    def getFeatures(self, gameState, action):
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        global midpointTiles2
        global targetTile
        global pacmanCounter
        global totalFood
        global totalCapsules
        global oldDefenders
        global oldClosestDefender
        global myOldPos
        global oldTotalFoodDistance
        global oldClosestFood
        global oldFoodList
        global oldGettableFood
        global oldNumCapsules
        global oldClosestCapsule
        global defenderID
        global targetTile

        # if abs(midpointTiles2[targetTile][0] - myPos[0]) > 4:
        #     features['isGhost'] = abs(midpointTiles2[targetTile][0] - myPos[0])

        if myState.isGhost:
            features['newPath'] = self.getMazeDistance(midpointTiles2[targetTile], myPos)
        else:
            features['isPacman'] = 1        
        enemies = self.getOpponents(successor)
        defenders = []
        attackers = []
        for enemy in enemies:
            if successor.getAgentState(enemy).isPacman():
                attackers.append(successor.getAgentState(enemy))
            else:
                defenders.append(successor.getAgentState(enemy))
    
            # features['newPath'] = self.getMazeDistance(midpointTiles2[targetTile], myPos)
        # *** Successor Ghost Distance ***
        closestDefender = float("inf")
        for defender in defenders:
            defenderDistance = self.getMazeDistance(defender.getPosition(), myPos)
            if defenderDistance < closestDefender:
                closestDefender = defenderDistance
                defenderID = defender
        if len(defenders) == 0:
            closestDefender = 0

        # *** Ghost Comparing ***

        if closestDefender > 5 and myState.isGhost():
            features['isGhost'] = 1
        if closestDefender <= 5:
            features['closestGhost'] = closestDefender
        if closestDefender <= 1:
            if not defenderID.isScared():
                features['tooClose'] = 1
        
        # *** Successor Food ***
        totalFoodDistance = 0
        closestFood = float("inf")
        foodList = self.getFood(successor).asList()
        gettableFood = float("inf")
        for food in foodList:
            currentFoodDistance = self.getMazeDistance(myPos, food)
            totalFoodDistance += currentFoodDistance
            
            if len(defenders) != 0:
                defendFood = self.getMazeDistance(defenderID.getPosition(), food)
                if currentFoodDistance < closestFood and defendFood >= currentFoodDistance:
                    gettableFood = currentFoodDistance
            if currentFoodDistance < closestFood:
                closestFood = currentFoodDistance
                gettableFood = currentFoodDistance
        # *** Food Comparing ***
        if len(oldFoodList) - len(foodList) == 0:
            if gettableFood < oldGettableFood:
                features['gotCloserToGettableFood'] = 1
            elif closestFood < oldClosestFood:
                features['ghostDistance'] = closestDefender
                # features['gotCloserToFood'] = 1
        else:
            features['eatFood'] = 1

        # *** Successor Capsules ***
        closestCapsuleID = 0
        numCapsules = self.getCapsules(successor)
        closestCapsule = float('inf')
        for capsule in numCapsules:
            currentCapsuleDistance = self.getMazeDistance(myPos, capsule)
            if len(defenders) != 0:
                defendCapsule = self.getMazeDistance(defenderID.getPosition(), capsule)
                if currentCapsuleDistance < closestCapsule and defendCapsule > currentCapsuleDistance:
                    closestCapsule = currentCapsuleDistance
                    # *** Capsules Comparing ***
                    if len(oldNumCapsules) > len(numCapsules):
                        features['gotCapsule'] = 1
                    if closestCapsule < oldClosestCapsule:
                        features['gotCloserToCapsule'] = 1
                
        if closestCapsule == float('inf'):
            closestCapsule = 0
        return features

    def getWeights(self, gameState, action):
        return {
            'newPath': -1000,
            'tooClose': -100,
            'isGhost': -10000,
            'isPacman': 0,
            'closestGhost': 100,
            'gotCloserToGettableFood': 200,
            'eatFood': 1000,
            'gotCapsule': 20000,
            'gotCloserToCapsule': 1500,
            'gotCloserToFood': 50,
            'ghostDistance': 5000

        }

class DefensiveAgent(ReflexCaptureAgent):
    def __init__(self, index, **kwargs):
        global test
        test = 0
        global assumedAttacker
        assumedAttacker = 0
        global tracker
        global midpointTiles
        midpointTiles = []
        global invader
        global deadends
        # tracker = [0, 0, 0, 0]
        super().__init__(index)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """
        enemies = self.getOpponents(gameState)
        invaders = []
        for enemy in enemies:
            if gameState.getAgentState(enemy).isPacman():
                invaders.append(gameState.getAgentState(enemy))
        global invader
        if len(invaders) != 0:
            invader = 1
        else:
            invader = 0
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def registerInitialState(self, gameState):
        self.red = gameState.isOnRedTeam(self.index)
        self.distancer = distanceCalculator.Distancer(gameState.getInitialLayout())
        self.distancer.getMazeDistances()
        global invader
        invader = 0
        
        abc = gameState.getInitialLayout()
        width = abc.getWidth()
        height = abc.getHeight()
        midpoint = int(width/2)
        global midpointTiles
        midpointTiles = []
        midpointTiles.append((0, 0))
        redPenalty = 1
        redPenalty2 = 0
        if self.red:
            redPenalty = -1
            redPenalty2 = -1
        
        for i in range(1, height - 1):
            counter = 0
            while abc.isWall((midpoint + redPenalty2 + ((2 + counter) * redPenalty), i)):
                counter += 1
            midpointTiles.append(((midpoint + redPenalty2 + ((2 + counter) * redPenalty)), i))
        global deadends
        deadends = {}
        for n in range(1, height - 1):
            for i in range(1, width - 1):
                if not abc.isWall((i, n)):
                    counter = 0
                    opening = (0, 0)
                    if not abc.isWall((i - 1, n)):
                        opening = (i - 1, n)
                        counter += 1
                    if not abc.isWall((i + 1, n)):
                        opening = (i + 1, n)
                        counter += 1
                    if not abc.isWall((i, n + 1)):
                        opening = (i, n + 1)
                        counter += 1
                    if not abc.isWall((i, n - 1)):
                        opening = (i, n - 1)
                        counter += 1
                    if counter == 1:
                        counter = 0
                        opening2 = (0,0)
                        if not abc.isWall((opening[0] - 1, opening[1])):
                            if((opening[0] - 1, opening[1]) != (i, n)):
                                opening2 = (opening[0] - 1, opening[1])
                            counter += 1
                        if not abc.isWall((opening[0] + 1, opening[1])):
                            if((opening[0] + 1, opening[1]) != (i, n)):
                                opening2 = (opening[0] + 1, opening[1])
                            counter += 1
                        if not abc.isWall((opening[0], opening[1] - 1)):
                            if((opening[0], opening[1] - 1) != (i, n)):
                                opening2 = (opening[0], opening[1] - 1)
                            counter += 1
                        if not abc.isWall((opening[0], opening[1] + 1)):
                            if((opening[0], opening[1] + 1) != (i, n)):
                                opening2 = (opening[0], opening[1] + 1)
                            counter += 1
                        if counter == 2:
                            deadends[opening2] = [opening, (i, n)]
        print(deadends)



    def minimax(self, state, depth):
        bestScore = -float("inf")
        bestMove = Directions.STOP
        listOfActions = state.getLegalActions(self.index)
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)  # Remove stop from moves
        for action in listOfActions:
            successorState = state.generateSuccessor(self.index, action)
            opponents = self.getOpponents(state)
            successorScore = self.minState(successorState, depth, opponents, 0, -1000000, 1000000)
            if bestScore <= successorScore:
                bestMove = action
                bestScore = successorScore
        return bestMove

    def minState(self, state, depth, ghosts, start, alpha, beta):
        lowestScore = 1000000000
        score = 0
        if state.isWin():
            return 10000
        if state.isLose():
            return -10000
        if depth == 0:
            return self.evaluate(state, Directions.STOP)
        listOfActions = state.getLegalActions(ghosts[start])
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)  # Remove stop from moves
        for action in listOfActions:
            nextState = state.generateSuccessor(ghosts[start], action)
            if start == len(ghosts) - 1:
                score = self.maxState(nextState, depth - 1, alpha, beta)
            else:
                score = self.minState(nextState, depth, ghosts, start + 1, alpha, beta)
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
            return self.evaluate(state, Directions.STOP)

        listOfActions = state.getLegalActions(self.index)
        if Directions.STOP in listOfActions:
            listOfActions.remove(Directions.STOP)

        for action in listOfActions:
            successorState = state.generateSuccessor(self.index, action)
            opponents = self.getOpponents(state)
            successorScore = self.minState(successorState, depth, opponents, 0, alpha, beta)
            if highestScore <= successorScore:
                highestScore = successorScore
            if alpha < successorScore:
                alpha = successorScore
            if beta < alpha:
                break
        return highestScore

    def getFeatures(self, gameState, action):
        global deadends
        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0
        else:
            features['isGhost'] = 1


        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        
        if ((int(myPos[0]), int(myPos[1]))) in deadends:
            myList = deadends[myPos]
            for attacker in invaders:
                if attacker.getPosition() in myList and not myState.isScared():
                    features['trapped'] = 1
                    print("Trapped")

        if invader == 1:
            features['numInvaders'] = 1
        
        
        features['invaderDistance'] = 2
        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
            if myState.isScared():
                if min(dists) == 2:
                    features['scared'] = 1

        enemies = self.getOpponents(gameState)
        defenders = []
        attackers = []
        d = 0
        for enemy in enemies:
            if gameState.getAgentState(enemy).isPacman():
                attackers.append(gameState.getAgentState(enemy))
                global assumedAttacker
                assumedAttacker = enemy
                global test
                test = 1
            else:
                defenders.append(gameState.getAgentState(enemy))
                d = enemy
        # Make defender not wait right on border
        if test == 1 and len(attackers) == 0:
            attackerState = successor.getAgentState(assumedAttacker)
            attPos = attackerState.getPosition()
            targetDest = midpointTiles[int(attPos[1])]
            features['chaser'] = self.getMazeDistance(targetDest, myPos)
        if len(attackers) == 0 and test == 0:
            attackerState = successor.getAgentState(d)
            attPos = attackerState.getPosition()
            targetDest = midpointTiles[int(attPos[1])]
            features['chaser'] = self.getMazeDistance(targetDest, myPos)
        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -100000,
            'onDefense': 100,
            'invaderDistance': -10,
            'closeFood': -.2,
            'xChase': -20,
            'yChase': -20,
            'chaser': -100,
            'isGhost': 100000,
            'trapped': 100000,
            'scared': 200
        }

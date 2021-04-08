"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    fringe = Stack()
    fringe.push((problem.startingState(), [], 0))
    visited = []
    while not fringe.isEmpty():
        state, currentPath, cost = fringe.pop()
        if problem.isGoal(state):
            return currentPath
        if state in visited:
            continue
        visited.append(state)
        for nextState in problem.successorStates(state):
            fringe.push((nextState[0], currentPath + [nextState[1]], cost + nextState[2]))
    return None


def breadthFirstSearch(problem):
    # print("Start: %s" % (str(problem.startingState())))
    fringe = Queue()
    fringe.push((problem.startingState(), [], 0))
    visited = []
    while not fringe.isEmpty():
        state, currentPath, cost = fringe.pop()
        if problem.isGoal(state):
            return currentPath
        if state in visited:
            continue
        visited.append(state)
        for nextState in problem.successorStates(state):
            fringe.push((nextState[0], currentPath + [nextState[1]], cost + nextState[2]))
    return None

def uniformCostSearch(problem):
    fringe = PriorityQueue()
    fringe.push((problem.startingState(), [], 0), 0)
    visited = []
    while not fringe.isEmpty():
        state, currentPath, cost = fringe.pop()
        if problem.isGoal(state):
            return currentPath
        if state in visited:
            continue
        visited.append(state)
        for nextState in problem.successorStates(state):
            fringe.push((nextState[0], currentPath + [nextState[1]], cost + nextState[2]), cost)
    return None

def aStarSearch(problem, heuristic):
    fringe = PriorityQueue()
    fringe.push((problem.startingState(), [], 0), 0)
    visited = []
    while not fringe.isEmpty():
        state, currentPath, cost = fringe.pop()
        if problem.isGoal(state):
            return currentPath
        if state in visited:
            continue
        visited.append(state)
        for nextState in problem.successorStates(state):
            fringe.push(
                (nextState[0], currentPath + [nextState[1]], cost + nextState[2]),
                cost + nextState[2] + heuristic(nextState[0], problem)
            )
    return None

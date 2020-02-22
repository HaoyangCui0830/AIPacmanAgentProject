# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint
from util import PriorityQueue
import sys
import math

Mode = "D"
foodMissed = None
lastFoodList = []
newFoodList = []
foodLostTimer = 0
enemyLastPosition = None
enemyTimer = 0


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'AstarAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)

class RewardTable():
    def __init__(self, start, states):
        self.start = start
        self.states = states
        self.rewards = util.Counter()
        actionsList = util.Counter()
        for s in states:
            i, j = s
            if (i + 1, j) in states:
                actionsList[(s, Directions.EAST)] = (i + 1, j)
            if (i - 1, j) in states:
                actionsList[(s, Directions.WEST)] = (i - 1, j)
            if (i, j + 1) in states:
                actionsList[(s, Directions.NORTH)] = (i, j + 1)
            if (i, j - 1) in states:
                actionsList[(s, Directions.SOUTH)] = (i, j - 1)
        self.actionsDict = actionsList
        #print(self.actionsDict)

    def rewardDistribution(self, state, reward):
        x, y = state
        self.rewards[state] += reward
        for i in range(-4, 5):
            for j in range(-4, 5):
                if (x+i, y+j) in list(self.states):
                    self.rewards[(x+i, y+j)] += reward / math.exp(abs(i)+abs(j))

    def assignReward(self, state, r):
        self.rewards[state] += r
        #self.rewardDistribution(state, r)

    def getReward(self, state):
        return self.rewards[state]

    def getStates(self):
        return list(self.states)

    def availableActions(self, state):
        availableActionsList = []
        for tuple in self.actionsDict.keys():
            if tuple[0] == state:
                availableActionsList.append(tuple[1])
        return availableActionsList

    def getNextState(self, state, action):
        # return [self.actionsDict[(state, action)],]
        for tuple in self.actionsDict.keys():
            if str(tuple[0]) == str(state) and str(tuple[1]) == str(action):
                # print('NoneType')
                # print([self.actionsDict.get(tuple)])
                return [self.actionsDict.get(tuple)]





class ValueIterationCalculator():
    '''
    This is a Iteration Calculate class, used to calculate MDP value
    '''

    def __init__(self, RewardTable, discount = 0.8, iteration = 10):
        self.RewardTable = RewardTable
        self.discount = discount
        self.iteration = iteration
        self.StateValues = util.Counter()

        for i in range(self.iteration):
            values = util.Counter()
            for s in self.RewardTable.getStates():
                nextAction = self.selectAction(s)
                if nextAction:
                    values[s] = self.getQSA(s, nextAction)
            self.StateValues = values

    def getQSA(self, state, action):
        Q = 0
        # print(self.RewardTable.getNextState(state, action))
        # print(len(self.RewardTable.getNextState(state, action)))
        if self.RewardTable.getNextState(state, action) != None:
            #print('has next state')
            for newstate in self.RewardTable.getNextState(state, action):
                reward = self.RewardTable.getReward(newstate)
                Q = reward + self.discount * self.StateValues[newstate]
        return Q

    def selectAction(self, state):
        values = util.Counter()
        best_action = ''
        best_value = -10000
        for action in self.RewardTable.availableActions(state):
            #print(values[action])
            #print(action)
            values[action] = self.getQSA(state, action)
            if values[action] > best_value:
                best_value = values[action]
                best_action = action
        #print('best_action' + str(best_action))
        #print(values)
        return best_action




class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.enemy = self.getOpponents(gameState)
    #print(self.enemy)
    # TODO need to change
    self.walls =  set(gameState.data.layout.walls.asList())
    self._maxx = max([item[0] for item in self.walls])
    self._maxy = max([item[1] for item in self.walls])
    self.sign = 1 if gameState.isOnRedTeam(self.index) else -1
    self.homeXBoundary = self.start[0] + ((self._maxx // 2 - 1) * self.sign)
    cells = [(self.homeXBoundary, y) for y in range(1, self._maxy)]
    self.homeBoundaryCells = [item for item in cells if item not in self.walls]
    available_states = self._getGrid(1, 1, self._maxx, self._maxy)
    self.available_actions = util.Counter()
    for cell in available_states:
        x, y = cell
        if (x - 1, y) in available_states:
            self.available_actions[cell] += 1
        if (x + 1, y) in available_states:
            self.available_actions[cell] += 1
        if (x, y - 1) in available_states:
            self.available_actions[cell] += 1
        if (x, y + 1) in available_states:
            self.available_actions[cell] += 1



  def chooseAction(self, gameState):
    #print(self.getOpponents(gameState))
    start_time = time.time()

    start = time.time()
    area_range = 6
    backhomeReward = 1.0
    getFoodReward = 0.15
    potentialDeadEndReward = -0.1
    ghostAttackReward = -1.5
    min_enemy_distance = 5
    currentState = gameState.getAgentState(self.index)
    currentPosition = currentState.getPosition()
    x, y = currentPosition

    # get the states needed to be calculate
    grid = self._getGrid(x - area_range, y - area_range, x + area_range, y + area_range)
    # retur the search area cycle
    grid = {cell for cell in grid if self.distancer.getDistance(currentPosition, cell) <= area_range}

    rewardTable = RewardTable(currentState, grid)

    # reward for food
    foodList =  self.getFood(gameState).asList()
    if len(foodList) > 2: #still hold less than 18 food
        for food in foodList:
            self.updateReward(grid, rewardTable, getFoodReward, currentPosition, food)

    # Initialize enemy information
    enemies = []
    enemyNearby = False;
    for i in self.getOpponents(gameState):
        enemyState = gameState.getAgentState(i)
        #print(enemyState)
        #print(i)
        if enemyState.getPosition():
            if self.distancer.getDistance(currentPosition, enemyState.getPosition()) <= min_enemy_distance:
                #print(self.distancer.getDistance(currentPosition, enemyState.getPosition()))
                enemyNearby = True
                enemies.append((enemyState, enemyState.getPosition()))

    # TODO handle multiple enemies
    if enemyNearby:
        #print(min([item.scaredTimer for item, _ in enemies]))
        closestEnemyDistance = 5
        for enemyState, enemyPosition in enemies:
            if enemyState.scaredTimer > 2:
                continue
            enemy_distance = self.distancer.getDistance(currentPosition, enemyPosition)
            reward = ghostAttackReward * (area_range + 1.0 - enemy_distance)
            #print(reward)
            # different reward assign method here
            rewardTable.assignReward(enemyPosition, reward)
            closestEnemyDistance = min(closestEnemyDistance, enemy_distance)
        for cell in grid:
            if self.atHome(cell):
                continue
            if min([item.scaredTimer for item, _ in enemies]) > 8:
                #print(min([item.scaredTimer for item, _ in enemies]))
                continue
            if self.getDistanceHome(cell) > self.getDistanceHome(currentPosition):
                rewardTable.rewardDistribution(cell, potentialDeadEndReward)
            if self.available_actions[cell] == 1 and cell not in foodList:
                rewardTable.assignReward(cell, potentialDeadEndReward * 5)
            if self.distancer.getDistance(currentPosition, enemyPosition) < 3 and self.available_actions[cell] == 1:
                #print("dead trap!")
                rewardTable.assignReward(cell, potentialDeadEndReward * 10)
            if self.distancer.getDistance(currentPosition, enemyPosition) < 2 * self.distancer.getDistance(cell, enemyPosition) + 1 and self.available_actions[cell] == 1:
                #print("dead trap!")
                rewardTable.assignReward(cell, potentialDeadEndReward * 10)
            if self.distancer.getDistance(currentPosition, enemyPosition) > self.distancer.getDistance(cell, enemyPosition) and self.distancer.getDistance(cell, enemyPosition) < 3:
                #print("triggered")
                rewardTable.rewardDistribution(cell, ghostAttackReward * (3 - self.distancer.getDistance(cell, enemyPosition)))
        # if carrying many food, better to back home soon
        CarryFoodFomeReward = backhomeReward * (max(min(currentState.numCarrying, 10), 0)**1.2) / 15.84
        for i in self.homeBoundaryCells:
            self.updateReward(grid, rewardTable, CarryFoodFomeReward, currentPosition, i)
    #print(self.getDistanceHome(currentPosition))
    for pelletPos in self.getCapsules(gameState):
        self.updateReward(grid, rewardTable, 1, currentPosition, pelletPos)







    timeLeft = gameState.data.timeleft // 4
    if (len(foodList) <= 2) or (timeLeft < 20) or (timeLeft < (self.getDistanceHome(currentPosition) + 10)):
        for i in self.homeBoundaryCells:
            self.updateReward(grid, rewardTable, backhomeReward, currentPosition, i)
    if currentState.numCarrying > 5 and self.getDistanceHome(currentPosition) < 5:
        for i in self.homeBoundaryCells:
            self.updateReward(grid, rewardTable, backhomeReward, currentPosition, i)
    #print("reward assign time" + str(time.time()-start_time))
    evaluator = ValueIterationCalculator(rewardTable, 0.75, 10)
    while(time.time()-start_time < 0.1):
        #print("more iteration")
        evaluator = ValueIterationCalculator(evaluator.RewardTable, 0.75, 10)

 # never go to grip with enemy
    for i in self.getOpponents(gameState):
        enemyState = gameState.getAgentState(i)
        if enemyState.getPosition():
            if self.distancer.getDistance(currentPosition, enemyState.getPosition()) <= 3:
                evaluator.RewardTable.assignReward(enemyState.getPosition(), -50)

    #print(rewardTable.rewards)
    bestAction = evaluator.selectAction((int(currentPosition[0]), int(currentPosition[1])))
    #print('currentLocation' + str((int(currentPosition[0]), int(currentPosition[1]))))
    #print(rewardTable.rewards)
    print("total time" + str(time.time()-start_time))
    # if(time.time()-start_time > 0.25):
    #     print('too long')
    return bestAction


  def _getGrid(self, x_min, y_min, x_max, y_max):
        x_min = int(max(1, x_min))
        x_max = int(min(self._maxx, x_max))
        y_min = int(max(1, y_min))
        y_max = int(min(self._maxy, y_max))

        all_cells = set()
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                all_cells.add((x, y))
        return all_cells.difference(self.walls)

  def atHome(self, cell):
        x, _ = cell
        return self.start[0] <= x <= self.homeXBoundary or self.homeXBoundary <= x <= self.start[0]

  def getDistanceHome(self, pos):
        x, _ = pos
        if (self.homeBoundaryCells[0][0] - x) * self.sign > 0:
            return 0
        distances = [self.distancer.getDistance(pos, cell) for cell in self.homeBoundaryCells]
        return min(distances)



  def updateReward(self, grid, RewardTable, RewardItem, currentPosition, targetPosition):
      for i in grid:
          if self.distancer.getDistance(i, targetPosition) <= self.distancer.getDistance(currentPosition, targetPosition):
            reward = RewardItem / max(float(self.distancer.getDistance(i, targetPosition)), 0.2) # to avoid distance == 0
            RewardTable.assignReward(i, reward)


  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}


class NodesRelation:
    def __init__(self, father, child, gvalue):
        self.father = father
        self.child = child
        self.g = gvalue + 1

class AstarAgent(ReflexCaptureAgent):
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.walls =  set(gameState.data.layout.walls.asList())
        self._maxx = max([item[0] for item in self.walls])
        self._maxy = max([item[1] for item in self.walls])
        self.sign = 1 if gameState.isOnRedTeam(self.index) else -1
        self.homeXBoundary = self.start[0] + ((self._maxx // 2 - 1) * self.sign)
        cells = [(self.homeXBoundary, y) for y in range(1, self._maxy)]
        self.homeBoundaryCells = [item for item in cells if item not in self.walls]
        available_states = self._getGrid(1, 1, self._maxx, self._maxy)
        self.available_actions = util.Counter()
        for cell in available_states:
            x, y = cell
            if (x - 1, y) in available_states:
                self.available_actions[cell] += 1
            if (x + 1, y) in available_states:
                self.available_actions[cell] += 1
            if (x, y - 1) in available_states:
                self.available_actions[cell] += 1
            if (x, y + 1) in available_states:
                self.available_actions[cell] += 1
    def chooseAction(self, gameState):
        #start_time = time.time()
        #print(self.getFoodYouAreDefending(gameState).asList())
        start_time = time.time()
        global Mode
        global foodLostTimer
        currentState = gameState.getAgentState(self.index)
        currentPosition = currentState.getPosition()
        enemies = []
        enemyNearby = False;
        for i in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(i)
            #print(enemyState)
            #print(i)
            if enemyState.getPosition():
                if self.distancer.getDistance(currentPosition, enemyState.getPosition()) <= 5:
                    #print(self.distancer.getDistance(currentPosition, enemyState.getPosition()))
                    enemyNearby = True
                    enemies.append((enemyState, enemyState.getPosition()))
        if(Mode == "D"):
            #print("D")
            action = self.computeActionFromQValues(gameState)
            if foodLostTimer > 50 and enemyNearby == False:
                Mode = "A"
            #print(self.getGoal(gameState))
            goal_grid = self.getGoal(gameState)
            if(goal_grid!= None):
                # print("current")
                # print(currentPosition)
                # print("goal")
                # print(goal_grid)
                # print("next")
                # print(self.AStarSearch(currentPosition, goal_grid))
                for AStaraction in gameState.getLegalActions(self.index):
                    successor = self.getSuccessor(gameState, AStaraction)
                    #print(successor)
                    if successor.getAgentState(self.index).getPosition() == self.AStarSearch(currentPosition, goal_grid):
                        #print("total time" + str(time.time()-start_time))
                        return AStaraction
            #print("total time" + str(time.time()-start_time))
            return action
        else:
            #print("A")
            if self.atHome(currentPosition) and (foodLostTimer < 50 or enemyNearby == True):
                Mode = "D"
            #print("total time" + str(time.time()-start_time))
            return self.chooseAttackAction(gameState)
        #print("total time" + str(time.time()-start_time))
        #return action



    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        CurrentState = successor.getAgentState(self.index)
        currentPosition = CurrentState.getPosition()

        if CurrentState.isPacman:
            features['defending'] = 0
        else:
            features['defending'] = 1

        features['distanceHome'] = self.getDistanceHomeBoundary(currentPosition)
        #features['distanceHome'] = 0
        #print(features['distanceHome'])

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        # TODO Why sometimes cannot see enemy ???
        pacman_enemies = [a for a in enemies if a.isPacman and a.getPosition() != None]
        #print('enemy')
        #print(pacman_enemies)
        #print(self.getOpponents(successor))
        features['num_of_pacman_enemies'] = len(pacman_enemies)

        if len(pacman_enemies) > 0:
            dists = [self.getMazeDistance(currentPosition, p.getPosition()) for p in pacman_enemies]
            features['closest_enemy_dist'] = min(dists)
        #print(features['closest_enemy_dist'])
        if action == Directions.STOP: features['stop'] = 1

        # try to make the agent not walk all map around
        reverse_act = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == reverse_act: features['reverse'] = 1

        global newFoodList
        newFoodList = self.getFoodYouAreDefending(gameState).asList()
        global lastFoodList
        if(len(lastFoodList) == 0):
            lastFoodList = self.getFoodYouAreDefending(gameState).asList()
        # not all food are eaten
        if(len(lastFoodList)!=0):
            global foodMissed
            localMissingFoodList = []
            localMissingFoodList = list(set(lastFoodList)-set(newFoodList))
            global foodLostTimer
            if(len(localMissingFoodList) == 0): # no food missed
                foodLostTimer += 1
            else:#some food missed
                #print("new food lost")
                foodLostTimer = 0
                foodMissed = localMissingFoodList[0]
                lastFoodList = self.getFoodYouAreDefending(gameState).asList()


            if(foodLostTimer <= 25 and foodMissed is not None):
                lostfoodgrid = foodMissed
                #print("seek for the lost food")
                #print(foodMissed)
                features['missingfooddistance'] = self.distancer.getDistance(currentPosition, lostfoodgrid)
            else:
                foodMissed = None
                features['missingfooddistance'] = 0
        return features

    def getWeights(self, gameState, action):
        return {'num_of_pacman_enemies': -1000, 'distanceHome':-2, 'defending': 100, 'closest_enemy_dist': -10, 'stop': -100, 'reverse': -2, 'missingfooddistance': -1}

    def weightedActionValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getDistanceHomeBoundary(self, pos):
          x, _ = pos
          if (self.homeBoundaryCells[0][0] - x) * self.sign > 0:
              distances = [self.distancer.getDistance(pos, cell) for cell in self.homeBoundaryCells]
              return min(distances)
          return 0

    def computeActionFromQValues(self, state):
        """
          compute the next best action based on the Q values of the baseline Agent
        """
        bestValue = -999999
        bestActions = None
        for action in state.getLegalActions(self.index):
            value = self.weightedActionValue(state, action)
            if value == bestValue:
                bestActions.append(action)
            elif value > bestValue:
                bestActions = [action]
                bestValue = value
        if bestActions == None:
            return Directions.STOP
        #print(bestActions)
        return random.choice(bestActions)

    def getGoal(self, gameState):
        CurrentState = gameState.getAgentState(self.index)
        currentPosition = CurrentState.getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [b for b in enemies if b.isPacman and b.getPosition() != None]
        dists = []
        global enemyLastPosition
        global enemyTimer
        if len(invaders) > 0:
            dists = [self.getMazeDistance(currentPosition, j.getPosition()) for j in invaders]
            for m in range(len(invaders)):
                if self.getMazeDistance(currentPosition, invaders[m].getPosition()) == min(dists):
                    #print("enemy")
                    enemyLastPosition = invaders[m].getPosition()
                    enemyTimer = 0
                    return invaders[m].getPosition()
        global foodMissed
        global foodLostTimer
        if foodMissed != None and foodLostTimer <= 25:
            #print("food")
            return foodMissed
        if enemyTimer < 3:
            enemyTimer += 1
            #print("enemy last location")
            return enemyLastPosition
        return None

    def aStarHeuristic(self, selfPostion, goalPosition):
        return abs(selfPostion[0] - goalPosition[0]) + abs(selfPostion[1] + goalPosition[1])

    def AStarSearch(self, selfPosition, goalPosition):
        available_states = self._getGrid(1, 1, self._maxx, self._maxy)
        #print(available_states)
        open = PriorityQueue()
        closed = []
        nodesRelationList = []
        open.push(selfPosition, 0 + self.aStarHeuristic(selfPosition, goalPosition))
        nodesRelationList.append(NodesRelation(None, selfPosition, 0))
        while open.isEmpty() == False:
            node = open.pop()
            #print("node poped")
            #print(node)
            if node not in closed:
                closed.append(node)
                if node == goalPosition:
                    #print("goal position")
                    #print(node)
                    return self.getRoot(node, nodesRelationList)
                if (node[0] - 1, node[1]) in available_states:
                    nodesRelationList.append(NodesRelation(node, (node[0] - 1, node[1]), self.getGValue(node, nodesRelationList)))
                    open.push((node[0] - 1, node[1]), self.getGValue(node, nodesRelationList) + self.aStarHeuristic(node, goalPosition))

                if (node[0] + 1, node[1]) in available_states:
                    nodesRelationList.append(NodesRelation(node, (node[0] + 1, node[1]), self.getGValue(node, nodesRelationList)))
                    open.push((node[0] + 1, node[1]), self.getGValue(node, nodesRelationList) + self.aStarHeuristic(node, goalPosition))
                if (node[0], node[1] - 1 ) in available_states:
                    nodesRelationList.append(NodesRelation(node, (node[0], node[1] -1 ), self.getGValue(node, nodesRelationList)))
                    open.push((node[0], node[1] -1 ), self.getGValue(node, nodesRelationList) + self.aStarHeuristic(node, goalPosition))
                if (node[0], node[1] + 1) in available_states:
                    nodesRelationList.append(NodesRelation(node, (node[0], node[1] + 1), self.getGValue(node, nodesRelationList)))
                    open.push((node[0], node[1] + 1), self.getGValue(node, nodesRelationList) + self.aStarHeuristic(node, goalPosition))


    def getFather(self, node, nodesRelationList):
        for i in range(len(nodesRelationList)):
            if node == nodesRelationList[i].child:
                return nodesRelationList[i].father
        #return None

    def getRoot(self, node, nodesRelationList):
        lastnode = node
        newnode = node
        while self.getFather(newnode, nodesRelationList) != None:
            lastnode = newnode
            newnode = self.getFather(newnode, nodesRelationList)
        return lastnode

    def getGValue(self, node, nodesRelationList):
        for i in range(len(nodesRelationList)):
            if node == nodesRelationList[i].child:
                return nodesRelationList[i].g

    # Attack model
    def chooseAttackAction(self, gameState):
      #print(self.getOpponents(gameState))
      start_time = time.time()

      start = time.time()
      area_range = 6
      backhomeReward = 1.0
      getFoodReward = 0.15
      potentialDeadEndReward = -0.1
      ghostAttackReward = -1.5
      min_enemy_distance = 5
      currentState = gameState.getAgentState(self.index)
      currentPosition = currentState.getPosition()
      x, y = currentPosition

      # get the states needed to be calculate
      grid = self._getGrid(x - area_range, y - area_range, x + area_range, y + area_range)
      # retur the search area cycle
      grid = {cell for cell in grid if self.distancer.getDistance(currentPosition, cell) <= area_range}

      rewardTable = RewardTable(currentState, grid)

      # reward for food
      foodList =  self.getFood(gameState).asList()
      if len(foodList) > 2: #still hold less than 18 food
          for food in foodList:
              self.updateReward(grid, rewardTable, getFoodReward, currentPosition, food)

      # Initialize enemy information
      enemies = []
      enemyNearby = False;
      for i in self.getOpponents(gameState):
          enemyState = gameState.getAgentState(i)
          #print(enemyState)
          #print(i)
          if enemyState.getPosition():
              if self.distancer.getDistance(currentPosition, enemyState.getPosition()) <= min_enemy_distance:
                  #print(self.distancer.getDistance(currentPosition, enemyState.getPosition()))
                  enemyNearby = True
                  enemies.append((enemyState, enemyState.getPosition()))

      # TODO handle multiple enemies
      if enemyNearby:
          #print(min([item.scaredTimer for item, _ in enemies]))
          closestEnemyDistance = 5
          for enemyState, enemyPosition in enemies:
              if enemyState.scaredTimer > 2:
                  continue
              enemy_distance = self.distancer.getDistance(currentPosition, enemyPosition)
              reward = ghostAttackReward * (area_range + 1.0 - enemy_distance)
              #print(reward)
              # different reward assign method here
              rewardTable.assignReward(enemyPosition, reward)
              closestEnemyDistance = min(closestEnemyDistance, enemy_distance)
          for cell in grid:
              if self.atHome(cell):
                  continue
              if min([item.scaredTimer for item, _ in enemies]) > 8:
                  continue
              if self.getDistanceHome(cell) > self.getDistanceHome(currentPosition):
                  rewardTable.rewardDistribution(cell, potentialDeadEndReward)
              if self.available_actions[cell] == 1 and cell not in foodList:
                  rewardTable.assignReward(cell, potentialDeadEndReward * 5)
              if self.distancer.getDistance(currentPosition, enemyPosition) < 3 and self.available_actions[cell] == 1:
                  #print("dead trap!")
                  rewardTable.assignReward(cell, potentialDeadEndReward * 10)
              if self.distancer.getDistance(currentPosition, enemyPosition) < 2 * self.distancer.getDistance(cell, enemyPosition) + 1 and self.available_actions[cell] == 1:
                  #print("dead trap!")
                  rewardTable.assignReward(cell, potentialDeadEndReward * 10)
              if self.distancer.getDistance(currentPosition, enemyPosition) > self.distancer.getDistance(cell, enemyPosition) and self.distancer.getDistance(cell, enemyPosition) < 3:
                  #print("triggered")
                  rewardTable.rewardDistribution(cell, ghostAttackReward * (3 - self.distancer.getDistance(cell, enemyPosition)))
          # if carrying many food, better to back home soon
          CarryFoodFomeReward = backhomeReward * (max(min(currentState.numCarrying, 10), 0)**1.2) / 15.84
          for i in self.homeBoundaryCells:
              self.updateReward(grid, rewardTable, CarryFoodFomeReward, currentPosition, i)
      #print(self.getDistanceHome(currentPosition))
      for pelletPos in self.getCapsules(gameState):
          self.updateReward(grid, rewardTable, 1, currentPosition, pelletPos)







      timeLeft = gameState.data.timeleft // 4
      if (len(foodList) <= 2) or (timeLeft < 20) or (timeLeft < (self.getDistanceHome(currentPosition) + 10)):
          for i in self.homeBoundaryCells:
              self.updateReward(grid, rewardTable, backhomeReward, currentPosition, i)
      if currentState.numCarrying > 5 and self.getDistanceHome(currentPosition) < 5:
          for i in self.homeBoundaryCells:
              self.updateReward(grid, rewardTable, backhomeReward, currentPosition, i)
      #print("reward assign time" + str(time.time()-start_time))
      evaluator = ValueIterationCalculator(rewardTable, 0.75, 10)
      while(time.time()-start_time < 0.1):
          #print("more iteration")
          evaluator = ValueIterationCalculator(evaluator.RewardTable, 0.75, 10)

      # never go to grip with enemy
      for i in self.getOpponents(gameState):
          enemyState = gameState.getAgentState(i)
          if enemyState.getPosition():
              if self.distancer.getDistance(currentPosition, enemyState.getPosition()) <= 3:
                  evaluator.RewardTable.assignReward(enemyState.getPosition(), -50)


      #print(rewardTable.rewards)
      bestAction = evaluator.selectAction((int(currentPosition[0]), int(currentPosition[1])))
      #print('currentLocation' + str((int(currentPosition[0]), int(currentPosition[1]))))
      #print(rewardTable.rewards)
      #print("total time" + str(time.time()-start_time))
      # if(time.time()-start_time > 0.25):
      #     print('too long')
      return bestAction


    def _getGrid(self, x_min, y_min, x_max, y_max):
          x_min = int(max(1, x_min))
          x_max = int(min(self._maxx, x_max))
          y_min = int(max(1, y_min))
          y_max = int(min(self._maxy, y_max))

          all_cells = set()
          for x in range(x_min, x_max + 1):
              for y in range(y_min, y_max + 1):
                  all_cells.add((x, y))
          return all_cells.difference(self.walls)

    def atHome(self, cell):
          x, _ = cell
          return self.start[0] <= x <= self.homeXBoundary or self.homeXBoundary <= x <= self.start[0]

    def getDistanceHome(self, pos):
          x, _ = pos
          if (self.homeBoundaryCells[0][0] - x) * self.sign > 0:
              return 0
          distances = [self.distancer.getDistance(pos, cell) for cell in self.homeBoundaryCells]
          return min(distances)



    def updateReward(self, grid, RewardTable, RewardItem, currentPosition, targetPosition):
        for i in grid:
            if self.distancer.getDistance(i, targetPosition) <= self.distancer.getDistance(currentPosition, targetPosition):
              reward = RewardItem / max(float(self.distancer.getDistance(i, targetPosition)), 0.2) # to avoid distance == 0
              RewardTable.assignReward(i, reward)

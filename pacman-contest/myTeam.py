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

# The flag for Agent mode, "D" for Defensive and "A" for Attack
Mode = "D"
# Record the last missed food postion
foodMissed = None
# Record the food list in last iteration
lastFoodList = []
# record the current food list
newFoodList = []
# the time since last food lost
foodLostTimer = 0
# the last enemy found postion
enemyLastPosition = None
# the time since last enemy show up
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

# this is a reward table to hold the rewards for the game
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

    # this function will assign reward to one grid together with its neighbours, the range is 5
    def rewardDistribution(self, state, reward):
        x, y = state
        self.rewards[state] += reward
        for i in range(-4, 5):
            for j in range(-4, 5):
                if (x+i, y+j) in list(self.states):
                    self.rewards[(x+i, y+j)] += reward / math.exp(abs(i)+abs(j))

    # this function will assign reward to one grid
    def assignReward(self, state, r):
        self.rewards[state] += r
        #self.rewardDistribution(state, r)

    # get reward details
    def getReward(self, state):
        return self.rewards[state]

    # get all current states
    def getStates(self):
        return list(self.states)

    # get all available actions
    def availableActions(self, state):
        availableActionsList = []
        for tuple in self.actionsDict.keys():
            if tuple[0] == state:
                availableActionsList.append(tuple[1])
        return availableActionsList

    # get the location after some action
    def getNextState(self, state, action):
        # return [self.actionsDict[(state, action)],]
        for tuple in self.actionsDict.keys():
            if str(tuple[0]) == str(state) and str(tuple[1]) == str(action):
                return [self.actionsDict.get(tuple)]




# this class update the state value using Value Iteration
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

    # Get the QSA table
    def getQSA(self, state, action):
        Q = 0
        if self.RewardTable.getNextState(state, action) != None:
            for newstate in self.RewardTable.getNextState(state, action):
                reward = self.RewardTable.getReward(newstate)
                Q = reward + self.discount * self.StateValues[newstate]
        return Q

    # select the best action based on current state values
    def selectAction(self, state):
        values = util.Counter()
        best_action = ''
        best_value = -10000
        for action in self.RewardTable.availableActions(state):
            values[action] = self.getQSA(state, action)
            if values[action] > best_value:
                best_value = values[action]
                best_action = action
        return best_action



# this is the Model-based MDP agent, used for offensive behaviours
class ReflexCaptureAgent(CaptureAgent):
  # initialize function
  def registerInitialState(self, gameState):
    # the start state
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    # identify the red/blue
    self.color_side = 1 if gameState.isOnRedTeam(self.index) else -1

    # get enemy index
    self.enemy = self.getOpponents(gameState)

    # collect maza structure information
    self.walls =  set(gameState.data.layout.walls.asList())
    self.horiza_boundary = max([item[0] for item in self.walls])
    self.virtical_boundary = max([item[1] for item in self.walls])

    # calculate the home boundary
    self.home_boundary = self.start[0] + ((self.horiza_boundary // 2 - 1) * self.color_side)
    cells = [(self.home_boundary, y) for y in range(1, self.virtical_boundary)]
    self.home_boundary_grids_list = [item for item in cells if item not in self.walls]

    # get all possible states based on map information
    available_states = self.get_map_grids_info(1, 1, self.horiza_boundary, self.virtical_boundary)
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


  # this function will assign reward based on different situation and select the best one using Value Iteration
  def chooseAction(self, gameState):
    #print(self.getOpponents(gameState))
    start_time = time.time()
    # record the time
    start = time.time()
    # the negative when pacman is attacked by ghost
    ghostAttackReward = -1.5
    # the reward for home when there is enemy
    backhomeReward = 1.0
    # the reward for get some food home
    getFoodReward = 0.15
    # the reward for get the capsule
    capsuleReward = 1
    # the negative for some possible dead-end road
    potentialDeadEndReward = -0.1
    # signle grid negative reward for enemy normal ghost position
    enemyNormalGhostReard = -50
    # the search range
    area_range = 5
    # minimal enemy distance that the pacman will care
    min_enemy_distance = 5

    # current location and game state
    currentState = gameState.getAgentState(self.index)
    currentPosition = currentState.getPosition()
    x, y = currentPosition

    # get the states needed to be calculate
    grid = self.get_map_grids_info(x - area_range, y - area_range, x + area_range, y + area_range)
    # retur the search area cycle
    grid = {cell for cell in grid if self.distancer.getDistance(currentPosition, cell) <= area_range}

    # get the reward table
    rewardTable = RewardTable(currentState, grid)

    # reward for food
    foodList =  self.getFood(gameState).asList()
    self.assignFoodReward(gameState, rewardTable, grid, getFoodReward, currentPosition)

    # reward for capsule
    self.assignCapsuleReward(gameState, rewardTable, grid, capsuleReward, currentPosition)

    # calculate the time left in case cannot carry food home
    self.assignTimeoutReward(gameState, rewardTable, grid, backhomeReward, currentPosition)

    # assign back home reward if the agnet is carrying food and far from home
    self.assignTooFarFromHomeReward(gameState, rewardTable, grid, backhomeReward, currentState, currentPosition)

    # Initialize enemy information and check out whether there is any enemy around
    enemies = []
    enemyNearby = self.checkoutEnemyAround(gameState, enemies, currentPosition, min_enemy_distance);

    # handle the situation when there is some enemy around
    if enemyNearby:
        self.assignEnemyAroundGridsReward(gameState, rewardTable, grid, potentialDeadEndReward, ghostAttackReward, backhomeReward, currentState, currentPosition, enemies, area_range)

    # input the total rewards and use MDP calculator to calculate the state values
    evaluator = ValueIterationCalculator(rewardTable, 0.75, 10)
    # do more calculation if there is some time left
    while(time.time()-start_time < 0.1):
        evaluator = ValueIterationCalculator(evaluator.RewardTable, 0.75, 10)

    # never go to grip with enemy
    for i in self.getOpponents(gameState):
        enemyState = gameState.getAgentState(i)
        if enemyState.getPosition():
            if self.distancer.getDistance(currentPosition, enemyState.getPosition()) <= 3:
                evaluator.RewardTable.assignReward(enemyState.getPosition(), enemyNormalGhostReard)

    #print(rewardTable.rewards)
    # select the best action result from MDP calculator
    bestAction = evaluator.selectAction((int(currentPosition[0]), int(currentPosition[1])))
    return bestAction


  # assign reward for grids with enemy around
  def assignEnemyAroundGridsReward(self, gameState, rewardTable, grid, potentialDeadEndReward, ghostAttackReward, backhomeReward, currentState,currentPosition, enemies, area_range):
      closestEnemyDistance = 5
      foodList =  self.getFood(gameState).asList()
      # assign negative reward for enemy to avoid the enemy
      for enemyState, enemyPosition in enemies:
          if enemyState.scaredTimer > 2:
              continue
          enemy_distance = self.distancer.getDistance(currentPosition, enemyPosition)
          reward = ghostAttackReward * (area_range + 1.0 - enemy_distance)
          rewardTable.assignReward(enemyPosition, reward)
          closestEnemyDistance = min(closestEnemyDistance, enemy_distance)
      # some detailed situations
      for cell in grid:
          # if the agent at home, wont care the enemy
          if self.atHome(cell):
              continue
          # if the enemy will be scared for some time, wont care
          if min([item.scaredTimer for item, _ in enemies]) > 8:
              #print(min([item.scaredTimer for item, _ in enemies]))
              continue
          # if go further from home boundary, some negative reward
          if self.getDistanceHome(cell) > self.getDistanceHome(currentPosition):
              rewardTable.rewardDistribution(cell, potentialDeadEndReward)
          # if only one action available, means this is a dead-end road, assign negative reward
          if self.available_actions[cell] == 1 and cell not in foodList:
              rewardTable.assignReward(cell, potentialDeadEndReward * 5)

          # if the enemy is closed, won't get food in some dead-end road
          if self.distancer.getDistance(currentPosition, enemyPosition) < 3 and self.available_actions[cell] == 1:
              rewardTable.assignReward(cell, potentialDeadEndReward * 10)
          # calculate whether some dead-end road food is safe to collect
          if self.distancer.getDistance(currentPosition, enemyPosition) < 2 * self.distancer.getDistance(cell, enemyPosition) + 1 and self.available_actions[cell] == 1:
              rewardTable.assignReward(cell, potentialDeadEndReward * 10)
          # if the enemy is closed, wont go deeper in enemy area
          if self.distancer.getDistance(currentPosition, enemyPosition) > self.distancer.getDistance(cell, enemyPosition) and self.distancer.getDistance(cell, enemyPosition) < 3:
              rewardTable.rewardDistribution(cell, ghostAttackReward * (3 - self.distancer.getDistance(cell, enemyPosition)))
      # if carrying many food, better to back home soon
      CarryFoodFomeReward = backhomeReward * (max(min(currentState.numCarrying, 10), 0)**1.2) / 15.84
      for i in self.home_boundary_grids_list:
          self.updateReward(grid, rewardTable, CarryFoodFomeReward, currentPosition, i)

  # get all grids except for walls
  def get_map_grids_info(self, x_min, y_min, x_max, y_max):
        x_min = int(max(1, x_min))
        x_max = int(min(self.horiza_boundary, x_max))
        y_min = int(max(1, y_min))
        y_max = int(min(self.virtical_boundary, y_max))

        all_cells = set()
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                all_cells.add((x, y))
        return all_cells.difference(self.walls)

  # assign reward on food grids
  def assignFoodReward(self, gameState, rewardTable, grid, getFoodReward, currentPosition):
      foodList =  self.getFood(gameState).asList()
      if len(foodList) > 2: # still hold less than 18 food
          for food in foodList:
              self.updateReward(grid, rewardTable, getFoodReward, currentPosition, food)

  # assign reward on capsule grids
  def assignCapsuleReward(self, gameState, rewardTable, grid, capsuleReward,currentPosition):
      for pelletPosition in self.getCapsules(gameState):
          self.updateReward(grid, rewardTable, capsuleReward, currentPosition, pelletPosition)

  # assign reward to take agent back home, if it's carrying food and far from home boundary
  def assignTooFarFromHomeReward(self, gameState, rewardTable, grid, backhomeReward, currentState, currentPosition):
      if currentState.numCarrying > 5 and self.getDistanceHome(currentPosition) < 5:
          for i in self.home_boundary_grids_list:
              self.updateReward(grid, rewardTable, backhomeReward, currentPosition, i)

  # calculate the time left in case cannot carry food home
  def assignTimeoutReward(self, gameState, rewardTable, grid, backhomeReward, currentPosition):
      foodList =  self.getFood(gameState).asList()
      timeLeft = gameState.data.timeleft // 4
      if (len(foodList) <= 2) or (timeLeft < 20) or (timeLeft < (self.getDistanceHome(currentPosition) + 10)):
          for i in self.home_boundary_grids_list:
              self.updateReward(grid, rewardTable, backhomeReward, currentPosition, i)

  # check whether the agent position is at home area or not
  def atHome(self, cell):
      x, _ = cell
      return self.start[0] <= x <= self.home_boundary or self.home_boundary <= x <= self.start[0]

  # find out whether there is any enemy around
  def checkoutEnemyAround(self, gameState, enemies, currentPosition, min_enemy_distance):
      enemyNearby = False;
      for i in self.getOpponents(gameState):
          enemyState = gameState.getAgentState(i)
          if enemyState.getPosition():
              # detect enemy distance smaller than 5, change enemyNearby as true
              if self.distancer.getDistance(currentPosition, enemyState.getPosition()) <= min_enemy_distance:
                  #print(self.distancer.getDistance(currentPosition, enemyState.getPosition()))
                  enemyNearby = True
                  enemies.append((enemyState, enemyState.getPosition()))
      return enemyNearby

  # calculate the distance from home, return 0 if it's at home, else, will return the direct distance
  def getDistanceHome(self, position):
      x, _ = position
      if (self.home_boundary_grids_list[0][0] - x) * self.color_side > 0:
          return 0
      distances = [self.distancer.getDistance(position, cell) for cell in self.home_boundary_grids_list]
      return min(distances)


  # update the reward table
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

# this agent will call the chooseaction function in ReflexCaptureAgent to select best offensive action
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

# this class is used to store the relationship between nodes, used in A star search
class NodesRelation:
    def __init__(self, father, child, gvalue):
        self.father = father
        self.child = child
        self.g = gvalue + 1

# the a start agent
class AstarAgent(ReflexCaptureAgent):

    # initialize, same with the initialise function in MDP agent
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.walls =  set(gameState.data.layout.walls.asList())
        self.horiza_boundary = max([item[0] for item in self.walls])
        self.virtical_boundary = max([item[1] for item in self.walls])
        self.color_side = 1 if gameState.isOnRedTeam(self.index) else -1
        self.home_boundary = self.start[0] + ((self.horiza_boundary // 2 - 1) * self.color_side)
        cells = [(self.home_boundary, y) for y in range(1, self.virtical_boundary)]
        self.home_boundary_grids_list = [item for item in cells if item not in self.walls]
        available_states = self.get_map_grids_info(1, 1, self.horiza_boundary, self.virtical_boundary)
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

    # will select the best action, will switch between defensive agent and offensive agent
    def chooseAction(self, gameState):

        start_time = time.time()
        global Mode
        global foodLostTimer
        currentState = gameState.getAgentState(self.index)
        currentPosition = currentState.getPosition()
        enemies = []
        enemyNearby = False;
        # check whether there is some enemy around
        for i in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(i)
            if enemyState.getPosition():
                if self.distancer.getDistance(currentPosition, enemyState.getPosition()) <= 5:
                    enemyNearby = True
                    enemies.append((enemyState, enemyState.getPosition()))
        # if in defensive mode currently
        if(Mode == "D"):
            #print("D")
            action = self.computeActionFromQValues(gameState)
            goal_grid = self.getGoal(gameState)
            # if nothing found for a while, switch into offensive mode
            if foodLostTimer > 60 and enemyNearby == False:
                Mode = "A"
            # handle the situation when the ghost is scared
            if self.isScared(gameState, self.index):
                # if some enemy (pacman) around, keep distance from it
                if goal_grid is not None:
                    if gameState.data.agentStates[self.index].scaredTimer * 2 < self.distancer.getDistance(currentPosition, goal_grid):
                        for AStaraction in gameState.getLegalActions(self.index):
                            successor = self.getSuccessor(gameState, AStaraction)
                            #print(successor)
                            if successor.getAgentState(self.index).getPosition() == self.AStarSearch(currentPosition, goal_grid):
                                return AStaraction
                return self.computeScariedActionFromQValues(gameState)

            # if not in scared mode, using a star tracking for enemy pacman
            if(goal_grid!= None):
                for AStaraction in gameState.getLegalActions(self.index):
                    successor = self.getSuccessor(gameState, AStaraction)
                    if successor.getAgentState(self.index).getPosition() == self.AStarSearch(currentPosition, goal_grid):
                        return AStaraction
            return action
        # if the agent is in offensive mode
        else:
            # if find some enemy at home, switch to defensive mode
            if self.atHome(currentPosition) and (foodLostTimer < 50 or enemyNearby == True):
                Mode = "D"
            # choose attack actions
            defenceAttackAction = self.chooseAttackAction(gameState)
            return defenceAttackAction


    # the different feature values
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        CurrentState = successor.getAgentState(self.index)
        currentPosition = CurrentState.getPosition()

        # whether the agent is a pacman
        if CurrentState.isPacman:
            features['defending'] = 0
        else:
            features['defending'] = 1

        # the distance to home
        features['distanceHome'] = self.getDistanceHomeBoundary(currentPosition)

        # find out how many enemies around
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        pacman_enemies = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['num_of_pacman_enemies'] = len(pacman_enemies)

        # find the closest enemy
        if len(pacman_enemies) > 0:
            dists = [self.getMazeDistance(currentPosition, p.getPosition()) for p in pacman_enemies]
            features['closest_enemy_dist'] = min(dists)
        if action == Directions.STOP: features['stop'] = 1

        # try to make the agent not walk all map around
        reverse_act = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == reverse_act: features['reverse'] = 1

        # update the food state to find out if any food lost
        global newFoodList
        newFoodList = self.getFoodYouAreDefending(gameState).asList()
        global lastFoodList

        # initialize food list
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
                foodLostTimer = 0
                foodMissed = localMissingFoodList[0]
                lastFoodList = self.getFoodYouAreDefending(gameState).asList()

            # if the food lost recently
            if(foodLostTimer <= 25 and foodMissed is not None):
                lostfoodgrid = foodMissed
                features['missingfooddistance'] = self.distancer.getDistance(currentPosition, lostfoodgrid)
            # if the food has been lost for a long time
            else:
                foodMissed = None
                features['missingfooddistance'] = 0
        return features

    # the weight of different factors
    def getWeights(self, gameState, action):
        return {'num_of_pacman_enemies': -1000, 'distanceHome':-2, 'defending': 100, 'closest_enemy_dist': -10, 'stop': -100, 'reverse': -2, 'missingfooddistance': -1}

    # the weight of different factors when the agent is scared
    def getScariedWeights(self, gameState, action):
        return {'num_of_pacman_enemies': -1000, 'distanceHome':-2, 'defending': 100, 'closest_enemy_dist': 10, 'stop': -100, 'reverse': -2, 'missingfooddistance': 1}

    # get the ghost action
    def weightedActionValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    # get the weight whne the ghost is scared
    def weightedScariedActionValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getScariedWeights(gameState, action)
        return features * weights

    # get the distance to home boundary
    def getDistanceHomeBoundary(self, pos):
          x, _ = pos
          if (self.home_boundary_grids_list[0][0] - x) * self.color_side > 0:
              distances = [self.distancer.getDistance(pos, cell) for cell in self.home_boundary_grids_list]
              return min(distances)
          return 0

    # get the action based on best Q value
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

    # get action when the agent is scared
    def computeScariedActionFromQValues(self, state):
        """
          compute the next best action based on the Q values of the baseline Agent
        """
        bestValue = -999999
        bestActions = None
        for action in state.getLegalActions(self.index):
            value = self.weightedScariedActionValue(state, action)
            if value == bestValue:
                bestActions.append(action)
            elif value > bestValue:
                bestActions = [action]
                bestValue = value
        if bestActions == None:
            return Directions.STOP
        #print(bestActions)
        return random.choice(bestActions)

    # get the goal for A star search. will return either enemy location or last missed food location
    def getGoal(self, gameState):
        CurrentState = gameState.getAgentState(self.index)
        currentPosition = CurrentState.getPosition()
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [b for b in enemies if b.isPacman and b.getPosition() != None]
        dists = []
        global enemyLastPosition
        global enemyTimer
        # if there is some enemy, return the enemy position and record it in global variable
        if len(invaders) > 0:
            dists = [self.getMazeDistance(currentPosition, j.getPosition()) for j in invaders]
            for m in range(len(invaders)):
                if self.getMazeDistance(currentPosition, invaders[m].getPosition()) == min(dists):
                    enemyLastPosition = invaders[m].getPosition()
                    enemyTimer = 0
                    return invaders[m].getPosition()
        global foodMissed
        global foodLostTimer
        # if no enemy, check whether some food lost, if so, set it as goal and record location and time in global variables
        if foodMissed != None and foodLostTimer <= 25:
            #print("food")
            return foodMissed
        if enemyTimer < 3:
            enemyTimer += 1
            #print("enemy last location")
            return enemyLastPosition
        return None

    # manhatten distance as the heuristic function
    def aStarHeuristic(self, selfPostion, goalPosition):
        return abs(selfPostion[0] - goalPosition[0]) + abs(selfPostion[1] + goalPosition[1])

    # the A star search function
    def AStarSearch(self, selfPosition, goalPosition):
        # get all available states
        available_states = self.get_map_grids_info(1, 1, self.horiza_boundary, self.virtical_boundary)
        # open list
        open = PriorityQueue()
        # closed list
        closed = []
        # node relationship list
        nodesRelationList = []
        open.push(selfPosition, 0 + self.aStarHeuristic(selfPosition, goalPosition))
        nodesRelationList.append(NodesRelation(None, selfPosition, 0))
        while open.isEmpty() == False:
            node = open.pop()
            if node not in closed:
                closed.append(node)
                # if goal reached
                if node == goalPosition:
                    return self.getRoot(node, nodesRelationList)
                # expend and search, add nodes into open list and nodes relationship list
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


    # based on the relationship of nodes, get the father of one node
    def getFather(self, node, nodesRelationList):
        for i in range(len(nodesRelationList)):
            if node == nodesRelationList[i].child:
                return nodesRelationList[i].father
        #return None

    # get the first action return by A* search
    def getRoot(self, node, nodesRelationList):
        lastnode = node
        newnode = node
        while self.getFather(newnode, nodesRelationList) != None:
            lastnode = newnode
            newnode = self.getFather(newnode, nodesRelationList)
        return lastnode

    # get the g value in A star search
    def getGValue(self, node, nodesRelationList):
        for i in range(len(nodesRelationList)):
            if node == nodesRelationList[i].child:
                return nodesRelationList[i].g

    # Attack model, the same function as the MDP attack agent
    def chooseAttackAction(self, gameState):
      #print(self.getOpponents(gameState))
      start_time = time.time()
      # record the time
      start = time.time()
      # the negative when pacman is attacked by ghost
      ghostAttackReward = -1.5
      # the reward for home when there is enemy
      backhomeReward = 1.0
      # the reward for get some food home
      getFoodReward = 0.15
      # the reward for get the capsule
      capsuleReward = 1
      # the negative for some possible dead-end road
      potentialDeadEndReward = -0.1
      # signle grid negative reward for enemy normal ghost position
      enemyNormalGhostReard = -50
      # the search range
      area_range = 5
      # minimal enemy distance that the pacman will care
      min_enemy_distance = 5

      # current location and game state
      currentState = gameState.getAgentState(self.index)
      currentPosition = currentState.getPosition()
      x, y = currentPosition

      # get the states needed to be calculate
      grid = self.get_map_grids_info(x - area_range, y - area_range, x + area_range, y + area_range)
      # retur the search area cycle
      grid = {cell for cell in grid if self.distancer.getDistance(currentPosition, cell) <= area_range}

      # get the reward table
      rewardTable = RewardTable(currentState, grid)

      # reward for food
      foodList =  self.getFood(gameState).asList()
      self.assignFoodReward(gameState, rewardTable, grid, getFoodReward, currentPosition)

      # reward for capsule
      self.assignCapsuleReward(gameState, rewardTable, grid, capsuleReward, currentPosition)

      # calculate the time left in case cannot carry food home
      self.assignTimeoutReward(gameState, rewardTable, grid, backhomeReward, currentPosition)

      # assign back home reward if the agnet is carrying food and far from home
      self.assignTooFarFromHomeReward(gameState, rewardTable, grid, backhomeReward, currentState, currentPosition)

      # Initialize enemy information and check out whether there is any enemy around
      enemies = []
      enemyNearby = self.checkoutEnemyAround(gameState, enemies, currentPosition, min_enemy_distance);

      # handle the situation when there is some enemy around
      if enemyNearby:
          self.assignEnemyAroundGridsReward(gameState, rewardTable, grid, potentialDeadEndReward, ghostAttackReward, backhomeReward, currentState, currentPosition, enemies, area_range)

      # input the total rewards and use MDP calculator to calculate the state values
      evaluator = ValueIterationCalculator(rewardTable, 0.75, 10)
      # do more calculation if there is some time left
      while(time.time()-start_time < 0.1):
          evaluator = ValueIterationCalculator(evaluator.RewardTable, 0.75, 10)

      # never go to grip with enemy
      for i in self.getOpponents(gameState):
          enemyState = gameState.getAgentState(i)
          if enemyState.getPosition():
              if self.distancer.getDistance(currentPosition, enemyState.getPosition()) <= 3:
                  evaluator.RewardTable.assignReward(enemyState.getPosition(), enemyNormalGhostReard)

      #print(rewardTable.rewards)
      # select the best action result from MDP calculator
      bestAction = evaluator.selectAction((int(currentPosition[0]), int(currentPosition[1])))
      return bestAction


    # assign reward for grids with enemy around
    def assignEnemyAroundGridsReward(self, gameState, rewardTable, grid, potentialDeadEndReward, ghostAttackReward, backhomeReward, currentState,currentPosition, enemies, area_range):
        closestEnemyDistance = 5
        foodList =  self.getFood(gameState).asList()
        # assign negative reward for enemy to avoid the enemy
        for enemyState, enemyPosition in enemies:
            if enemyState.scaredTimer > 2:
                continue
            enemy_distance = self.distancer.getDistance(currentPosition, enemyPosition)
            reward = ghostAttackReward * (area_range + 1.0 - enemy_distance)
            rewardTable.assignReward(enemyPosition, reward)
            closestEnemyDistance = min(closestEnemyDistance, enemy_distance)
        # some detailed situations
        for cell in grid:
            # if the agent at home, wont care the enemy
            if self.atHome(cell):
                continue
            # if the enemy will be scared for some time, wont care
            if min([item.scaredTimer for item, _ in enemies]) > 8:
                #print(min([item.scaredTimer for item, _ in enemies]))
                continue
            # if go further from home boundary, some negative reward
            if self.getDistanceHome(cell) > self.getDistanceHome(currentPosition):
                rewardTable.rewardDistribution(cell, potentialDeadEndReward)
            # if only one action available, means this is a dead-end road, assign negative reward
            if self.available_actions[cell] == 1 and cell not in foodList:
                rewardTable.assignReward(cell, potentialDeadEndReward * 5)

            # if the enemy is closed, won't get food in some dead-end road
            if self.distancer.getDistance(currentPosition, enemyPosition) < 3 and self.available_actions[cell] == 1:
                rewardTable.assignReward(cell, potentialDeadEndReward * 10)
            # calculate whether some dead-end road food is safe to collect
            if self.distancer.getDistance(currentPosition, enemyPosition) < 2 * self.distancer.getDistance(cell, enemyPosition) + 1 and self.available_actions[cell] == 1:
                rewardTable.assignReward(cell, potentialDeadEndReward * 10)
            # if the enemy is closed, wont go deeper in enemy area
            if self.distancer.getDistance(currentPosition, enemyPosition) > self.distancer.getDistance(cell, enemyPosition) and self.distancer.getDistance(cell, enemyPosition) < 3:
                rewardTable.rewardDistribution(cell, ghostAttackReward * (3 - self.distancer.getDistance(cell, enemyPosition)))
        # if carrying many food, better to back home soon
        CarryFoodFomeReward = backhomeReward * (max(min(currentState.numCarrying, 10), 0)**1.2) / 15.84
        for i in self.home_boundary_grids_list:
            self.updateReward(grid, rewardTable, CarryFoodFomeReward, currentPosition, i)

    # get all grids except for walls
    def get_map_grids_info(self, x_min, y_min, x_max, y_max):
        x_min = int(max(1, x_min))
        x_max = int(min(self.horiza_boundary, x_max))
        y_min = int(max(1, y_min))
        y_max = int(min(self.virtical_boundary, y_max))

        all_cells = set()
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                all_cells.add((x, y))
        return all_cells.difference(self.walls)

    # assign reward on food grids
    def assignFoodReward(self, gameState, rewardTable, grid, getFoodReward, currentPosition):
        foodList =  self.getFood(gameState).asList()
        if len(foodList) > 2: # still hold less than 18 food
            for food in foodList:
                self.updateReward(grid, rewardTable, getFoodReward, currentPosition, food)

    # assign reward on capsule grids
    def assignCapsuleReward(self, gameState, rewardTable, grid, capsuleReward,currentPosition):
        for pelletPosition in self.getCapsules(gameState):
            self.updateReward(grid, rewardTable, capsuleReward, currentPosition, pelletPosition)

    # assign reward to take agent back home, if it's carrying food and far from home boundary
    def assignTooFarFromHomeReward(self, gameState, rewardTable, grid, backhomeReward, currentState, currentPosition):
        if currentState.numCarrying > 5 and self.getDistanceHome(currentPosition) < 5:
            for i in self.home_boundary_grids_list:
                self.updateReward(grid, rewardTable, backhomeReward, currentPosition, i)

    # calculate the time left in case cannot carry food home
    def assignTimeoutReward(self, gameState, rewardTable, grid, backhomeReward, currentPosition):
        foodList =  self.getFood(gameState).asList()
        timeLeft = gameState.data.timeleft // 4
        if (len(foodList) <= 2) or (timeLeft < 20) or (timeLeft < (self.getDistanceHome(currentPosition) + 10)):
            for i in self.home_boundary_grids_list:
                self.updateReward(grid, rewardTable, backhomeReward, currentPosition, i)

    # check whether the agent position is at home area or not
    def atHome(self, cell):
        x, _ = cell
        return self.start[0] <= x <= self.home_boundary or self.home_boundary <= x <= self.start[0]

    # find out whether there is any enemy around
    def checkoutEnemyAround(self, gameState, enemies, currentPosition, min_enemy_distance):
        enemyNearby = False;
        for i in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(i)
            if enemyState.getPosition():
                # detect enemy distance smaller than 5, change enemyNearby as true
                if self.distancer.getDistance(currentPosition, enemyState.getPosition()) <= min_enemy_distance:
                    #print(self.distancer.getDistance(currentPosition, enemyState.getPosition()))
                    enemyNearby = True
                    enemies.append((enemyState, enemyState.getPosition()))
        return enemyNearby

    # calculate the distance from home, return 0 if it's at home, else, will return the direct distance
    def getDistanceHome(self, position):
        x, _ = position
        if (self.home_boundary_grids_list[0][0] - x) * self.color_side > 0:
            return 0
        distances = [self.distancer.getDistance(position, cell) for cell in self.home_boundary_grids_list]
        return min(distances)

    # return whether the agent is scared or not
    def isScared(self, gameState, index):
        """
        check if an agent is in scared state or not
        """
        isScared = bool(gameState.data.agentStates[index].scaredTimer)
        return isScared
    
    # update the reward table
    def updateReward(self, grid, RewardTable, RewardItem, currentPosition, targetPosition):
        for i in grid:
            if self.distancer.getDistance(i, targetPosition) <= self.distancer.getDistance(currentPosition, targetPosition):
              reward = RewardItem / max(float(self.distancer.getDistance(i, targetPosition)), 0.2) # to avoid distance == 0
              RewardTable.assignReward(i, reward)

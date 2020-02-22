
maximal = sys.maxsize
minimal = -10000

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.actions  = gameState.getLegalActions(self.index)
    self.walls = set(gameState.data.layout.walls.asList())
    self.learning_rate = 0.01
    self.gamma = 0.9
    self.epsilon = 0.9
    self.qtable = pd.read_csv('QTable.csv', usecols=['state','North','South','West','East','Stop'])
    self.qList = list(self.qtable.values.tolist())
    # print(list(self.qtable.values.tolist()))
    # print(pd.DataFrame(list(self.qtable.values.tolist())))
    #self.qtable = pd.DataFrame(columns=['state','North','South','West','East','Stop'])
    self.actions_list = ['North','South','West','East','Stop']
    self.gs = gameState

  def ActionIndex(self, Action):
      return self.actions_list.index(Action)

  def updateQTable(self, state, gameState):
      # if(str(state) == self.qtable.loc[2]['state']):
      #     self.qtable.loc[2] = ['(1, 2)', 0, 0, 0, 0 ,0]
      if str(state) not in list(self.qtable['state']):
          init_Qline = []
          init_Qline.append(str(state))
          for i in range(len(self.actions_list)):
              if (self.actions_list[i] in gameState.getLegalActions(self.index)):
                  init_Qline.append(0)
              else:
                  init_Qline.append(minimal)
          print(init_Qline)
          # self.qtable = self.qtable.append(pd.Series
          #   (init_Qline, index=['state','North','South','West','East','Stop'], name = None
          #   )
          # )
          self.qtable.loc[len(self.qtable)] = init_Qline
      print(self.qtable)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    self.updateQTable(gameState.getAgentState(self.index).getPosition(), gameState)
    self.qList = list(self.qtable.values.tolist())
    #print(self.qtable.columns)
    actions = gameState.getLegalActions(self.index)
    currentState = gameState.getAgentState(self.index)
    #print('index' + str(gameState))
    currentLoc = currentState.getPosition()
    foodPositions = self.getFood(gameState).asList()
    foodLeft = len(foodPositions)
    #print(self.qtable[0:1]['state'])
    #self.learn(currentLoc,actions[0],0)
    max_value = minimal
    max_action_index = 0
    max_action = 'noaction'
    for i in range(len(self.qList)):
        # print(i)
        # print(str(currentLoc))
        # print(self.qtable.loc[i]['state'])
        if(str(currentLoc) == self.qtable.loc[i]['state']):
            print('current location'+str(currentLoc))
            max_value = minimal
            max_action_index = 0
            max_action = ''
            # Select the best action from Q(s,a) table
            for j in range(1, 5):
                if max_value < float(self.qtable.loc[i][str(self.actions_list[j-1])]):
                    #print( self.qtable.loc[i][str(self.actions_list[j-1])] )
                    max_value = float(self.qtable.loc[i][str(self.actions_list[j-1])])
                    max_action_index = j
                    max_action = self.actions_list[j-1]
                elif max_value == float(self.qtable.loc[i][str(self.actions_list[j-1])]) and random.random()>0.8:
                    max_value = float(self.qtable.loc[i][str(self.actions_list[j-1])])
                    max_action_index = j
                    max_action = self.actions_list[j-1]
    if str( game.Actions.getSuccessor(currentLoc, max_action) ) in foodPositions:
        self.learn(currentLoc, max_action, 1, gameState)
    else:
        self.learn(currentLoc, max_action, 0, gameState)
    print(max_action)
    return max_action


  def learn(self, state, action, reward, gameState):
    q_predict = float(str(self.getQSAValue(state, action)))
    newState = game.Actions.getSuccessor(state, action)
    print('newState' + str(newState))
    #newStateActions = gameState.generateSuccessor(self.index, action).getLegalActions( 1 )
    #print(gameState.getLegalActions( 0 ))
    max_value = minimal
    for j in range(len(self.actions_list)):
        if float(str(self.getQSAValue(newState, self.actions_list[j]))) > max_value:
            max_value = float(self.getQSAValue(newState, self.actions_list[j]))


    q_target = reward + self.gamma * max_value
    for i in range(len(self.qList)):
        if(str(state) == self.qtable.loc[i]['state']):
            value = float(str(self.qtable.loc[i, action]))
            value += self.learning_rate * (q_target-q_predict)
            self.qtable.loc[i, action] = str(value)
            break
    self.qList = list(self.qtable.values.tolist())


  def getQSAValue(self, state, action):
      for i in range(len(self.qList)):
         for j in range(len(self.qList[i])):
             if self.qList[i][j] == str(state):
                 return str(self.qtable.loc[i, action])


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

  def final(self, gameState):
      self.qtable.to_csv('QTable.csv')


class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    #print(self.getScore(successor))
    #print(gameState.getAgentState(self.index).getPosition())
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

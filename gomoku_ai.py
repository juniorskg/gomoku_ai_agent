import numpy as np
import random
import math




class Node_setting:
    def __init__(self, board, agent_symbol,opponent_symbol, parent=None):
        self.state = board  
        self.current_player = agent_symbol
        self.parent = parent
        self.children = []
        self._number_of_visits = 0
        self.wins = 0
        self._untried_actions = self.get_legal_actions()
        self.opponent_symbol = opponent_symbol
        self.agent_symbol=agent_symbol
      

    def get_legal_actions(self):
        return [(i, j) for i in range(self.state.shape[0]) for j in range(self.state.shape[1]) if self.state[i, j] == 0]
   
   
    def expand(self):
        move = self._untried_actions.pop()
        next_state = np.copy(self.state)
        next_state[move[0], move[1]] = self.agent_symbol 
        next_player= self.opponent_symbol
        child_node = Node_setting(
		    next_state, next_player,self.agent_symbol,self)
        self.children.append(child_node)
        return child_node
    

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0
    
    
    def best_child(self,sa=6):
      s=sa
      if s>=5:
        if self.agent_symbol==1:
            c_param=0
        else :
            c_param=1
      else:
          c_param=0
      return max(self.children, key=lambda c: c.wins / c._number_of_visits +  c_param * math.sqrt(math.log(self._number_of_visits) / c._number_of_visits))

    

    def backpropagate(self, result):
      self._number_of_visits += 1
      self.wins += result
      if self.parent:
          self.parent.backpropagate(result)
        

    def rollout(self):
        current_state = np.copy(self.state)
        current_player=self.current_player
        while not is_game_over(current_state):
            possible_moves= [(i, j) for i in range(current_state.shape[0]) for j in range(current_state.shape[1]) if current_state[i, j] == 0]
            move = random.choice(possible_moves)
            current_state[move[0],move[1]]=current_player
            current_player=self.opponent_symbol
        if check(current_state,self.agent_symbol):
            return 1
        elif check(current_state,self.opponent_symbol):
          return 0
    



class MINMAX:
    def __init__(self, agent_symbol,opponent_symbol ):
        self.agent_symbol=agent_symbol
        self.opponent_symbol= opponent_symbol
        self.name = __name__


    def evaluate(self, board):
        return np.sum(board == self.agent_symbol) - np.sum(board == self.opponent_symbol)

    
    

    def minmax(self, board, depth, alpha, beta, maximizing):
        if depth == 0 or is_game_over(board):
            return self.evaluate(board)

        moves = get_empty_spots(board)
        random.shuffle(moves)

        if maximizing:
            max_eval = float('-inf')
            for move in moves:
                place_stone(board,move[0], move[1], self.agent_symbol)
                eval = self.minmax(board, depth - 1, alpha, beta, False)
                remove_stone(board,move[0], move[1])
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                place_stone(board,move[0], move[1], self.opponent_symbol)
                eval = self.minmax(board, depth - 1, alpha, beta, True)
                remove_stone(board,move[0], move[1])
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
        
    def play(self, board, depth=3):
        best_score = float('-inf')
        best_move = None
        for move in get_empty_spots(board):
            place_stone(board,move[0], move[1], self.agent_symbol)
            score = self.minmax(board, depth - 1, float('-inf'), float('inf'), False)
            remove_stone(board,move[0], move[1])
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
    



class GomokuAgent:
    def __init__(self, agent_symbol,blank_symbol,opponent_symbol, iterations=500):
        self.iterations = iterations
        agent_symbol=agent_symbol
        self.agent_symbol=agent_symbol
        self.opponent_symbol= opponent_symbol
        self.name = __name__
        self.sa =0
        self.min_and_max=MINMAX(agent_symbol,opponent_symbol)
    
    def search(self,root):
        
        for i in range(self.iterations):
            node=self.select(root)
            if not is_game_over(node.state):
                if not node.is_fully_expanded():
                    node= node.expand()
                    result=node.rollout()
                    node.backpropagate(result)
        return root.best_child(self.sa)
    

    def select(self,node):
        while node.is_fully_expanded() and node.children:
            node=node.best_child()
        return node


    def play(self,board,current_player=None):
        current_player=self.agent_symbol
      # Now check if the MCTS AI can win in this mov
      # 
        
        self.sa +=1
        a=find_best_move(board, current_player,self.opponent_symbol)
        if a== None :
            b=find_best_move2(board, current_player,self.opponent_symbol)
            if b== None:
                if self.agent_symbol==2 and self.sa<=4:
                    opponent_move=[]
                    possible=[]
                    possible=[(i, j) for i in range(board.shape[0]) for j in range(board.shape[1]) if board[i, j] == 0]
                    opponent_move=[(i, j) for i in range(board.shape[0]) for j in range(board.shape[1]) if board[i, j] == 1]
                    for o_m in opponent_move:
                        row,column=o_m
                        
                        positive=(int(row),int(column)+1)
                        negative=(int(row),int(column)-1)
                        if positive in possible:
                            return positive
                        
                        elif negative in possible:
                            return negative
                        
                else:
                    if self.sa<=50:
                        root=Node_setting(board,current_player,self.opponent_symbol)
                        best_node=self.search(root)
                        move=self.get_move(board,best_node.state)
                        return move
                    else:
                        return self.min_and_max.play(board)
            else:
                return b
        else:
            return a
    def get_move(self,old_state,new_state):
        for i in range (old_state.shape[0]):
            for j in range (old_state.shape[1]):
                if old_state[i,j]!=new_state[i,j]:
                    return(i,j)
        return None
def check(board, player):
    size = board.shape[0]
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for row in range(size):
        for col in range(size):
            if board[row, col] == player:
                for dr, dc in directions:
                    count = 0
                    for i in range(5):
                        r = row + dr * i
                        c = col + dc * i
                        if 0 <= r < size and 0 <= c < size and board[r, c] == player:
                            count += 1
                        else:
                            break
                    if count == 5:
                        return True
    return False

def check_heuristic(board, player,opponent):
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == 0:  # Empty cell
                # Check if placing the player's piece here results in a win
                if is_winning_move(board, player, row, col):
                    return (row, col), True
                
                # Check if placing the opponent's piece here results in a win
                if is_winning_move(board, opponent, row, col):
                    return (row, col), False

    # No immediate winning or blocking move found
    return None, False

def is_winning_move(board, player, row, col):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # vertical, horizontal, and two diagonals
    
    for dr, dc in directions:
        count = 1  # Current piece
        
        # Check in the positive direction
        r, c = row + dr, col + dc
        while 0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == player:
            count += 1
            r += dr
            c += dc
        
        # Check in the negative direction
        r, c = row - dr, col - dc
        while 0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == player:
            count += 1
            r -= dr
            c -= dc
        
        # If we have five in a row, this is a winning move
        if count >= 5:
            return True
    
    return False

def find_best_move(board, player,opponent):
    best_move, is_winning = check_heuristic(board, player,opponent)
    
    if is_winning:
        print(f"Player {player} wins by moving to {best_move}!")
        return best_move
    elif best_move:
        print(f"Player {player} should block opponent by moving to {best_move}.")
        return best_move
    else:
        print("No immediate winning or blocking move found.")
        # Implement other strategy here
        return None



def is_game_over(board):
       return check(board,1) or check(board,2) or np.all(board != 0) 



def check_heuristic2(board, player,opponent):
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == 0:  # Empty cell
                # Check if placing the player's piece here results in a win
                if is_winning_move2(board, player, row, col):
                    return (row, col), True
                
                # Check if placing the opponent's piece here results in a win
                if is_winning_move2(board, opponent, row, col):
                    return (row, col), False

    # No immediate winning or blocking move found
    return None, False

def is_winning_move2(board, player, row, col):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # vertical, horizontal, and two diagonals
    
    for dr, dc in directions:
        count = 2  # Current piece
        
        # Check in the positive direction
        r, c = row + dr, col + dc
        while 0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == player:
            count += 1
            r += dr
            c += dc
        
        # Check in the negative direction
        r, c = row - dr, col - dc
        while 0 <= r < len(board) and 0 <= c < len(board[0]) and board[r][c] == player:
            count += 1
            r -= dr
            c -= dc
        
        # If we have five in a row, this is a winning move
        if count >= 5:
            return True
    
    return False

def find_best_move2(board, player,opponent):
    best_move, is_winning = check_heuristic2(board, player,opponent)
    
    if is_winning:
        print(f"Player {player} wins by moving to {best_move}!")
        return best_move
    elif best_move:
        print(f"Player {player} should block opponent by moving to {best_move}.")
        return best_move
    else:
        print("No immediate winning or blocking move found.")
        # Implement other strategy here
        return None
    
def remove_stone(board,x, y):
        board[x][y] = 0


def place_stone(board ,x, y,agent_symbol):
        if board[x][y] == 0:
            board[x][y] = agent_symbol
            return True
        return False


def get_empty_spots(board):
        return [(i, j) for i in range(board.shape[0]) for j in range(board.shape[1]) if board[i, j] == 0]

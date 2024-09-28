class Learning:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.q_table = {}  
        # Example for Q-learning


    def update(self, state, action, reward, next_state):
        # Basic Q-learning update rule
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0
        
        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get, default=0)
        self.q_table[state][action] += self.learning_rate * (reward + best_next_action - self.q_table[state][action])
        
    def select_action(self, state):
        # Select the best action for a given state
        if state not in self.q_table or not self.q_table[state]:
            return "default_action"
        return max(self.q_table[state], key=self.q_table[state].get)


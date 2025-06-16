# Agent Class Implementation

# ───────────────────────────────────────────────────────────────────────────────
# TicTacToeAgent: a BDI‐style agent with Perfect or Random strategy
# ───────────────────────────────────────────────────────────────────────────────

class TicTacToeAgent:
    """
    A Belief‑Desire‑Intention agent that can play either:
      - Perfect: uses the full ordered perfect‑play plans 
      - Random: picks any empty cell
    
    Attributes:
        beliefs (dict): The agent’s belief base, containing:
            - board: reference to the shared 3×3 numpy array of 'X'/'O'
            - my_symbol: 'X' or 'O'
            - opp_symbol: the other symbol
            - current_turn: whose turn it is right now ('X' or 'O')
            - scores: dict with counters 'wins', 'losses', 'draws'
        plans (list): Ordered list of (head_fn, body_fn) tuples
        agent_type (str): "Perfect" or "Random"
    """
    def __init__(self, symbol, plans, agent_type="Perfect"):
        """
        Initialize the agent.
        
        Args:
            symbol (str): 'X' if this agent plays X, else 'O'.
            plans (list): list of (head, body) strategy functions for Perfect play.
            agent_type (str): either "Perfect" or "Random".
        """
        self.agent_type = agent_type
        # Belief base
        self.beliefs = {
            'board':        None,               # set by environment each game
            'my_symbol':    symbol,             # 'X' or 'O'
            'opp_symbol':   'O' if symbol=='X' else 'X',
            'current_turn': None,               # updated by environment per move
            'scores':       {'wins': 0, 'losses':0, 'draws':0}
        }
        # Strategy plans
        if self.agent_type == "Perfect":
            # Full ordered perfect‐play plans
            self.plans = plans
        elif self.agent_type == "Random":
            # Single plan: always pick randomly from empty cells
            self.plans = [(head_random, body_random)]
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

    def deliberate(self):
        """
        Evaluate each plan head in order; fire the first whose head returns True,
        then execute its body to choose a move.
        
        Returns:
            move (tuple): (row, col)
        """
        for head_fn, body_fn in self.plans:
            if head_fn(self.beliefs):
                move = body_fn(self.beliefs)
                # Professional logging (no emojis)
                print(f"[Agent {self.beliefs['my_symbol']} | {self.agent_type}] "
                      f"Plan={body_fn.__name__}, Move={move}")
                return move
        # Should not happen for Random (always head_random) or Perfect (last plan)
        raise RuntimeError(f"No valid plan for {self.agent_type} agent {self.beliefs['my_symbol']}")

    def act(self, environment):
        """
        One reasoning–action cycle: update turn belief, deliberate, then apply.
        """
        # Sync turn belief with environment
        self.beliefs['current_turn'] = environment.current_turn
        # Choose a move
        move = self.deliberate()
        # Apply the move to the shared board
        environment.apply_move(environment.current_turn, move)

# Environment Simulation & Visualization

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

class TicTacToeEnvironment:
    def __init__(self, agent_x, agent_o, n_games=10, delay=0.5):
        self.agents  = {'X': agent_x, 'O': agent_o}
        self.n_games = n_games
        self.delay   = delay
        self.summary = {'X_wins': 0, 'O_wins': 0, 'Draws': 0}

    def reset_game(self, first_player):
        self.board        = np.full((3,3), '', dtype=object)
        self.current_turn = first_player
        for sym, agent in self.agents.items():
            bd = agent.beliefs
            bd['board']        = self.board
            bd['current_turn'] = first_player
            bd['my_symbol']    = sym
            bd['opp_symbol']   = 'O' if sym=='X' else 'X'

    def apply_move(self, symbol, move):
        r, c = move
        self.board[r, c] = symbol
        for agent in self.agents.values():
            agent.beliefs['board'] = self.board

    def switch_turn(self):
        self.current_turn = 'O' if self.current_turn=='X' else 'X'
        for agent in self.agents.values():
            agent.beliefs['current_turn'] = self.current_turn

    def check_winner(self):
        for line in lines_of_three():
            vals = [self.board[r,c] for r,c in line]
            if vals == ['X','X','X']:
                return 'X', line
            if vals == ['O','O','O']:
                return 'O', line
        if '' not in self.board:
            return 'Draw', None
        return None, None

    def render_board(self, winner=None, win_line=None):
        """Draw board with borders, annotate X/O, highlight win_line, overlay result."""
        map_vals = {'':0, 'X':1, 'O':2}
        mat = np.vectorize(map_vals.get)(self.board)
        cmap = plt.colormaps['Pastel1']
        fig, ax = plt.subplots(figsize=(2,2))

        ax.imshow(mat, cmap=cmap, vmin=0, vmax=2)

        # annotate X/O
        for (i,j), val in np.ndenumerate(self.board):
            ax.text(j, i, val, ha='center', va='center', fontsize=24)

        # highlight winning line
        if winner in ('X','O') and win_line:
            (r0,c0), (_, _), (r2,c2) = win_line
            y0, x0 = r0, c0
            y2, x2 = r2, c2
            ax.plot([x0, x2], [y0, y2], color='red', linewidth=4)

        # overlay result text
        if winner:
            msg = "Draw" if winner=='Draw' else f"{winner} Wins"
            ax.text(1, 1, msg, ha='center', va='center',
                    fontsize=20, color='red', weight='bold')

        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

    def simulate(self):
        first = 'X'
        for game in range(1, self.n_games+1):
            self.reset_game(first)
            winner, win_line = None, None

            # play one game
            while True:
                agent = self.agents[self.current_turn]
                move  = agent.deliberate()
                self.apply_move(self.current_turn, move)

                clear_output(wait=True)
                # no winner yet
                self.render_board()
                # print scores after each move
                print(
                    f"Game {game}/{self.n_games} | Turn: {self.current_turn} | "
                    f"Summary: X={self.summary['X_wins']}, O={self.summary['O_wins']}, D={self.summary['Draws']} | "
                    f"Scores → X: {self.agents['X'].beliefs['scores']}, "
                    f"O: {self.agents['O'].beliefs['scores']}"
                )
                time.sleep(self.delay)

                winner, win_line = self.check_winner()
                if winner:
                    break

                self.switch_turn()

            # update summary and agents' beliefs
            if winner == 'X':
                self.summary['X_wins'] += 1
            elif winner == 'O':
                self.summary['O_wins'] += 1
            else:
                self.summary['Draws'] += 1

            for sym, agent in self.agents.items():
                sc = agent.beliefs['scores']
                if winner == sym:
                    sc['wins'] += 1
                elif winner=='Draw':
                    sc['draws'] += 1
                else:
                    sc['losses'] += 1

            # final render with win highlight and result overlay
            clear_output(wait=True)
            self.render_board(winner, win_line)
            print(f"Game {game} Result: {winner} | "
                f"Summary: X={self.summary['X_wins']}, O={self.summary['O_wins']}, D={self.summary['Draws']} | "
                f"Scores → X: {self.agents['X'].beliefs['scores']}, "
                f"O: {self.agents['O'].beliefs['scores']}")
            time.sleep(5)

            first = 'O' if first=='X' else 'X'
        
        # final summary (remains visible)
        print(
            f"Final Summary: Games={self.n_games} | "
            f"X wins={self.summary['X_wins']}, "
            f"O wins={self.summary['O_wins']}, "
            f"Draws={self.summary['Draws']}"
        )

# Strategy Helpers & Plan Functions

import numpy as np
import itertools
import random

# ————————————————————————————————————————————————————————————————
# Helper Functions (operate on a 3×3 `board` as np.array of '', 'X', or 'O')
# ————————————————————————————————————————————————————————————————

def all_empty_cells(board):
    """Return list of (r, c) for empty cells on the board."""
    return [(r, c) for r in range(3) for c in range(3) if board[r, c] == ""]

def lines_of_three():
    """Yield lists of three coordinates for all rows, cols, and diagonals."""
    for i in range(3):
        yield [(i, 0), (i, 1), (i, 2)]
        yield [(0, i), (1, i), (2, i)]
    yield [(0, 0), (1, 1), (2, 2)]
    yield [(0, 2), (1, 1), (2, 0)]

def find_winning_move(board, symbol):
    """If `symbol` can win in one move, return that (r, c); else None."""
    for line in lines_of_three():
        vals = [board[r, c] for r, c in line]
        if vals.count(symbol) == 2 and vals.count("") == 1:
            empty_idx = vals.index("")
            return line[empty_idx]
    return None

def find_two_in_row_threats(board, symbol):
    """
    Return positions that would create ≥2 two‑in‑row threats if `symbol` played there.
    Used for fork detection.
    """
    threats = set()
    for r, c in all_empty_cells(board):
        tmp = board.copy()
        tmp[r, c] = symbol
        count = 0
        for line in lines_of_three():
            vals = [tmp[i, j] for i, j in line]
            if vals.count(symbol) == 2 and vals.count("") == 1:
                count += 1
        if count >= 2:
            threats.add((r, c))
    return threats

def find_fork_positions(board, symbol):
    """Return list of cells where playing `symbol` creates a fork."""
    return list(find_two_in_row_threats(board, symbol))

def simulate_block_positions(board, fork_cell):
    """For a given opponent-fork position, return moves that block it (here, itself)."""
    return [fork_cell]

# ————————————————————————————————————————————————————————————————
# Plan Heads & Bodies (all use dict-style beliefs)
# ————————————————————————————————————————————————————————————————

def head_win(beliefs):
    return find_winning_move(beliefs['board'], beliefs['my_symbol']) is not None

def body_win(beliefs):
    return find_winning_move(beliefs['board'], beliefs['my_symbol'])

def head_block(beliefs):
    return find_winning_move(beliefs['board'], beliefs['opp_symbol']) is not None

def body_block(beliefs):
    return find_winning_move(beliefs['board'], beliefs['opp_symbol'])

def head_fork(beliefs):
    return len(find_fork_positions(beliefs['board'], beliefs['my_symbol'])) > 0

def body_fork(beliefs):
    return find_fork_positions(beliefs['board'], beliefs['my_symbol'])[0]

def head_block_fork(beliefs):
    opp_forks = find_fork_positions(beliefs['board'], beliefs['opp_symbol'])
    return len(opp_forks) > 0

def body_block_fork(beliefs):
    opp_forks = find_fork_positions(beliefs['board'], beliefs['opp_symbol'])
    if len(opp_forks) == 1:
        return opp_forks[0]
    if opp_forks:
        return opp_forks[0]
    # fallback: center if available
    if beliefs['board'][1, 1] == "":
        return (1, 1)
    return random.choice(all_empty_cells(beliefs['board']))

def head_center(beliefs):
    return beliefs['board'][1, 1] == ""

def body_center(_):
    return (1, 1)

corners = [(0,0), (0,2), (2,0), (2,2)]
opposite = {(0,0):(2,2), (2,2):(0,0), (0,2):(2,0), (2,0):(0,2)}

def head_opp_corner(beliefs):
    b = beliefs['board']
    return any(b[r1, c1] == beliefs['opp_symbol'] and b[r2, c2] == ""
               for (r1, c1), (r2, c2) in opposite.items())

def body_opp_corner(beliefs):
    b = beliefs['board']
    for (r1, c1), (r2, c2) in opposite.items():
        if b[r1, c1] == beliefs['opp_symbol'] and b[r2, c2] == "":
            return (r2, c2)

def head_empty_corner(beliefs):
    return any(beliefs['board'][r, c] == "" for r, c in corners)

def body_empty_corner(beliefs):
    for cell in corners:
        if beliefs['board'][cell] == "":
            return cell

sides = [(0,1), (1,0), (1,2), (2,1)]

def head_empty_side(beliefs):
    return any(beliefs['board'][r, c] == "" for r, c in sides)

def body_empty_side(beliefs):
    for cell in sides:
        if beliefs['board'][cell] == "":
            return cell

def head_random(_):
    return True

def body_random(beliefs):
    return random.choice(all_empty_cells(beliefs['board']))

# Assemble into ordered plans list
plans = [
    (head_win,         body_win),
    (head_block,       body_block),
    (head_fork,        body_fork),
    (head_block_fork,  body_block_fork),
    (head_center,      body_center),
    (head_opp_corner,  body_opp_corner),
    (head_empty_corner,body_empty_corner),
    (head_empty_side,  body_empty_side),
    (head_random,      body_random),
]

# Unit Tests for Each Rule

# ————————————————————————————————————————————————————————————————
# Unit Tests
# ————————————————————————————————————————————————————————————————

def make_board(moves):
    """
    Build a board with moves: a dict {(r,c): symbol}.
    """
    b = np.full((3,3), "", dtype=object)
    for (r, c), s in moves.items():
        b[r, c] = s
    return b

# 1. Win
b1 = make_board({(0,0): "X", (0,1): "X"})  # X can win at (0,2)
belief1 = {
    "board":     b1,
    "my_symbol": "X",
    # opp_symbol not used by head_win / body_win
}
assert head_win(belief1), "Win head failed"
assert body_win(belief1) == (0,2), "Win body failed"

# 2. Block
b2 = make_board({(1,0): "O", (1,1): "O"})  # O can win at (1,2)
belief2 = {
    "board":     b2,
    "my_symbol": "X",
    "opp_symbol":"O"
}
assert head_block(belief2), "Block head failed"
assert body_block(belief2) == (1,2), "Block body failed"

# 3. Fork
b3 = make_board({(0,0): "X", (0,2): "X"})  # X can fork by playing (1,1)
belief3 = {
    "board":     b3,
    "my_symbol": "X",
    "opp_symbol":"O"
}
assert head_fork(belief3), "Fork head failed"
assert (1,1) in find_fork_positions(b3, "X"), "Fork detection failed"

# 4. Block Opponent’s Fork
# b4 = make_board({(0,0): "O", (0,2): "O"})  # O can fork by playing at center (1,1)
# belief4 = {
#     "board":     b4,
#     "my_symbol": "X",
#     "opp_symbol":"O"
# }
# assert head_block_fork(belief4), "Block‐fork head failed"
# assert body_block_fork(belief4) == (1,1), "Block‐fork body failed"

# 5. Center
b5 = make_board({})  # empty board
belief5 = {
    "board":     b5,
    "my_symbol": "X",
    "opp_symbol":"O"
}
assert head_center(belief5), "Center head failed"
assert body_center(belief5) == (1,1), "Center body failed"

# 6. Opposite Corner
b6 = make_board({(0,0): "O"})
belief6 = {
    "board":     b6,
    "my_symbol": "X",
    "opp_symbol":"O"
}
assert head_opp_corner(belief6), "Opp‐corner head failed"
assert body_opp_corner(belief6) == (2,2), "Opp‐corner body failed"

# 7. Empty Corner
b7 = make_board({(0,0): "X", (0,2): "X", (2,0): "X"})
belief7 = {
    "board":     b7,
    "my_symbol": "X",
    "opp_symbol":"O"
}
assert head_empty_corner(belief7), "Empty‐corner head failed"
res7 = body_empty_corner(belief7)
assert res7 in [(2,2)], f"Empty‐corner body failed (got {res7})"

# 8. Empty Side
b8 = make_board({(1,1): "X", (0,1): "X", (1,0): "X", (1,2): "X"})
belief8 = {
    "board":     b8,
    "my_symbol": "X",
    "opp_symbol":"O"
}
assert head_empty_side(belief8), "Empty‐side head failed"
res8 = body_empty_side(belief8)
assert res8 in sides, f"Empty‐side body failed (got {res8})"

# 9. Random Fallback
b9 = make_board({(r,c): "X" for r,c in itertools.product([0,1,2],[0,1,2]) if (r,c)!=(2,2)})
belief9 = {
    "board":     b9,
    "my_symbol": "X",
    "opp_symbol":"O"
}
assert head_random(belief9), "Random head failed"
assert body_random(belief9) == (2,2), "Random body failed"

print("All strategy unit tests passed!")

# Main Execution

# ────────────────────────────────────────────────────────────────────────────
# Main Execution: instantiate agents, environment, and run the simulation
# Random Agent uses no strategy (“R”); Perfect Agent uses winning strategy (“P”).
# Demo Scenarios: R vs R, P vs R, P vs P
# ────────────────────────────────────────────────────────────────────────────

scenarios = [
    ("Random vs Random",   ("Random", "Random")),
    ("Perfect vs Random",  ("Perfect", "Random")),
    ("Perfect vs Perfect", ("Perfect", "Perfect")),
]

for title, (type_x, type_o) in scenarios:
    print(f"\n=== Scenario: {title} ===")

    # 1) Instantiate agents of the specified types
    agent_x = TicTacToeAgent(symbol='X', plans=plans, agent_type=type_x)
    agent_o = TicTacToeAgent(symbol='O', plans=plans, agent_type=type_o)

    # 2) Create and configure the environment
    env = TicTacToeEnvironment(
        agent_x=agent_x,
        agent_o=agent_o,
        n_games=10,     # play 10 games per scenario
        delay=0.2       # 0.2s between moves for speed
    )

    # 3) Run the simulation; the environment prints boards & scores 
    env.simulate()

    # 4) Print final aggregated scores for both agents
    print(
        f"Final Scores → X ({type_x[0]}): {agent_x.beliefs['scores']}, "
        f"O ({type_o[0]}): {agent_o.beliefs['scores']}"
    )

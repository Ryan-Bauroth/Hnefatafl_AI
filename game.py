import pygame
import sys
from pygame import gfxdraw

# Constants
BOARD_SIZE = 11
TILE_SIZE = 60
WINDOW_SIZE = BOARD_SIZE * TILE_SIZE
FPS = 60

# Color Constants
BACKGROUND_COLOR = (247, 247, 247)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (180, 180, 180)
LATEST_MOVE_COLOR = (206, 223, 159)
KILL_COLOR = (245, 232, 175)
KING_COLOR = (255, 255, 0)
CORNER_COLOR = (233, 196, 177)
PIECE_COLORS = [(0, 0, 0), BLACK, WHITE, KING_COLOR]

# Board Constants
BOARD_START = [
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 2],
    [1, 0, 0, 0, 2, 2],
    [1, 1, 0, 2, 2, 3]
]
CORNERS = [(0, 0), (0, BOARD_SIZE - 1), (BOARD_SIZE - 1, 0), (BOARD_SIZE - 1, BOARD_SIZE - 1)]
CENTER = (BOARD_SIZE // 2, BOARD_SIZE // 2)

# Load piece images and scale them to fit the tile size
attacker_img = pygame.image.load('pieces/attacker.png')
attacker_img = pygame.transform.smoothscale(attacker_img, (TILE_SIZE - 10, TILE_SIZE - 10))

defender_img = pygame.image.load('pieces/defender.png')
defender_img = pygame.transform.smoothscale(defender_img, (TILE_SIZE - 10, TILE_SIZE - 10))

king_img = pygame.image.load('pieces/king.png')
king_img = pygame.transform.smoothscale(king_img, (TILE_SIZE - 10, TILE_SIZE - 10))

# Create the Pygame window
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Hnefatafl")


class Game:
    def __init__(self):
        self.board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.recent_move_coords = {}
        self.clear_field_colors()
        self.piece_arr = []
        self.turn = 1
        self.kill_coords = []
        self.possible_moves = []
        self.winning_team = 0
        self.reward_vals = {
            1: 0,
            2: 0,
        }

    def setup_board(self):
        """
        Sets up the game board by mirroring one-fourth of the board to the other quadrants.
        The attackers are represented by 1, defenders by 2, and the king by 3.

        :return: None
        """
        for row, vals in enumerate(BOARD_START):
            for col, val in enumerate(vals):
                self.board[row][col] = val
                self.board[-row - 1][col] = val
                self.board[-row - 1][-col - 1] = val
                self.board[row][-col - 1] = val

    def reset(self):
        self.board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.recent_move_coords = {}
        self.piece_arr = []
        self.turn = 1
        self.kill_coords = []
        self.possible_moves = []
        self.winning_team = 0
        self.reward_vals = {
            1: 0,
            2: 0,
        }
        self.clear_field_colors()
        self.setup_board()
        return self.get_state_representation()

    def get_state_representation(self):
        # Flatten the board for input into a neural network
        flat_board = []
        for row in self.board:
            flat_board.extend(row)
        return flat_board

    def get_reward(self, player):
        if player == self.winning_team:
            return 1
        elif self.winning_team != 0:
            return -1
        else:
            return self.reward_vals[player]

    def is_over(self):
        if self.check_king_win():
            self.winning_team = 2
            return True
        return self.winning_team != 0

    def get_piece_possible_moves(self, col, row, piece):
        """
        Gets all the possibles moves for a specific piece

        :param col: The column index of the piece on the board.
        :param row: The row index of the piece on the board.
        :param piece: The type of the piece being evaluated.
        :return: A list of possible moves for the given piece from the given position.
        """
        moves = []
        for r in range(row - 1, -1, -1):
            if self.board[r][col] == 0:
                if piece == 3 or ((r, col) not in CORNERS and (r, col) != CENTER):
                    moves.append((r, col))
            else:
                break
        for r in range(row + 1, BOARD_SIZE):
            if self.board[r][col] == 0:
                if piece == 3 or ((r, col) not in CORNERS and (r, col) != CENTER):
                    moves.append((r, col))
            else:
                break
        for c in range(col - 1, -1, -1):
            if self.board[row][c] == 0:
                if piece == 3 or ((row, c) not in CORNERS and (row, c) != CENTER):
                    moves.append((row, c))
            else:
                break
        for c in range(col + 1, BOARD_SIZE):
            if self.board[row][c] == 0:
                if piece == 3 or ((row, c) not in CORNERS and (row, c) != CENTER):
                    moves.append((row, c))
            else:
                break

        return moves

    def get_possible_moves(self):
        all_possible_moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board[row][col] == self.turn:
                    piece_moves = self.get_piece_possible_moves(col, row, self.turn)
                    for move in piece_moves:
                        all_possible_moves.append((row, col, move[0], move[1]))
                elif self.turn == 2 and self.board[row][col] == 3:
                    piece_moves = self.get_piece_possible_moves(col, row, 3)
                    for move in piece_moves:
                        all_possible_moves.append((row, col, move[0], move[1]))

        return all_possible_moves

    def place_piece(self, grid_x, grid_y, row, col, piece):
        """
        Places a piece on the board and handles necessary updates
        -> Input goes Col, Row, Row, Col because I am a crappy designer

        :param grid_x: The x-coordinate on the grid where the piece is to be placed.
        :param grid_y: The y-coordinate on the grid where the piece is to be placed.
        :param row: The current row of the piece before being moved.
        :param col: The current column of the piece before being moved.
        :param piece: The game piece to be placed on the board.
        :return: None
        """
        # play the piece
        self.board[grid_y][grid_x] = piece
        # if the piece is not going back to its og location, change turns
        if not (grid_y == row and grid_x == col):
            self.reward_vals[self.turn] = 0
            self.board[row][col] = 0
            self.kill_coords = self.check_kills(grid_y, grid_x, piece)
            self.recent_move_coords = {
                "original_col": col,
                "original_row": row,
                "new_col": grid_x,
                "new_row": grid_y,
            }
            self.possible_moves = []
            self.turn = 3 - self.turn
            if not self.get_possible_moves():
                self.winning_team = piece

    def check_kills(self, row, col, piece):
        """
        Checks for kills

        :param row: int the most recently moved piece's row
        :param col: int the most recently moved piece's col
        :param piece: int the most recently moved piece's piece id
        """
        kill_coords = []
        enemy_piece = 1 if piece == 2 else 2
        # north direction check
        if row - 2 >= 0:
            # normal piece capture
            if self.board[row - 1][col] == enemy_piece and (
                    self.board[row - 2][col] == piece or (row - 2, col) in CORNERS):
                self.board[row - 1][col] = 0
                kill_coords.append((row - 1, col))
        # south direction check
        if row + 2 <= BOARD_SIZE - 1:
            if self.board[row + 1][col] == enemy_piece and (
                    self.board[row + 2][col] == piece or (row + 2, col) in CORNERS):
                self.board[row + 1][col] = 0
                kill_coords.append((row + 1, col))
        # west direction check
        if col - 2 >= 0:
            if self.board[row][col - 1] == enemy_piece and (
                    self.board[row][col - 2] == piece or (row, col - 2) in CORNERS):
                self.board[row][col - 1] = 0
                kill_coords.append((row, col - 1))
        # east direction check
        if col + 2 <= BOARD_SIZE - 1:
            if self.board[row][col + 1] == enemy_piece and (
                    self.board[row][col + 2] == piece or (row, col + 2) in CORNERS):
                self.board[row][col + 1] = 0
                kill_coords.append((row, col + 1))
        # king checks
        if piece == 1:
            # north king capture
            if col - 1 > 0 and col + 1 < BOARD_SIZE - 1 and row != BOARD_SIZE - 1 and row != 0:
                if self.board[row - 1][col] == 3 and (row - 2 < 0 or self.board[row - 2][col] == 1) and \
                        self.board[row - 1][col - 1] == 1 and self.board[row - 1][col + 1] == 1:
                    self.winning_team = 1
            # south king capture
            if col - 1 > 0 and col + 1 < BOARD_SIZE - 1 and row != BOARD_SIZE - 1 and row != 0:
                if self.board[row + 1][col] == 3 and (row + 2 > BOARD_SIZE - 1 or self.board[row + 2][col] == piece) and \
                        self.board[row + 1][col - 1] == 1 and self.board[row + 1][col + 1] == 1:
                    self.winning_team = 1
            # west king check
            if row - 1 > 0 and row + 1 < BOARD_SIZE - 1 and col != BOARD_SIZE - 1 and col != 0:
                if self.board[row][col - 1] == 3 and (col - 2 < 0 or self.board[row][col - 2] == piece) and \
                        self.board[row - 1][col - 1] == 1 and self.board[row + 1][col - 1] == 1:
                    self.winning_team = 1
            # east king check
            if row - 1 > 0 and row + 1 < BOARD_SIZE - 1 and col != BOARD_SIZE - 1 and col != 0:
                if self.board[row][col + 1] == 3 and (col + 2 > BOARD_SIZE - 1 or self.board[row][col + 2] == piece) and \
                        self.board[row - 1][col + 1] == 1 and self.board[row + 1][col + 1] == 1:
                    self.winning_team = 1
        if len(kill_coords) > 0:
            self.reward_vals[self.turn] += .05
            self.reward_vals[3 - self.turn] -= .05
        return kill_coords

    def check_king_win(self):
        for row, col in CORNERS:
            if self.board[row][col] == 3:
                return True
        return False

    def check_play(self, grid_y, row, grid_x, col, piece):
        """
        Checks to see if an intended play is legal

        :param grid_y: int the intended play's row
        :param row: int the moving pieces original row
        :param grid_x: int the intended play's col
        :param col: int the moving pieces original col
        """
        # if the ai makes an illegal move its bad
        self.reward_vals[self.turn] -= .5

        # pieces move like the rook in chess
        if not (grid_y == row or grid_x == col):
            return False
        # pieces cannot move onto the corners (except for kings)
        if (grid_y, grid_x) in CORNERS or (grid_y, grid_x) == CENTER:
            if piece != 3:
                return False
        # thus moving in the x direction
        if grid_y == row:
            # if moving upward
            if grid_x - col > 0:
                for c in range(col + 1, grid_x):
                    if self.board[row][c] != 0:
                        return False
            # if moving downward
            if grid_x - col < 0:
                for c in range(col - 1, grid_x, -1):
                    if self.board[row][c] != 0:
                        return False
        # thus moving in the y direction
        else:
            # if moving right
            if grid_y - row > 0:
                for r in range(row + 1, grid_y):
                    if self.board[r][col] != 0:
                        return False
            # if moving left
            if grid_y - row < 0:
                for r in range(row - 1, grid_y, -1):
                    if self.board[r][col] != 0:
                        return False
        self.reward_vals[self.turn] += .5
        return True

    def append_draw_piece(self, row, col, piece):
        """
        Updates piece arr and draws piece on the screen

        :param row: The row index where the piece is to be placed on the board.
        :param col: The column index where the piece is to be placed on the board.
        :param piece: An integer representing the type of piece to be drawn.
        :return: None
        """
        if piece != 0:
            self.piece_arr.append({
                "rect": pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
                "piece": piece,
                "row": row,
                "col": col,
            })
            pygame.gfxdraw.filled_circle(
                screen,
                col * TILE_SIZE + TILE_SIZE // 2,
                row * TILE_SIZE + TILE_SIZE // 2,
                TILE_SIZE // 2 - 9,
                PIECE_COLORS[piece],
            )
            pygame.gfxdraw.aacircle(
                screen,
                col * TILE_SIZE + TILE_SIZE // 2,
                row * TILE_SIZE + TILE_SIZE // 2,
                TILE_SIZE // 2 - 9,
                BLACK
            )

    def draw_board(self):
        # Draw the board
        pygame.draw.rect(screen, LIGHT_GRAY, (0, 0, WINDOW_SIZE, WINDOW_SIZE), 2)
        for corner in CORNERS:
            row, col = corner
            pygame.draw.rect(screen, CORNER_COLOR, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 0)
        c_row, c_col = CENTER
        pygame.draw.rect(screen, CORNER_COLOR, (c_col * TILE_SIZE, c_row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 0)
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pygame.draw.rect(screen, LIGHT_GRAY, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)
                # Draw pieces
                self.append_draw_piece(row, col, self.board[row][col])

    def draw_possible_moves(self):
        """
        Draws all possible moves for the game piece onto the screen. Each possible move
        is represented as a filled circle with an anti-aliased circle border at the positions
        calculated from the move coordinates (row, col).
        """
        for move in self.possible_moves:
            row, col = move
            pygame.gfxdraw.filled_circle(
                screen,
                col * TILE_SIZE + TILE_SIZE // 2,
                row * TILE_SIZE + TILE_SIZE // 2,
                TILE_SIZE // 2 - 20,
                KILL_COLOR,
            )
            pygame.gfxdraw.aacircle(
                screen,
                col * TILE_SIZE + TILE_SIZE // 2,
                row * TILE_SIZE + TILE_SIZE // 2,
                TILE_SIZE // 2 - 20,
                LIGHT_GRAY,
            )

    def update_field_colors(self, col, row, color):
        """
        Updates a tile on the field

        :param col: Column index of the tile to be updated.
        :param row: Row index of the tile to be updated.
        :param color: Color to be used to fill the tile.
        :return: None
        """
        pygame.draw.rect(screen, color, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 0)

    def clear_field_colors(self):
        """
        Clears the coordinates of recent moves and resets them to their default values.

        :return: None
        """
        self.recent_move_coords = {
            "original_col": -1,
            "original_row": -1,
            "new_col": -1,
            "new_row": -1,
        }

    def play_game(self):
        """
        Main method to run all game functions
        """

        # sets up game
        pygame.init()
        clock = pygame.time.Clock()
        self.setup_board()
        self.draw_board()
        self.clear_field_colors()
        is_dragging = False
        selected_piece = None
        while True:
            for event in pygame.event.get():
                # quits game
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # resets game on r key
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    piece_clicked = False
                    mouse_x, mouse_y = event.pos
                    # Check if we are clicking on a piece and sets it as current piece if so
                    for piece in self.piece_arr:
                        if piece["rect"].collidepoint(mouse_x, mouse_y) and (
                                self.turn == piece["piece"] or (piece["piece"] == 3 and self.turn == 2)):
                            self.possible_moves = []
                            self.possible_moves = self.get_piece_possible_moves(piece["col"], piece["row"],
                                                                                piece["piece"])
                            is_dragging = True
                            piece_clicked = True
                            selected_piece = piece
                            self.board[selected_piece["row"]][selected_piece["col"]] = 0
                            break
                    # allows user to click on location for a move to move a piece
                    if not piece_clicked:
                        if selected_piece is not None:
                            grid_x = mouse_x // TILE_SIZE
                            grid_y = mouse_y // TILE_SIZE
                            row = selected_piece["row"]
                            col = selected_piece["col"]
                            piece = selected_piece["piece"]
                            if (grid_y, grid_x) in self.possible_moves:
                                self.place_piece(grid_x, grid_y, row, col, piece)
                        self.possible_moves = []

                elif event.type == pygame.MOUSEBUTTONUP:
                    # if the user was dragging a piece, places it
                    if is_dragging:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        row = selected_piece["row"]
                        col = selected_piece["col"]
                        piece = selected_piece["piece"]
                        grid_x = mouse_x // TILE_SIZE
                        grid_y = mouse_y // TILE_SIZE
                        # If the user is playing in a empty spot
                        if self.board[grid_y][grid_x] == 0 and self.check_play(grid_y, row, grid_x, col, piece):
                            self.place_piece(grid_x, grid_y, row, col, piece)
                        else:
                            self.board[row][col] = piece
                    is_dragging = False

            # updates board
            self.piece_arr = []
            screen.fill(BACKGROUND_COLOR)
            if self.recent_move_coords["original_row"] != -1:
                self.update_field_colors(self.recent_move_coords["original_col"],
                                         self.recent_move_coords["original_row"], LATEST_MOVE_COLOR)
                self.update_field_colors(self.recent_move_coords["new_col"], self.recent_move_coords["new_row"],
                                         LATEST_MOVE_COLOR)
            for kill_cord in self.kill_coords:
                x, y = kill_cord
                self.update_field_colors(y, x, KILL_COLOR)
            self.draw_board()
            self.draw_possible_moves()

            # draws dragged piece
            if is_dragging:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                pygame.gfxdraw.filled_circle(
                    screen,
                    mouse_x,
                    mouse_y,
                    TILE_SIZE // 2 - 9,
                    PIECE_COLORS[selected_piece["piece"]],
                )
                pygame.gfxdraw.aacircle(
                    screen,
                    mouse_x,
                    mouse_y,
                    TILE_SIZE // 2 - 9,
                    BLACK
                )

            # checks if player 2 wins
            if self.check_king_win() and self.winning_team != 2:
                self.winning_team = 2

            pygame.display.flip()
            clock.tick(FPS)


if __name__ == "__main__":
    app = Game()
    app.play_game()
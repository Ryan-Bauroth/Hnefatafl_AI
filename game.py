import pygame
import sys
from pygame import gfxdraw

# Constants
BOARD_SIZE = 11
TILE_SIZE = 60
WINDOW_SIZE = BOARD_SIZE * TILE_SIZE
FPS = 60

# Color Constants
BACKGROUND_COLOR = (247,247,247)
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
    [0,0,0,1,1,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,0],
    [1,0,0,0,0,2],
    [1,0,0,0,2,2],
    [1,1,0,2,2,3]
]
CORNERS = [(0, 0), (0, BOARD_SIZE-1), (BOARD_SIZE-1, 0), (BOARD_SIZE-1, BOARD_SIZE-1)]
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
        self.piece_arr = []
        self.turn = 1
        self.kill_coords = []
        self.possible_moves = []
    def play_game(self):
        pygame.init()
        clock = pygame.time.Clock()
        self.setup_board()
        self.draw_board()
        self.clear_field_colors()
        is_dragging = False
        selected_piece = None
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    piece_clicked = False
                    mouse_x, mouse_y = event.pos
                    # Check if we are clicking on a piece
                    for piece in self.piece_arr:
                        if piece["rect"].collidepoint(mouse_x, mouse_y) and (self.turn == piece["piece"] or (piece["piece"] == 3 and self.turn == 2)):
                            self.update_possible_moves(piece["col"], piece["row"], piece["piece"])
                            is_dragging = True
                            piece_clicked = True
                            selected_piece = piece
                            self.board[selected_piece["row"]][selected_piece["col"]] = 0
                            break
                    if not piece_clicked:
                        if selected_piece is not None:
                            grid_x = mouse_x // TILE_SIZE
                            grid_y = mouse_y // TILE_SIZE
                            row = selected_piece["row"]
                            col = selected_piece["col"]
                            piece = selected_piece["piece"]
                            if (grid_y, grid_x) in self.possible_moves:
                                #if self.board[grid_y][grid_x] == 0 and self.check_play(grid_y, row, grid_x, col, piece):
                                self.place_piece(grid_x, grid_y, row, col, piece)
                        self.possible_moves = []
                elif event.type == pygame.MOUSEBUTTONUP:
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
            self.piece_arr = []
            screen.fill(BACKGROUND_COLOR)
            if self.recent_move_coords["original_row"] != -1:
                self.update_field_colors(self.recent_move_coords["original_col"], self.recent_move_coords["original_row"], LATEST_MOVE_COLOR)
                self.update_field_colors(self.recent_move_coords["new_col"], self.recent_move_coords["new_row"], LATEST_MOVE_COLOR)
            for kill_cord in self.kill_coords:
                x, y = kill_cord
                self.update_field_colors(y, x, KILL_COLOR)
            self.draw_board()
            self.draw_possible_moves()
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
            # Draw the piece
            pygame.display.flip()
            clock.tick(FPS)
    def setup_board(self):
        # Set up board by mirroring 1/4th board (Attackers = 1, Defenders = 2, King = 3)
        for row, vals in enumerate(BOARD_START):
            for col, val in enumerate(vals):
                self.board[row][col] = val
                self.board[-row-1][col] = val
                self.board[-row-1][-col-1] = val
                self.board[row][-col-1] = val
    def append_draw_piece(self, row, col, piece):
        if piece != 0:
            # Updates piece arr and draws piece on the screen
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
        if row-2 >= 0:
            # normal piece capture
            if self.board[row-1][col] == enemy_piece and (self.board[row-2][col] == piece or (row-2, col) in CORNERS):
                self.board[row-1][col] = 0
                kill_coords.append((row-1, col))
        # south direction check
        if row+2 <= BOARD_SIZE-1:
            if self.board[row+1][col] == enemy_piece and (self.board[row+2][col] == piece or (row+2, col) in CORNERS):
                self.board[row+1][col] = 0
                kill_coords.append((row+1, col))
        # west direction check
        if col-2 >= 0:
            if self.board[row][col-1] == enemy_piece and (self.board[row][col-2] == piece or (row, col-2) in CORNERS):
                self.board[row][col-1] = 0
                kill_coords.append((row, col-1))
        # east direction check
        if col+2 <= BOARD_SIZE-1:
            if self.board[row][col + 1] == enemy_piece and (self.board[row][col + 2] == piece or (row, col + 2) in CORNERS):
                self.board[row][col+1] = 0
                kill_coords.append((row, col+1))
        # king checks
        if piece == 1:
            # north king capture
            if col-1 > 0 and col+1 < BOARD_SIZE-1 and row != BOARD_SIZE-1 and row != 0:
                if self.board[row-1][col] == 3 and (row-2 < 0 or self.board[row-2][col] == 1) and self.board[row-1][col-1] == 1 and self.board[row-1][col+1] == 1:
                    print("ATTACKERS WIN")
            # south king capture
            if col-1 > 0 and col+1 < BOARD_SIZE-1 and row != BOARD_SIZE-1 and row != 0:
                if self.board[row+1][col] == 3 and (row+2 < BOARD_SIZE or self.board[row+2][col] == piece) and self.board[row+1][col-1] == 1 and self.board[row+1][col+1] == 1:
                    print("ATTACKERS WIN")
            # west king check
            if row-1 > 0 and row+1 < BOARD_SIZE-1 and col != BOARD_SIZE-1 and col != 0:
                if self.board[row][col-1] == 3 and (col-2 < 0 or self.board[row][col-2] == piece) and self.board[row-1][col-1] == 1 and self.board[row+1][col-1] == 1:
                    print("ATTACKERS WIN")
            # east king check
            if row-1 > 0 and row+1 < BOARD_SIZE-1 and col != BOARD_SIZE-1 and col != 0:
                if self.board[row][col+1] == 3 and (col+2 > BOARD_SIZE-1 or self.board[row][col+2] == piece) and self.board[row-1][col+1] == 1 and self.board[row+1][col+1] == 1:
                    print("ATTACKERS WIN")
        return kill_coords
    def check_play(self, grid_y, row, grid_x, col, piece):
        """
        Checks to see if an intended play is legal
    
        :param grid_y: int the intended play's row
        :param row: int the moving pieces original row
        :param grid_x: int the intended play's col
        :param col: int the moving pieces original col
        """
        # pieces move like the rook in chess
        if not (grid_y == row or grid_x == col):
            return False
        # pieces cannot move onto the corners (except for kings)
        if (grid_y, grid_x) in CORNERS or (grid_y, grid_x) == CENTER:
            if piece != 3:
                return False
            elif (grid_y, grid_x) in CORNERS:
                print("DEFENDERS WIN")
                return True
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
        return True
    def update_field_colors(self, col, row, color):
        pygame.draw.rect(screen, color, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 0)
    def clear_field_colors(self):
        self.recent_move_coords = {
            "original_col": -1,
            "original_row": -1,
            "new_col": -1,
            "new_row": -1,
        }
    def update_possible_moves(self, col, row, piece):
        self.possible_moves = []
        for r in range(row - 1, -1, -1):
            if self.board[r][col] == 0:
                if piece == 3 or ((r, col) not in CORNERS and (r, col) != CENTER):
                    self.possible_moves.append((r, col))
            else:
                break
        for r in range(row + 1, BOARD_SIZE):
            if self.board[r][col] == 0:
                if piece == 3 or ((r, col) not in CORNERS and (r, col) != CENTER):
                    self.possible_moves.append((r, col))
            else:
                break
        for c in range(col - 1, -1, -1):
            if self.board[row][c] == 0:
                if piece == 3 or ((row, c) not in CORNERS and (row, c) != CENTER):
                    self.possible_moves.append((row, c))
            else:
                break
        for c in range(col + 1, BOARD_SIZE):
            if self.board[row][c] == 0:
                if piece == 3 or ((row, c) not in CORNERS and (row, c) != CENTER):
                    self.possible_moves.append((row, c))
            else:
                break
    def draw_possible_moves(self):
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
    def place_piece(self, grid_x, grid_y, row, col, piece):
        # play the piece
        self.board[grid_y][grid_x] = piece
        # if the piece is not going back to its og location, change turns
        if not (grid_y == row and grid_x == col):
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

if __name__ == "__main__":
    app = Game()
    app.play_game()
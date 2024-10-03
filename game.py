import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 11
TILE_SIZE = 60
WINDOW_SIZE = BOARD_SIZE * TILE_SIZE
FPS = 60

# Color Constants
BACKGROUND_COLOR = (210, 79, 51)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ATTACKER_COLOR = BLACK
DEFENDER_COLOR = WHITE
KING_COLOR = (255, 255, 0)
PIECE_COLORS = [WHITE, ATTACKER_COLOR, DEFENDER_COLOR, KING_COLOR]

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
ADJUSTMENTS = [[5, 5, -6, -6], [0, 5, -5, -6], [5, 0, -6, -5], [0, 0, -5, -5]]

# Create the Pygame window
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Hnefatafl")

# Game Vars
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
piece_arr = []
turn = 1

# Set up board by mirroring 1/4th board (Attackers = 1, Defenders = 2, King = 3)
def setup_board():
    for row, vals in enumerate(BOARD_START):
        for col, val in enumerate(vals):
            board[row][col] = val
            board[-row-1][col] = val
            board[-row-1][-col-1] = val
            board[row][-col-1] = val

# Updates piece arr and draws piece on the screen
def append_draw_piece(row, col, piece):
    piece_arr.append({
        "rect": pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE),
        "piece": piece,
        "row": row,
        "col": col,
    })
    pygame.draw.circle(
        screen,
        PIECE_COLORS[piece],
        (col * TILE_SIZE + TILE_SIZE // 2, row * TILE_SIZE + TILE_SIZE // 2),
        TILE_SIZE // 2 - 5
    )

# Draw the board
def draw_board():
    screen.fill(BACKGROUND_COLOR)
    pygame.draw.rect(screen, BLACK, (0, 0, WINDOW_SIZE, WINDOW_SIZE), 5)
    for corner, adjustment in zip(CORNERS, ADJUSTMENTS):
        row, col = corner
        pygame.draw.rect(screen, (170, 53, 28), (col * TILE_SIZE + adjustment[0], row * TILE_SIZE + adjustment[1], TILE_SIZE + adjustment[2], TILE_SIZE + adjustment[3]), 0)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            pygame.draw.rect(screen, BLACK, (col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)

            # Draw pieces
            if board[row][col] == 1:  # Attackers
                append_draw_piece(row, col, 1)
            elif board[row][col] == 2:  # Defenders
                append_draw_piece(row, col, 2)
            elif board[row][col] == 3:  # King
                append_draw_piece(row, col, 3)

"""
Checks for kills

:param row: int the most recently moved piece's row
:param col: int the most recently moved piece's col
:param piece: int the most recently moved piece's piece id
"""
def check_kills(row, col, piece):
    enemy_piece = 1 if piece == 2 else 2
    # north direction check
    if row - 2 >= 0:
        # normal piece capture
        if board[row - 1][col] == enemy_piece and (board[row - 2][col] == piece or (row - 2, col) in CORNERS):
            board[row - 1][col] = 0
    # north king capture
    if col - 1 > 0 and col + 1 < BOARD_SIZE - 1 and row != BOARD_SIZE - 1 and row != 0:
        if piece == 1 and board[row - 1][col] == 3 and (row - 2 < 0 or board[row - 2][col] == piece) and board[row - 1][col - 1] == piece and board[row - 1][col + 1] == piece:
            print("ATTACKERS WIN")
    # south direction check
    if row + 2 <= BOARD_SIZE - 1:
        if board[row + 1][col] == enemy_piece and (board[row + 2][col] == piece or (row + 2, col) in CORNERS):
            board[row + 1][col] = 0
    # south king capture
    if col - 1 > 0 and col + 1 < BOARD_SIZE - 1 and row != BOARD_SIZE - 1 and row != 0:
        if piece == 1 and board[row + 1][col] == 3 and (row + 2 < BOARD_SIZE or board[row + 2][col] == piece) and \
                board[row + 1][col - 1] == piece and board[row + 1][col + 1] == piece:
            print("ATTACKERS WIN")
    # west direction check
    if col - 2 >= 0:
        if board[row][col - 1] == enemy_piece and (board[row][col - 2] == piece or (row, col - 2) in CORNERS):
            board[row][col - 1] = 0
    # west king check
    if row - 1 > 0 and row + 1 < BOARD_SIZE - 1 and col != BOARD_SIZE - 1 and col != 0:
        if piece == 1 and board[row][col - 1] == 3 and (col - 2 < 0 or board[row][col - 2] == piece) and \
                board[row - 1][col - 1] == piece and board[row + 1][col - 1] == piece:
            print("ATTACKERS WIN")
    # east direction check
    if col + 2 <= BOARD_SIZE - 1:
        if board[row][col + 1] == enemy_piece and (board[row][col + 2] == piece or (row, col + 2) in CORNERS):
            board[row][col + 1] = 0
    # east king check
    if row - 1 > 0 and row + 1 < BOARD_SIZE - 1 and col != BOARD_SIZE - 1 and col != 0:
        if piece == 1 and board[row][col + 1] == 3 and (col + 2 > BOARD_SIZE - 1 or board[row][col + 2] == piece) and \
                board[row - 1][col + 1] == piece and board[row + 1][col + 1] == piece:
            print("ATTACKERS WIN")

"""
Checks to see if an intended play is legal

:param grid_y: int the intended play's row
:param row: int the moving pieces original row
:param grid_x: int the intended play's col
:param col: int the moving pieces original col
"""
def check_play(grid_y, row, grid_x, col, piece):
    # pieces move like the rook in chess
    if not (grid_y == row or grid_x == col):
        return False
    # pieces cannot move onto the corners (except for kings)
    if (grid_y, grid_x) in CORNERS:
        if piece != 3:
            return False
        else:
            print("DEFENDERS WIN")
            return True
    # thus moving in the x direction
    if grid_y == row:
        # if moving upward
        if grid_x - col > 0:
            for c in range(col + 1, grid_x):
                if board[row][c] != 0:
                    return False
        # if moving downward
        if grid_x - col < 0:
            for c in range(col - 1, grid_x, -1):
                if board[row][c] != 0:
                    return False
    # thus moving in the y direction
    else:
        # if moving right
        if grid_y - row > 0:
            for r in range(row + 1, grid_y):
                if board[r][col] != 0:
                    return False
        # if moving left
        if grid_y - row < 0:
            for r in range(row - 1, grid_y, -1):
                if board[r][col] != 0:
                    return False
    return True

"""   
Game loop

Piece movement code assisted by AI
"""
clock = pygame.time.Clock()
def main():
    setup_board()
    draw_board()

    global piece_arr
    global turn

    is_dragging = False
    selected_piece = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos

                # Check if we are clicking on a piece
                for piece in piece_arr:
                    if piece["rect"].collidepoint(mouse_x, mouse_y) and (turn == piece["piece"] or (piece["piece"] == 3 and turn == 2)):
                        is_dragging = True
                        selected_piece = piece
                        board[selected_piece["row"]][selected_piece["col"]] = 0
                        break

            elif event.type == pygame.MOUSEBUTTONUP:
                if is_dragging:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    row = selected_piece["row"]
                    col = selected_piece["col"]
                    piece = selected_piece["piece"]
                    grid_x = mouse_x // TILE_SIZE
                    grid_y = mouse_y // TILE_SIZE

                    # If the user is playing in a empty spot
                    if board[grid_y][grid_x] == 0 and check_play(grid_y, row, grid_x, col, piece):
                        # play the piece
                        board[grid_y][grid_x] = piece
                        # if the piece is not going back to its og location, change turns
                        if not (grid_y == row and grid_x == col):
                            check_kills(grid_y, grid_x, piece)
                            turn = 1 if turn == 2 else 2
                    else:
                        # if the user is playing in a full spot, return the piece to its og location
                        board[row][col] = piece
                is_dragging = False

        piece_arr = []
        draw_board()

        if is_dragging:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            selected_piece["rect"].center = (mouse_x, mouse_y)
            pygame.draw.circle(
                screen,
                PIECE_COLORS[selected_piece["piece"]],
                selected_piece["rect"].center,
                TILE_SIZE // 2 - 5,
            )

        # Draw the piece
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()